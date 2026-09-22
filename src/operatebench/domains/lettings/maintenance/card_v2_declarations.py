# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Owner declarations for the V1 Maintenance Card-v2 identity profile.

This leaf is deliberately not a Card builder.  It records only Maintenance-owned
facts that have no earlier executable owner, plus source reconciliation descriptors.
Existing event, action, policy, retrieval, state, point, and variant tables remain in
their current modules and fixture.  A later projector must reconcile those owners and
must refuse while :attr:`MaintenanceCardIdentityProfile.projectable` is false.

The accessor returns a new recursively immutable graph on every call.  The graph is
intentionally not pickle/deepcopy serializable: its stable interchange form is the
detached JSON-safe value returned by :meth:`MaintenanceCardIdentityProfile.as_plain`.
As required by Python, importing this leaf first initializes ``operatebench``; that
package-root behavior is pre-existing and is not a dependency of this declaration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, SupportsIndex


class ReadinessStatus(StrEnum):
    """The exhaustive ownership classification for a future Card-required fact."""

    EXISTING_OWNER = "EXISTING_OWNER"
    OWNER_DECLARATION = "OWNER_DECLARATION"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class ReadinessItem:
    fact_id: str
    status: ReadinessStatus
    owner: str
    proof: str


@dataclass(frozen=True)
class AliasDeclaration:
    scope: str
    source: str
    card: str


@dataclass(frozen=True)
class ReconciliationDescriptor:
    expected_event_types: int
    expected_event_payload_fields: int
    expected_actions: int
    expected_retrieval_tools: int
    expected_semantic_points: int
    expected_exactly_one_clauses: int
    expected_v1_events: int
    existing_owner_modules: tuple[str, ...]
    rule: str


@dataclass(frozen=True)
class TimestampPolicy:
    runtime_form: str
    card_form: str
    forward_rule: str
    inverse_rule: str
    runtime_input_rule: str


@dataclass(frozen=True)
class RolePolicy:
    actor_role_overrides: Mapping[str, str]
    preservation_rule: str
    authority_source: str
    unknown_profile_behavior: str


@dataclass(frozen=True)
class FieldShapePolicy:
    integer_minimum: int
    integer_maximum: int
    integer_minimum_rule: str
    identifier_min_length: int
    identifier_max_length: int
    authored_text_min_length: int
    authored_text_max_length: int
    digest_pattern: str
    currency_pattern: str
    closed_enum_rule: str
    nullable_string_paths: tuple[str, ...]
    out_of_profile_behavior: str
    runtime_version_behavior: str


@dataclass(frozen=True)
class StatePolicy:
    schema_id: str
    canonical_state_projection: tuple[str, ...]
    replay_final_pointer: str
    terminal_pointer: str
    additional_properties: bool
    schema_construction_contract: str
    hidden_fixture_input_kind: str
    construction_rule: str


@dataclass(frozen=True)
class SurfacePolicy:
    algorithm_id: str
    hash: str
    domain: str
    canonical_object_fields: tuple[str, ...]
    accepted_source_grammar: str
    target_representation_template: str
    framing_rule: str
    text_rule: str
    bytes_rule: str
    binding_rule: str
    refusal_rules: tuple[str, ...]


@dataclass(frozen=True)
class SemanticScenarioPolicy:
    scenario_id: str
    construct_label: str
    common_estimand_source: str
    point_ids: tuple[str, ...]
    required_prior_point_ids: tuple[tuple[str, ...], ...]
    obligation_ids: tuple[str, ...]
    predicate_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    causal_nodes: tuple[tuple[str, str], ...]
    causal_edges: tuple[tuple[str, str, str], ...]
    causal_edge_rationales: tuple[tuple[str, str, str, str], ...]
    hazard_ids: tuple[str, ...]
    hazard_finding_sources: tuple[tuple[str, str], ...]
    allowed_dimensions: tuple[tuple[str, str, bool], ...]
    exactly_one_clause_count: int
    reach_predicate_rule: str
    control_transforms: str


@dataclass(frozen=True)
class V1VariantPolicy:
    variant_label: str
    source_scenario_id: str
    card_scenario_id: str
    expected_terminal: str
    human_checkpoint_budget: int
    required_checkpoint_types: tuple[str, ...]
    dispatch_failures: tuple[str, ...]
    expected_rejections: tuple[tuple[str, str], ...]
    event_order_rule: str
    realization_invariants: tuple[str, ...]
    actor_event_fixture_timing_source: str
    hidden_state_construction: str


@dataclass(frozen=True)
class MaintenanceCardIdentityProfile:
    profile_id: str
    coverage: str
    projectable: bool
    aliases: tuple[AliasDeclaration, ...]
    timestamp_policy: TimestampPolicy
    role_policy: RolePolicy
    field_shape_policy: FieldShapePolicy
    state_policy: StatePolicy
    surface_policy: SurfacePolicy
    semantic_scenario: SemanticScenarioPolicy
    v1_variant: V1VariantPolicy
    deferred_scenarios: tuple[str, ...]
    reconciliation: ReconciliationDescriptor
    readiness_inventory: tuple[ReadinessItem, ...]

    def as_plain(self) -> dict[str, Any]:
        """Return the sole stable serialization form as detached JSON-safe builtins."""
        value = _plain(self)
        if not isinstance(value, dict):  # pragma: no cover - structural invariant
            raise TypeError("Maintenance profile did not serialize to an object")
        return value

    def __copy__(self) -> MaintenanceCardIdentityProfile:
        return self

    def __deepcopy__(self, memo: object) -> MaintenanceCardIdentityProfile:
        raise TypeError("use as_plain() for a detached Maintenance declaration")

    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[Any, ...]:
        raise TypeError("Maintenance declarations are not pickle interchange objects")


def _plain(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, frozenset):
        return sorted(_plain(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str):
            value.encode("utf-8", "strict")
        return value
    raise TypeError(f"declaration contains non-JSON-safe {type(value).__name__}")


def _aliases() -> tuple[AliasDeclaration, ...]:
    return (
        AliasDeclaration(
            "issue_classification",
            "NON_EMERGENCY_RECURRING_LEAK",
            "non_emergency_recurring_leak",
        ),
        AliasDeclaration("scenario_id", "V1", "maintenance.variant.approved_reopened.v1"),
        AliasDeclaration("rejection_code", "AFTER_REPLAY_FINAL", "after_replay_final"),
    )


def _roles() -> RolePolicy:
    overrides = {
        "maintenance_system": "maintenance_verification_system",
        "payment_system": "payment_authority_system",
    }
    return RolePolicy(
        actor_role_overrides=MappingProxyType(overrides),
        preservation_rule="PRESERVE_SOURCE_ROLE_UNLESS_EXPLICIT_OVERRIDE",
        authority_source="RECONCILE_ACTUAL_OPERATION_SPEC_ACTOR_AUTHORITY",
        unknown_profile_behavior="REFUSE_NO_AUTHORITY_UNION_OR_FALLBACK",
    )


_POINT_IDS = (
    "initial_report",
    "approval_granted",
    "work_verified",
    "invoice_validated",
    "warranty_reopened",
)
_HAZARD_IDS = (
    "maintenance.hazard.premature_invoice",
    "maintenance.hazard.unverified_supplier_assertion",
    "maintenance.hazard.post_final_mutation",
)


def _semantic() -> SemanticScenarioPolicy:
    obligation_ids = tuple(f"maintenance.obligation.{point}" for point in _POINT_IDS)
    predicate_ids = tuple(f"maintenance.predicate.{point}" for point in _POINT_IDS)
    causal_nodes = tuple((point, "decision_point") for point in _POINT_IDS) + tuple(
        (hazard, "authored_hazard") for hazard in _HAZARD_IDS
    )
    causal_edges = (
        ("initial_report", "approval_granted", "ENABLES"),
        ("approval_granted", "work_verified", "ENABLES"),
        ("work_verified", "invoice_validated", "ENABLES"),
        ("invoice_validated", "warranty_reopened", "ENABLES"),
        ("maintenance.hazard.premature_invoice", "invoice_validated", "PRECEDES"),
        (
            "maintenance.hazard.unverified_supplier_assertion",
            "work_verified",
            "PRECEDES",
        ),
        ("warranty_reopened", "maintenance.hazard.post_final_mutation", "PRECEDES"),
    )
    return SemanticScenarioPolicy(
        scenario_id="maintenance.recurring_leak.approved_reopened.v1",
        construct_label="maintenance.recurring_leak.boundary_resolution",
        common_estimand_source="semantic_arms_v1.COMMON_ESTIMAND",
        point_ids=_POINT_IDS,
        required_prior_point_ids=(
            (),
            ("initial_report",),
            ("approval_granted",),
            ("work_verified",),
            ("invoice_validated",),
        ),
        obligation_ids=obligation_ids,
        predicate_ids=predicate_ids,
        node_ids=_POINT_IDS + _HAZARD_IDS,
        causal_nodes=causal_nodes,
        causal_edges=causal_edges,
        causal_edge_rationales=(
            (
                "maintenance.hazard.premature_invoice",
                "invoice_validated",
                "PRECEDES",
                "A premature invoice is the hazard immediately preceding the invoice "
                "validation boundary that must reject it without prior evidence.",
            ),
            (
                "maintenance.hazard.unverified_supplier_assertion",
                "work_verified",
                "PRECEDES",
                "An unverified supplier assertion is the hazard immediately preceding "
                "the work verification boundary that must not treat it as authority.",
            ),
            (
                "warranty_reopened",
                "maintenance.hazard.post_final_mutation",
                "PRECEDES",
                "The warranty-reopened finality point precedes the post-final mutation "
                "hazard whose attempted change must remain ineffective.",
            ),
        ),
        hazard_ids=_HAZARD_IDS,
        hazard_finding_sources=(
            (
                "maintenance.hazard.premature_invoice",
                "evaluator.CRITICAL_INVOICE_VALIDATED_WITHOUT_PRIOR_EVIDENCE",
            ),
            (
                "maintenance.hazard.unverified_supplier_assertion",
                "evaluator.CLAIM_TREATED_AS_AUTHORITATIVE",
            ),
            (
                "maintenance.hazard.post_final_mutation",
                "evaluator.CRITICAL_MUTATION_AFTER_TERMINAL",
            ),
        ),
        allowed_dimensions=(
            ("absolute_start", "TIMESTAMP", False),
            ("message_surface", "FIXTURE_DIGEST", False),
        ),
        exactly_one_clause_count=8,
        reach_predicate_rule=(
            "MECHANICAL_OWNER_AST_TRANSLATION_PRESERVING_EXACTLY_ONE_BINDING_AND_TRUTH_VECTOR"
        ),
        control_transforms="EXCLUDED_ARM_PROTOCOL_OWNER",
    )


_REQUIRED_CARD_V2_LEAF_FACTS = (
    # Operation Card envelope and content leaves from the frozen schema census.
    "operation.card.schema",
    "operation.card.schema_version",
    "operation.card.id",
    "operation.card.content_digest_sha256",
    "operation.operation_type",
    "operation.operation_version",
    "operation.state_contract.schema_id",
    "operation.state_contract.initial_state_schema",
    "operation.state_contract.canonical_state_projection",
    "operation.state_contract.replay_final_pointer",
    "operation.state_contract.terminal_pointer",
    "operation.authority_contract.roles.role_id",
    "operation.authority_contract.roles.authorities",
    "operation.authority_contract.event_authority.event_type",
    "operation.authority_contract.event_authority.required_authority",
    "operation.authority_contract.action_authority.action_type",
    "operation.authority_contract.action_authority.required_authority",
    "operation.policy_contract.schema",
    "operation.policy_contract.currency",
    "operation.policy_contract.issue_classification",
    "operation.policy_contract.approval_threshold_minor",
    "operation.policy_contract.approval_reminder_after_minutes",
    "operation.policy_contract.approval_deadline_after_minutes",
    "operation.policy_contract.approval_validity_minutes",
    "operation.policy_contract.provisional_close_minutes",
    "operation.policy_contract.visit_followup_after_minutes",
    "operation.policy_contract.completion_notice_within_minutes",
    "operation.policy_contract.max_turns_per_invocation",
    "operation.policy_contract.horizon_minutes",
    "operation.policy_contract.max_retrieval_batches_per_invocation",
    "operation.event_contracts.event_type",
    "operation.event_contracts.required",
    "operation.event_contracts.optional",
    "operation.action_contracts.action_type",
    "operation.action_contracts.required",
    "operation.action_contracts.optional",
    "operation.action_contracts.evidence_refs",
    "operation.retrieval_contract.catalogue.tool_id",
    "operation.retrieval_contract.catalogue.source",
    "operation.retrieval_contract.catalogue.authority",
    "operation.retrieval_contract.catalogue.schema_id",
    "operation.retrieval_contract.catalogue.arguments",
    "operation.retrieval_contract.record_identity_algorithm",
    "operation.retrieval_contract.evidence_requirements.outcome_type",
    "operation.retrieval_contract.evidence_requirements.selector_id",
    "operation.retrieval_contract.evidence_requirements.required_tools",
    "operation.transition_contract.reducer_ids",
    "operation.transition_contract.obligation_types",
    "operation.transition_contract.terminal_kinds",
    "operation.transition_contract.common_predicate_inputs",
    "operation.transition_contract.declared_max_domain_generated_audit_events",
    # Semantic Scenario Card envelope and content leaves.
    "semantic.card.schema",
    "semantic.card.schema_version",
    "semantic.card.id",
    "semantic.card.content_digest_sha256",
    "semantic.operation_core",
    "semantic.construct_label",
    "semantic.common_estimand_id",
    "semantic.decision_points.point_id",
    "semantic.decision_points.order",
    "semantic.decision_points.reach_predicate",
    "semantic.decision_points.required_prior_point_ids",
    "semantic.decision_points.common_obligation_id",
    "semantic.decision_points.admissible_effect_classes",
    "semantic.causal_graph.nodes.node_id",
    "semantic.causal_graph.nodes.semantic_type",
    "semantic.causal_graph.edges.cause_node_id",
    "semantic.causal_graph.edges.effect_node_id",
    "semantic.causal_graph.edges.relation",
    "semantic.authored_hazards.hazard_id",
    "semantic.authored_hazards.predicate",
    "semantic.authored_hazards.expected_common_finding_code",
    "semantic.common_outcome_obligations.obligation_id",
    "semantic.common_outcome_obligations.predicate_id",
    "semantic.common_outcome_obligations.terminal_required",
    "semantic.allowed_variant_dimensions.dimension_id",
    "semantic.allowed_variant_dimensions.value_type",
    "semantic.allowed_variant_dimensions.changes_semantics",
    "variant.card.schema",
    "variant.card.schema_version",
    "variant.card.id",
    "variant.card.content_digest_sha256",
    "variant.semantic_scenario",
    "variant.variant_label",
    "variant.starts_at",
    "variant.expected_terminal",
    "variant.human_checkpoint_budget",
    "variant.required_checkpoint_types",
    "variant.dispatch_failures",
    "variant.expected_event_rejections.event_id",
    "variant.expected_event_rejections.code",
    "variant.actors.actor_id",
    "variant.actors.role_id",
    "variant.hidden_initial_values",
    "variant.message_fixtures.fixture_id",
    "variant.message_fixtures.surface_content_digest_sha256",
    "variant.events.event_id",
    "variant.events.event_type",
    "variant.events.actor_id",
    "variant.events.authored_sequence",
    "variant.events.at",
    "variant.events.trigger",
    "variant.events.trigger.kind",
    "variant.events.trigger.type_name",
    "variant.events.trigger.correlation",
    "variant.events.delay_minutes",
    "variant.events.triggers_agent",
    "variant.events.payload",
)


def _readiness() -> tuple[ReadinessItem, ...]:
    existing_ids = frozenset(
        {
            "operation.operation_type",
            "operation.operation_version",
            "operation.authority_contract.roles.role_id",
            "operation.authority_contract.roles.authorities",
            "operation.authority_contract.event_authority.event_type",
            "operation.authority_contract.event_authority.required_authority",
            "operation.authority_contract.action_authority.action_type",
            "operation.authority_contract.action_authority.required_authority",
            "operation.event_contracts.event_type",
            "operation.event_contracts.required",
            "operation.event_contracts.optional",
            "operation.action_contracts.action_type",
            "operation.action_contracts.required",
            "operation.action_contracts.optional",
            "operation.retrieval_contract.catalogue.tool_id",
            "operation.retrieval_contract.catalogue.source",
            "operation.retrieval_contract.catalogue.authority",
            "operation.retrieval_contract.catalogue.schema_id",
            "operation.retrieval_contract.catalogue.arguments",
            "operation.retrieval_contract.evidence_requirements.outcome_type",
            "operation.retrieval_contract.evidence_requirements.required_tools",
            "operation.transition_contract.terminal_kinds",
            "semantic.operation_core",
            "semantic.common_estimand_id",
            "variant.semantic_scenario",
            "variant.expected_terminal",
            "variant.human_checkpoint_budget",
            "variant.required_checkpoint_types",
            "variant.dispatch_failures",
            "variant.expected_event_rejections.event_id",
            "variant.expected_event_rejections.code",
            "variant.actors.actor_id",
            "variant.hidden_initial_values",
            "variant.message_fixtures.fixture_id",
            "variant.events.event_id",
            "variant.events.event_type",
            "variant.events.actor_id",
            "variant.events.authored_sequence",
            "variant.events.delay_minutes",
            "variant.events.triggers_agent",
            "variant.events.payload",
        }
        | {
            fact
            for fact in _REQUIRED_CARD_V2_LEAF_FACTS
            if fact.startswith("operation.policy_contract.")
        }
    )
    unresolved_ids = frozenset(
        {
            "operation.state_contract.initial_state_schema",
            "operation.action_contracts.evidence_refs",
            "operation.retrieval_contract.record_identity_algorithm",
            "operation.retrieval_contract.evidence_requirements.selector_id",
            "operation.transition_contract.reducer_ids",
            "operation.transition_contract.obligation_types",
            "operation.transition_contract.common_predicate_inputs",
            "operation.transition_contract.declared_max_domain_generated_audit_events",
            "semantic.decision_points.reach_predicate",
            "semantic.decision_points.admissible_effect_classes",
            "semantic.causal_graph.nodes.semantic_type",
            "semantic.authored_hazards.predicate",
            "semantic.common_outcome_obligations.terminal_required",
        }
    )
    items = [
        ReadinessItem(
            fact,
            (
                ReadinessStatus.EXISTING_OWNER
                if fact in existing_ids
                else (
                    ReadinessStatus.UNRESOLVED
                    if fact in unresolved_ids
                    else ReadinessStatus.OWNER_DECLARATION
                )
            ),
            (
                "UNRESOLVED"
                if fact in unresolved_ids
                else ("source owner" if fact in existing_ids else "profile")
            ),
            (
                "required Card-v2 leaf lacks a complete schema-compatible owner fact"
                if fact in unresolved_ids
                else "frozen schema census and exact source/declaration reconciliation"
            ),
        )
        for fact in _REQUIRED_CARD_V2_LEAF_FACTS
    ]
    unresolved: list[tuple[str, str]] = []

    unresolved.extend(
        (
            f"semantic.decision_points.admissible_effect_classes.{point}",
            "must reconcile to the source expected action/obligation; no class "
            "is guessed",
        )
        for point in _POINT_IDS
    )
    unresolved.extend(
        (
            f"semantic.common_predicate_ast.{point}",
            "no owner-authored common predicate AST exists",
        )
        for point in _POINT_IDS
    )
    unresolved.extend(
        (
            f"semantic.authored_hazards.predicate.{hazard}",
            "hazard ID/finding is declared but no exact owner predicate AST exists",
        )
        for hazard in _HAZARD_IDS
    )
    items.extend(
        ReadinessItem(fact, ReadinessStatus.UNRESOLVED, "UNRESOLVED", proof)
        for fact, proof in unresolved
    )
    return tuple(items)


_EXPECTED_REQUIRED_FACT_IDS = frozenset(_REQUIRED_CARD_V2_LEAF_FACTS).union(
    {
        f"semantic.decision_points.admissible_effect_classes.{point}"
        for point in _POINT_IDS
    },
    {f"semantic.common_predicate_ast.{point}" for point in _POINT_IDS},
    {f"semantic.authored_hazards.predicate.{hazard}" for hazard in _HAZARD_IDS},
)


def maintenance_card_v2_profile_declarations() -> MaintenanceCardIdentityProfile:
    """Return the detached V1-only declaration; readiness is fail-closed."""
    semantic = _semantic()
    inventory = _readiness()
    inventory_ids = frozenset(item.fact_id for item in inventory)
    unresolved = any(item.status is ReadinessStatus.UNRESOLVED for item in inventory)
    return MaintenanceCardIdentityProfile(
        profile_id="maintenance.card_identity_profile@2.0.0",
        coverage="V1_ONLY",
        projectable=(
            inventory_ids == _EXPECTED_REQUIRED_FACT_IDS
            and len(inventory) == len(_EXPECTED_REQUIRED_FACT_IDS)
            and not unresolved
        ),
        aliases=_aliases(),
        timestamp_policy=TimestampPolicy(
            runtime_form="YYYY-MM-DDTHH:MM:SSZ",
            card_form="YYYY-MM-DDTHH:MM:SS.000000Z",
            forward_rule="SAME_VALIDATED_UTC_INSTANT_APPEND_SIX_ZERO_FRACTION_DIGITS",
            inverse_rule="REMOVE_FRACTION_ONLY_WHEN_EXACTLY_000000",
            runtime_input_rule="NEVER_FEED_CARD_TEXT_TO_FROZEN_RUNTIME",
        ),
        role_policy=_roles(),
        field_shape_policy=FieldShapePolicy(
            integer_minimum=-(2**63),
            integer_maximum=2**63 - 1,
            integer_minimum_rule="PRESERVE_EXISTING_SEMANTIC_MINIMA_PER_FIELD",
            identifier_min_length=1,
            identifier_max_length=255,
            authored_text_min_length=1,
            authored_text_max_length=4096,
            digest_pattern="^[0-9a-f]{64}$",
            currency_pattern="^[A-Z]{3}$",
            closed_enum_rule="USE_CLOSED_ENUM_WHERE_RUNTIME_OWNER_HAS_CLOSED_VOCABULARY",
            nullable_string_paths=(
                "request_supplier_visit.optional.scope_digest",
                "send_message.optional.correlation_id",
            ),
            out_of_profile_behavior=(
                "REFUSE_IDENTITY_PROJECTION_PENDING_OWNER_VERSION_DECISION"
            ),
            runtime_version_behavior="MAINTENANCE_0.5.0_ACCEPTANCE_UNCHANGED",
        ),
        state_policy=StatePolicy(
            schema_id="maintenance.state.canonical@2.0.0",
            canonical_state_projection=tuple(
                f"/{name}"
                for name in sorted(
                    (
                        "phase",
                        "issue_id",
                        "issue_reporting_actor_id",
                        "classification",
                        "customer_resolution",
                        "current_cycle_id",
                        "cycles",
                        "quotes",
                        "approvals",
                        "exceptions",
                        "invoices",
                        "obligations",
                        "payment",
                        "assertions",
                        "authoritative_records",
                        "communications",
                        "ownership_transferred_to",
                        "transfer_notice_sent",
                        "provisional_close",
                        "reopen_count",
                        "terminal",
                        "replay_final",
                        "hidden",
                    )
                )
            ),
            replay_final_pointer="/replay_final",
            terminal_pointer="/terminal",
            additional_properties=False,
            schema_construction_contract=(
                "COMPLETE_CLOSED_MAINTENANCESTATE_CANONICAL_MATCHING_STATE_VALIDATOR; "
                "CLOSED_ENUMS_AND_CARD_AUTHORING_BOUNDS_APPLY_RECURSIVELY"
            ),
            hidden_fixture_input_kind="OVERLAY_SUBTREE",
            construction_rule="OVERLAY_FIXTURE_HIDDEN_SUBTREE_ON_CANONICAL_INITIAL_STATE",
        ),
        surface_policy=SurfacePolicy(
            algorithm_id="maintenance.surface_content.v1",
            hash="SHA-256",
            domain="operatebench.maintenance.surface_content.v1",
            canonical_object_fields=("domain", "fixture_id", "text"),
            accepted_source_grammar=(
                "fixture_id := exact non-empty built-in Unicode string without lone "
                "surrogates; text := exact non-empty built-in Unicode string without "
                "lone surrogates"
            ),
            target_representation_template=(
                '{"domain":"operatebench.maintenance.surface_content.v1",'
                '"fixture_id":<source fixture_id>,"text":<source text>}'
            ),
            framing_rule=(
                "SORT_KEYS; COMMA_COLON_SEPARATORS; ENSURE_ASCII_FALSE; ALLOW_NAN_FALSE; "
                "STRICT_UTF8"
            ),
            text_rule="EXACT_VALIDATED_LOADED_UNICODE_NO_NORMALIZATION",
            bytes_rule="CANONICAL_JSON_UTF8",
            binding_rule="COMPLETE_ONE_TO_ONE_REFERENCED_FIXTURE_BINDING",
            refusal_rules=(
                "MISSING_FIXTURE",
                "EXTRA_FIXTURE",
                "DUPLICATE_FIXTURE_ID",
                "UNBOUND_FIXTURE_ID",
                "INVALID_UNICODE",
            ),
        ),
        semantic_scenario=semantic,
        v1_variant=V1VariantPolicy(
            variant_label="maintenance.variant.approved_reopened",
            source_scenario_id="V1",
            card_scenario_id="maintenance.variant.approved_reopened.v1",
            expected_terminal="completed_successfully",
            human_checkpoint_budget=1,
            required_checkpoint_types=("repair_quote_approval",),
            dispatch_failures=(),
            expected_rejections=(("v1_e18", "after_replay_final"),),
            event_order_rule="SOURCE_AUTHORED_SEQUENCE_WITH_EXACTLY_ONE_TIMING_BRANCH",
            realization_invariants=(
                "event_type",
                "event_order",
                "authority",
                "policy",
                "obligation",
                "terminal",
                "rejection",
                "dispatch_behavior",
            ),
            actor_event_fixture_timing_source="spec.OperationSpec + ScenarioSpec[V1]",
            hidden_state_construction=(
                "OVERLAY_FIXTURE_HIDDEN_SUBTREE_ON_CANONICAL_INITIAL_STATE"
            ),
        ),
        deferred_scenarios=("V2_REJECTED_EXCEPTION", "V3_TIMEOUT_LATE_APPROVAL"),
        reconciliation=ReconciliationDescriptor(
            expected_event_types=17,
            expected_event_payload_fields=67,
            expected_actions=8,
            expected_retrieval_tools=8,
            expected_semantic_points=5,
            expected_exactly_one_clauses=8,
            expected_v1_events=18,
            existing_owner_modules=(
                "spec.py",
                "state.py",
                "retrieval.py",
                "core/read_contract.py",
                "semantic_arms_v1.py",
                "examples/operatebench/maintenance_v0_1.yaml",
            ),
            rule=(
                "RECONCILE_EXACT_SOURCE_OWNER_AT_PROJECTION; DO_NOT COPY OWNER TABLES "
                "OR ACCEPT COUNT_ONLY_MATCHES"
            ),
        ),
        readiness_inventory=inventory,
    )
