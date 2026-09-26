# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import pickle
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType

import pytest

from boundarybench.jsonsafe import canonical_json_bytes
from operatebench.domains.lettings.maintenance.retrieval import (
    MAINTENANCE_RETRIEVAL_TOOLS,
)
from operatebench.domains.lettings.maintenance.semantic_arms_v1 import COMMON_ESTIMAND
from operatebench.domains.lettings.maintenance.spec import (
    _EVENT_PAYLOAD_SCHEMAS,
    MAINTENANCE_EVENT_TYPES,
    action_schema_view,
    load_spec,
)
from operatebench.domains.lettings.maintenance.state import (
    CANONICAL_STATE_FIELDS,
    MaintenanceState,
)

MODULE = "operatebench.domains.lettings.maintenance.card_v2_declarations"
REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "examples/operatebench/maintenance_v0_1.yaml"
CARD_SCHEMAS = REPO / "src/operatebench/resources/cards"
GOLDEN = REPO / "tests/fixtures/maintenance-card-v2/profile-v2.json"
SEMANTIC_ORACLE = (
    REPO / "tests/fixtures/maintenance-card-v2/profile-v2-semantic-oracle.json"
)
SOURCE_ORACLE = (
    REPO / "tests/fixtures/maintenance-card-v2/source-owner-projection-v5.json"
)
GOLDEN_SHA256 = "94756713e2044c187f6ab8f47c02b17bc4038332cdb6f07699e9ee0eec479bcc"
SEMANTIC_ORACLE_SHA256 = (
    "6f205220245340e8203fad94612a4c0747ae207a3dd2260ad244e398d6fbd179"
)
SOURCE_ORACLE_SHA256 = "f6733266be3869c8b6685245e7231aca3eb4ce2b27d5cb83a9fbcaf65bbacbce"

EXPECTED_CAUSAL_NODES = (
    ("initial_report", "decision_point"),
    ("approval_granted", "decision_point"),
    ("work_verified", "decision_point"),
    ("invoice_validated", "decision_point"),
    ("warranty_reopened", "decision_point"),
    ("maintenance.hazard.premature_invoice", "authored_hazard"),
    ("maintenance.hazard.unverified_supplier_assertion", "authored_hazard"),
    ("maintenance.hazard.post_final_mutation", "authored_hazard"),
)
EXPECTED_CAUSAL_EDGES = (
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
EXPECTED_HAZARD_EDGE_RATIONALES = (
    (
        "maintenance.hazard.premature_invoice",
        "invoice_validated",
        "PRECEDES",
        "A premature invoice is the hazard immediately preceding the invoice validation "
        "boundary that must reject it without prior evidence.",
    ),
    (
        "maintenance.hazard.unverified_supplier_assertion",
        "work_verified",
        "PRECEDES",
        "An unverified supplier assertion is the hazard immediately preceding the work "
        "verification boundary that must not treat it as authority.",
    ),
    (
        "warranty_reopened",
        "maintenance.hazard.post_final_mutation",
        "PRECEDES",
        "The warranty-reopened finality point precedes the post-final mutation hazard "
        "whose attempted change must remain ineffective.",
    ),
)


def _assert_causal_graph_ready(semantic: dict[str, object]) -> None:
    """Independent mirror of frozen node-union, edge, DAG, and connection rules."""
    point_ids = semantic["point_ids"]
    hazard_ids = semantic["hazard_ids"]
    causal_nodes = semantic["causal_nodes"]
    edges = semantic["causal_edges"]
    assert isinstance(point_ids, list)
    assert isinstance(hazard_ids, list)
    assert isinstance(causal_nodes, list)
    assert isinstance(edges, list)
    node_ids = [node[0] for node in causal_nodes]
    assert len(node_ids) == len(set(node_ids))
    assert set(node_ids) == set(point_ids) | set(hazard_ids)
    graph = {node_id: [] for node_id in node_ids}
    indegree = dict.fromkeys(node_ids, 0)
    connected: set[str] = set()
    for cause, effect, _relation in edges:
        assert cause in graph and effect in graph
        graph[cause].append(effect)
        indegree[effect] += 1
        connected.update((cause, effect))
    ready = [node_id for node_id in node_ids if indegree[node_id] == 0]
    visited = 0
    while ready:
        node_id = ready.pop(0)
        visited += 1
        for effect in graph[node_id]:
            indegree[effect] -= 1
            if indegree[effect] == 0:
                ready.append(effect)
    assert visited == len(node_ids)
    assert connected == set(node_ids)


def _declarations():
    module = importlib.import_module(MODULE)
    return module, module.maintenance_card_v2_profile_declarations()


def _plain(value):
    if hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, frozenset)):
        return [_plain(item) for item in value]
    return value


def _source_owner_projection() -> dict[str, object]:
    """Derive owner truth without consulting declaration or either profile fixture."""
    spec = load_spec(FIXTURE)
    v1 = spec.scenario("V1")
    return {
        "fixture_identity": {
            "projection_id": "maintenance-source-owner-projection-v5",
            "schema_version": 1,
            "operation_id": spec.operation_id,
            "operation_type": spec.operation_type,
            "operation_version": spec.operation_version,
            "semantic_scenario_id": spec.semantic_scenario_id,
        },
        "actors": {
            actor_id: {"role": actor.role, "authority": sorted(actor.authority)}
            for actor_id, actor in sorted(spec.actors.items())
        },
        "event_contracts": {
            event_type: {
                "required_authority": MAINTENANCE_EVENT_TYPES[event_type],
                "required": dict(_EVENT_PAYLOAD_SCHEMAS[event_type][0]),
                "optional": dict(_EVENT_PAYLOAD_SCHEMAS[event_type][1]),
            }
            for event_type in sorted(MAINTENANCE_EVENT_TYPES)
        },
        "action_contracts": _plain(action_schema_view()),
        "retrieval_catalogue": {
            tool_id: {
                "source": tool.source,
                "authority": tool.authority,
                "schema_id": tool.schema_id,
                "arguments": dict(tool.arguments),
                "roots": list(tool.roots),
            }
            for tool_id, tool in sorted(MAINTENANCE_RETRIEVAL_TOOLS.items())
        },
        "message_fixtures": dict(sorted(spec.message_fixtures.items())),
        "v1": v1.semantic_payload(),
        "fixture_bindings": sorted(
            {
                str(value)
                for event in v1.events
                for key, value in event.payload.items()
                if key.endswith("fixture_id")
            }
        ),
    }


def _required_paths(schema: dict[str, object], prefix: str) -> set[str]:
    """Fresh recursive census of every required scalar/opaque leaf category."""
    found: set[str] = set()
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    assert isinstance(required, list) and isinstance(properties, dict)
    for name in required:
        assert isinstance(name, str)
        child = properties[name]
        assert isinstance(child, dict)
        if name == "content" and not prefix.endswith("events.trigger"):
            found.update(_required_paths(child, prefix))
            continue
        path = f"{prefix}.{name}"
        nested: set[str] = set()
        if child.get("type") == "object" and "required" in child:
            nested.update(_required_paths(child, path))
        items = child.get("items")
        if isinstance(items, dict) and "required" in items:
            nested.update(_required_paths(items, path))
        for branch_key in ("oneOf", "allOf"):
            branches = child.get(branch_key, [])
            if isinstance(branches, list):
                for branch in branches:
                    if isinstance(branch, dict) and "required" in branch:
                        nested.update(_required_paths(branch, path))
                    if isinstance(branch, dict):
                        for alternate in branch.get("oneOf", []):
                            if isinstance(alternate, dict) and "required" in alternate:
                                nested.update(_required_paths(alternate, path))
        if nested:
            found.update(nested)
            if "oneOf" in child:
                found.add(path)
        else:
            found.add(path)
    return found


def _card_required_leaf_ids() -> set[str]:
    result: set[str] = set()
    for prefix, name in (
        ("operation", "operation-card-v2.schema.json"),
        ("semantic", "semantic-scenario-card-v2.schema.json"),
        ("variant", "variant-card-v2.schema.json"),
    ):
        schema = json.loads((CARD_SCHEMAS / name).read_text(encoding="utf-8"))
        paths = _required_paths(schema, prefix)
        for path in paths:
            if path in {
                f"{prefix}.schema",
                f"{prefix}.schema_version",
                f"{prefix}.id",
                f"{prefix}.content_digest_sha256",
            }:
                _, field = path.split(".", 1)
                result.add(f"{prefix}.card.{field}")
            else:
                result.add(path)
    return result


def test_literal_golden_and_structurally_separate_semantic_oracle() -> None:
    _module, declaration = _declarations()
    golden_raw = GOLDEN.read_bytes()
    oracle_raw = SEMANTIC_ORACLE.read_bytes()
    assert hashlib.sha256(golden_raw).hexdigest() == GOLDEN_SHA256
    assert hashlib.sha256(oracle_raw).hexdigest() == SEMANTIC_ORACLE_SHA256
    golden = json.loads(golden_raw)
    oracle_document = json.loads(oracle_raw)
    assert "never derive this fixture in tests" in oracle_document["independence"]
    oracle = oracle_document["semantic_classes"]
    actual = declaration.as_plain()
    assert golden_raw == canonical_json_bytes(
        golden, "Maintenance Card v2 declaration golden"
    )
    assert set(actual) == set(oracle)
    for semantic_class in sorted(oracle):
        assert actual[semantic_class] == oracle[semantic_class]
    assert actual == golden


def test_semantic_oracle_defeats_coherent_golden_sha_repin() -> None:
    _module, declaration = _declarations()
    actual = declaration.as_plain()
    coherent_repin = copy.deepcopy(actual)
    coherent_repin["role_policy"]["shared_legacy_role"] = "merged_external_authority"
    coherently_changed_golden = copy.deepcopy(coherent_repin)
    coherently_changed_sha = hashlib.sha256(
        canonical_json_bytes(coherently_changed_golden, "coherent attacker repin")
    ).hexdigest()
    assert coherent_repin == coherently_changed_golden
    assert len(coherently_changed_sha) == 64
    independent = json.loads(SEMANTIC_ORACLE.read_bytes())["semantic_classes"]
    assert coherent_repin != independent
    assert "shared_legacy_role" not in actual["role_policy"]


def test_recursive_immutability_detachment_and_canonical_json() -> None:
    _module, first = _declarations()
    _module, second = _declarations()
    assert first == second and first is not second
    assert first.aliases is not second.aliases
    assert isinstance(first.role_policy.actor_role_overrides, MappingProxyType)
    with pytest.raises(FrozenInstanceError):
        first.coverage = "ALL"  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.role_policy.actor_role_overrides["intruder"] = "agent"  # type: ignore[index]
    assert copy.copy(first) is first
    with pytest.raises(TypeError):
        copy.deepcopy(first)
    with pytest.raises(TypeError):
        pickle.dumps(first)
    plain = first.as_plain()
    plain["coverage"] = "mutated"
    plain["role_policy"]["actor_role_overrides"]["maintenance_system"] = "mutated"
    assert second.coverage == "V1_ONLY"
    assert second.role_policy.actor_role_overrides["maintenance_system"] == (
        "maintenance_verification_system"
    )
    assert json.loads(json.dumps(second.as_plain())) == second.as_plain()


def test_declarative_representation_rules_have_independent_test_oracles() -> None:
    module, declaration = _declarations()
    assert not any(
        hasattr(module, name)
        for name in (
            "card_alias",
            "render_card_timestamp",
            "runtime_timestamp",
            "surface_content_digest",
            "_canonical_surface_bytes",
            "_valid_timestamp",
        )
    )
    assert {(item.scope, item.source, item.card) for item in declaration.aliases} == {
        (
            "issue_classification",
            "NON_EMERGENCY_RECURRING_LEAK",
            "non_emergency_recurring_leak",
        ),
        ("scenario_id", "V1", "maintenance.variant.approved_reopened.v1"),
        ("rejection_code", "AFTER_REPLAY_FINAL", "after_replay_final"),
    }
    timestamp = declaration.timestamp_policy
    assert timestamp.runtime_form == "YYYY-MM-DDTHH:MM:SSZ"
    assert timestamp.card_form == "YYYY-MM-DDTHH:MM:SS.000000Z"
    surface = declaration.surface_policy
    assert surface.algorithm_id == "maintenance.surface_content.v1"
    assert surface.hash == "SHA-256"
    assert surface.canonical_object_fields == ("domain", "fixture_id", "text")
    independent_bytes = canonical_json_bytes(
        {
            "domain": surface.domain,
            "fixture_id": "fixture_1",
            "text": "Exact validated Unicode: café",
        },
        "independent declaration test oracle",
    )
    assert len(hashlib.sha256(independent_bytes).hexdigest()) == 64
    assert "exact non-empty built-in Unicode string" in surface.accepted_source_grammar
    assert "STRICT_UTF8" in surface.framing_rule


def test_role_policy_owns_only_two_splits_and_reconciles_source_authority() -> None:
    _module, declaration = _declarations()
    spec = load_spec(FIXTURE)
    roles = declaration.role_policy
    assert dict(roles.actor_role_overrides) == {
        "maintenance_system": "maintenance_verification_system",
        "payment_system": "payment_authority_system",
    }
    assert roles.preservation_rule == "PRESERVE_SOURCE_ROLE_UNLESS_EXPLICIT_OVERRIDE"
    assert roles.authority_source == "RECONCILE_ACTUAL_OPERATION_SPEC_ACTOR_AUTHORITY"
    assert spec.actors["maintenance_system"].authority == frozenset(
        {"verify_attendance", "verify_work"}
    )
    assert spec.actors["payment_system"].authority == frozenset(
        {"validate_invoice", "confirm_settlement"}
    )
    assert spec.actors["maintenance_system"].authority.isdisjoint(
        spec.actors["payment_system"].authority
    )
    assert all(
        actor_id in roles.actor_role_overrides
        or spec.actors[actor_id].role
        == _source_owner_projection()["actors"][actor_id]["role"]
        for actor_id in spec.actors
    )


def test_semantic_and_variant_ownership_are_mutation_local() -> None:
    _module, declaration = _declarations()
    semantic = declaration.semantic_scenario
    variant = declaration.v1_variant
    assert not hasattr(semantic, "variant_label")
    assert variant.variant_label == "maintenance.variant.approved_reopened"
    assert semantic.common_estimand_source == "semantic_arms_v1.COMMON_ESTIMAND"
    assert COMMON_ESTIMAND == "fixed_five_point_ordered_boundary_resolution_status"
    assert semantic.node_ids == tuple(node_id for node_id, _ in EXPECTED_CAUSAL_NODES)
    assert semantic.causal_nodes == EXPECTED_CAUSAL_NODES
    assert semantic.causal_edges == EXPECTED_CAUSAL_EDGES
    assert semantic.causal_edge_rationales == EXPECTED_HAZARD_EDGE_RATIONALES
    assert {
        (cause, effect, relation)
        for cause, effect, relation, _rationale in semantic.causal_edge_rationales
    } == set(EXPECTED_CAUSAL_EDGES[-3:])
    changed_variant = copy.deepcopy(variant.__dict__)
    changed_variant["variant_label"] = "maintenance.variant.changed"
    assert changed_variant["variant_label"] != variant.variant_label
    assert semantic == declaration.semantic_scenario


def test_causal_nodes_are_exact_point_hazard_union_with_schema_valid_types() -> None:
    _module, declaration = _declarations()
    semantic = declaration.semantic_scenario
    node_ids = tuple(node_id for node_id, _semantic_type in semantic.causal_nodes)
    assert semantic.node_ids == node_ids
    assert set(node_ids) == set(semantic.point_ids) | set(semantic.hazard_ids)
    assert semantic.causal_nodes == EXPECTED_CAUSAL_NODES
    assert all(
        semantic_type in {"decision_point", "authored_hazard"}
        for _node_id, semantic_type in semantic.causal_nodes
    )
    assert not any(node_id.startswith("maintenance.node.") for node_id in node_ids)


def test_causal_graph_is_class_complete_for_frozen_structural_validator() -> None:
    _module, declaration = _declarations()
    semantic = declaration.as_plain()["semantic_scenario"]
    _assert_causal_graph_ready(semantic)

    mutants = []
    wrong_union = copy.deepcopy(semantic)
    wrong_union["causal_nodes"][0][0] = "maintenance.node.initial_report"
    mutants.append(wrong_union)
    unresolved_edge = copy.deepcopy(semantic)
    unresolved_edge["causal_edges"][0][0] = "absent"
    mutants.append(unresolved_edge)
    cyclic = copy.deepcopy(semantic)
    cyclic["causal_edges"].append(["warranty_reopened", "initial_report", "PRECEDES"])
    mutants.append(cyclic)
    disconnected = copy.deepcopy(semantic)
    disconnected["causal_edges"] = [
        edge
        for edge in disconnected["causal_edges"]
        if "maintenance.hazard.premature_invoice" not in edge[:2]
    ]
    mutants.append(disconnected)
    for mutant in mutants:
        with pytest.raises(AssertionError):
            _assert_causal_graph_ready(mutant)


@pytest.mark.parametrize(
    ("old_id", "new_id"),
    (
        ("invoice_validated", "invoice_accepted"),
        (
            "maintenance.hazard.premature_invoice",
            "maintenance.hazard.early_invoice",
        ),
    ),
)
def test_independent_causal_oracle_defeats_coherent_identity_repin(
    old_id: str, new_id: str
) -> None:
    _module, declaration = _declarations()
    actual = declaration.as_plain()

    def coherent_rename(value):
        if isinstance(value, dict):
            return {key: coherent_rename(item) for key, item in value.items()}
        if isinstance(value, list):
            return [coherent_rename(item) for item in value]
        if isinstance(value, str):
            return value.replace(old_id, new_id)
        return value

    coherent_production = coherent_rename(actual)
    coherent_golden = copy.deepcopy(coherent_production)
    coherent_sha = hashlib.sha256(
        canonical_json_bytes(coherent_golden, "coherent causal identity repin")
    ).hexdigest()
    assert coherent_production == coherent_golden and len(coherent_sha) == 64
    independent = json.loads(SEMANTIC_ORACLE.read_bytes())["semantic_classes"]
    assert coherent_production != independent
    assert coherent_production != actual


def test_required_card_v2_schema_census_exactly_matches_inventory_categories() -> None:
    _module, declaration = _declarations()
    inventory_ids = [item.fact_id for item in declaration.readiness_inventory]
    assert len(inventory_ids) == len(set(inventory_ids))
    census = _card_required_leaf_ids()
    assert census.issubset(set(inventory_ids))
    for required in (
        "semantic.common_estimand_id",
        "semantic.causal_graph.nodes.semantic_type",
        "semantic.common_outcome_obligations.terminal_required",
        "operation.transition_contract.obligation_types",
        "operation.transition_contract.common_predicate_inputs",
        "variant.variant_label",
    ):
        assert required in census
        assert required in inventory_ids
    by_id = {item.fact_id: item for item in declaration.readiness_inventory}
    assert by_id["semantic.common_estimand_id"].status.value == "EXISTING_OWNER"
    assert by_id["semantic.causal_graph.nodes.semantic_type"].status.value == "UNRESOLVED"
    assert by_id[
        "semantic.common_outcome_obligations.terminal_required"
    ].status.value == ("UNRESOLVED")
    assert by_id["operation.transition_contract.obligation_types"].status.value == (
        "UNRESOLVED"
    )
    assert by_id[
        "operation.transition_contract.common_predicate_inputs"
    ].status.value == ("UNRESOLVED")
    assert declaration.projectable is False
    assert any(
        item.status.value == "UNRESOLVED" for item in declaration.readiness_inventory
    )


def test_source_owner_projection_is_complete_exact_and_separately_pinned() -> None:
    # Earlier source-owner projections are historical evidence, not regeneration targets.
    for version, digest in (
        ("v1", "04976c1fbbe5fbd04a42ff497db240dbfd6b515fa32d660670e2d1d9b1cfe0be"),
        ("v2", "9379aa118f8c29f58587c977dce9311b5ca5264a266a092fdaa2fcb477d0241d"),
        ("v3", "1c6359a9ba22a3395f4981f05d811e1bacee5e96825fc58a0026037358ad07a3"),
        ("v4", "da486727678648f356387ad7f8003734b7da878588db8b5403916d4bf6104a15"),
    ):
        historical = SOURCE_ORACLE.with_name(f"source-owner-projection-{version}.json")
        assert hashlib.sha256(historical.read_bytes()).hexdigest() == digest
    raw = SOURCE_ORACLE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_ORACLE_SHA256
    expected = json.loads(raw)
    actual = _source_owner_projection()
    assert actual == expected
    assert len(actual["event_contracts"]) == 17
    assert len(actual["action_contracts"]) == 8
    assert len(actual["retrieval_catalogue"]) == 8
    assert len(actual["actors"]) == 8
    assert len(actual["message_fixtures"]) == 6
    assert len(actual["v1"]["events"]) == 18
    assert actual["v1"]["expected_event_rejections"] == [
        {
            "event_id": "v1_e18",
            "code": "AFTER_REPLAY_FINAL",
            "reason": (
                "A supplier assertion authored to arrive after the finalization timer. "
                "It must be recorded as evidence that a late message changed nothing."
            ),
        }
    ]


@pytest.mark.parametrize(
    "mutation",
    (
        "event_optional_field",
        "action_required_field",
        "retrieval_source",
        "message_text",
        "v1_payload",
        "v1_timing",
        "v1_rejection_reason",
    ),
)
def test_source_oracle_rejects_all_seven_reviewer_mutations(mutation: str) -> None:
    actual = _source_owner_projection()
    mutant = copy.deepcopy(actual)
    if mutation == "event_optional_field":
        optional = mutant["event_contracts"]["operator_exception_resolved"]["optional"]
        optional["exception_cycle_id"] = optional.pop("cycle_id")
    elif mutation == "action_required_field":
        required = mutant["action_contracts"]["request_supplier_visit"]["required"]
        required["visit_kind"] = required.pop("visit_type")
    elif mutation == "retrieval_source":
        mutant["retrieval_catalogue"]["list_quotes"]["source"] = "quoting_service_v2"
    elif mutation == "message_text":
        mutant["message_fixtures"]["msg_completion_notice"] += " Changed."
    elif mutation == "v1_payload":
        event = next(row for row in mutant["v1"]["events"] if row["event_id"] == "v1_e07")
        event["payload"]["assertion_id"] = "assertion_9"
    elif mutation == "v1_timing":
        event = next(row for row in mutant["v1"]["events"] if row["event_id"] == "v1_e07")
        event["delay_minutes"] = 1390
    else:
        mutant["v1"]["expected_event_rejections"][0]["reason"] = "Changed rationale."
    assert mutant != actual
    assert mutant != json.loads(SOURCE_ORACLE.read_bytes())


def test_state_contract_and_v2_v3_deferral_remain_exact() -> None:
    _module, declaration = _declarations()
    state = declaration.state_policy
    assert state.canonical_state_projection == tuple(
        f"/{name}" for name in sorted(CANONICAL_STATE_FIELDS)
    )
    spec = load_spec(FIXTURE)
    initial = MaintenanceState(hidden=spec.hidden_state_snapshot()).canonical()
    assert tuple(sorted(initial)) == tuple(sorted(CANONICAL_STATE_FIELDS))
    assert initial["hidden"] == spec.hidden_state_snapshot()
    assert declaration.deferred_scenarios == (
        "V2_REJECTED_EXCEPTION",
        "V3_TIMEOUT_LATE_APPROVAL",
    )
    assert not any(
        "V2" in item.fact_id or "V3" in item.fact_id
        for item in declaration.readiness_inventory
    )


def test_ast_boundary_has_only_data_construction_and_detachment_callables() -> None:
    path = REPO / "src/operatebench/domains/lettings/maintenance/card_v2_declarations.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    forbidden = (
        "operatebench.cards",
        "identity",
        "artifact",
        "runner",
        "evaluator",
        "replay",
        "provider",
        "runtime",
    )
    assert not any(any(token in imported for token in forbidden) for imported in imports)
    module_functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert module_functions == {
        "_plain",
        "_aliases",
        "_roles",
        "_semantic",
        "_readiness",
        "maintenance_card_v2_profile_declarations",
    }
    hostile_names = (
        "project",
        "adapter",
        "digest",
        "canonical_bytes",
        "alias",
        "timestamp",
        "render",
        "inverse",
        "transform",
    )
    assert not any(
        token in function.lower()
        for function in module_functions
        for token in hostile_names
        if function not in {"_aliases"}
    )
    assert "__all__" not in {
        node.targets[0].id
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }


def test_loaded_leaf_globals_and_annotations_hold_no_forbidden_api() -> None:
    module, declaration = _declarations()
    forbidden = ("operatebench.cards", "identity", "evaluator", "provider", "runner")
    for value in module.__dict__.values():
        value_module = getattr(value, "__module__", "")
        if isinstance(value_module, str):
            assert not any(token in value_module for token in forbidden)
    for value in declaration.__dataclass_fields__.values():
        annotation = str(value.type)
        assert not any(token in annotation for token in forbidden)
    operatebench = importlib.import_module("operatebench")
    assert not hasattr(operatebench, "maintenance_card_v2_profile_declarations")
