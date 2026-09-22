# SPDX-License-Identifier: Apache-2.0
"""Executable, non-self-referential B3.3 identity-adapter contract checks.

Every expectation below is authored in this file. Nothing is read out of the
registry schema, the census schema, or either resource and then compared back to
itself: the complete pointer surface, the owner-to-source/consumer policy, the
authoritative source leaves, the dependency bindings, the nullability map and the
rejection vocabulary are literals here, and the frozen B1 field census and B2
component registry supply the independent bindings.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs" / "schemas"
CARDS = ROOT / "src" / "operatebench" / "resources" / "cards"
PACKAGED = ROOT / "src" / "operatebench" / "resources" / "identity"
REGISTRY_SCHEMA = "runtime-identity-adapter-registry-v1.schema.json"
REGISTRY = "runtime-identity-adapter-registry-v1.json"
CENSUS_SCHEMA = "runtime-identity-adapter-census-v1.schema.json"
CENSUS = "runtime-identity-adapter-census-v1.json"
B1_CENSUS = "identity-field-census-v1.registry.json"
B2_REGISTRY = "identity-component-registry-v1.json"
B3_SCHEMAS = (
    "build-provenance-descriptor-v1.schema.json",
    "clean-tracked-source-tree-v1.schema.json",
    "decision-tape-v1.schema.json",
    "episode-outcome-v2.schema.json",
    "executed-runtime-evidence-v1.schema.json",
    "execution-record-v1.schema.json",
    "provider-execution-binding-v1.schema.json",
    REGISTRY_SCHEMA,
    CENSUS_SCHEMA,
)

# The eight runtime-bundle owners, in the order B2 froze them.
EXPECTED_OWNERS = (
    "operation_core_content",
    "semantic_scenario_content",
    "variant_content",
    "runtime_contract_content",
    "scaffold_content",
    "arm_protocol_content",
    "provider_run_plan_content",
    "build_provenance_content",
)

# Owner -> (census source, census consumer). Authored independently of both
# runtime adapter resources.
OWNER_POLICY = {
    "operation_core_content": (
        "b1_operation_card_projector",
        "runtime_manifest_binding",
    ),
    "semantic_scenario_content": (
        "b1_semantic_scenario_card_projector",
        "runtime_manifest_binding",
    ),
    "variant_content": ("b1_variant_card_projector", "runtime_manifest_binding"),
    "runtime_contract_content": ("live_runtime_declarations", "runtime_replay"),
    "scaffold_content": ("selected_agent_authority", "runtime_replay"),
    "arm_protocol_content": ("compiled_arm_and_lane_registry", "runtime_replay"),
    "provider_run_plan_content": ("execution_plan_and_ledger_audit", "runtime_replay"),
    "build_provenance_content": ("build_provenance_descriptor", "runtime_replay"),
}

# Owner -> the B1 card schema it projects, or None where no B1 card exists.
OWNER_CARD_SCHEMA = {
    "operation_core_content": (
        "src/operatebench/resources/cards/operation-card-v1.schema.json"
    ),
    "semantic_scenario_content": (
        "src/operatebench/resources/cards/semantic-scenario-card-v1.schema.json"
    ),
    "variant_content": "src/operatebench/resources/cards/variant-card-v1.schema.json",
    "runtime_contract_content": None,
    "scaffold_content": None,
    "arm_protocol_content": None,
    "provider_run_plan_content": None,
    "build_provenance_content": None,
}

# The three card roots the frozen B1 census owns, and their owners.
B1_CARD_ROOTS = {
    "operation_card": "operation_core_content",
    "semantic_scenario_card": "semantic_scenario_content",
    "variant_card": "variant_content",
}

EXPECTED_DEPENDENCY_ORDER = (
    "contract_amendment",
    "b1_projectors_and_build_descriptor",
    "private_runtime_manifest_binding",
    "canonical_immutable_constituents_and_evidence_digest",
    "can_runtime_replay",
    "evaluator_compatibility",
    "capability_consume",
)

EXPECTED_STAGES = (
    "contract",
    "b1_projectors",
    "build_descriptor",
    "runtime_manifest_binding",
    "canonical_evidence",
    "can_runtime_replay",
    "evaluator_compatibility",
    "capability_consume",
)

RESOURCE_LEAVES = frozenset({"uv.lock", "pyproject.toml"})


EXPECTED_OWNER_PATHS = {
    "operation_core_content": (
        "/operation_card/schema",
        "/operation_card/schema_version",
        "/operation_card/id",
        "/operation_card/content_digest_sha256",
        "/operation_card/content/operation_type",
        "/operation_card/content/operation_version",
        "/operation_card/content/state_contract",
        "/operation_card/content/authority_contract",
        "/operation_card/content/policy_contract",
        "/operation_card/content/event_contracts",
        "/operation_card/content/action_contracts",
        "/operation_card/content/retrieval_contract",
        "/operation_card/content/transition_contract",
    ),
    "semantic_scenario_content": (
        "/semantic_scenario_card/schema",
        "/semantic_scenario_card/schema_version",
        "/semantic_scenario_card/id",
        "/semantic_scenario_card/content_digest_sha256",
        "/semantic_scenario_card/content/operation_core",
        "/semantic_scenario_card/content/construct_label",
        "/semantic_scenario_card/content/common_estimand_id",
        "/semantic_scenario_card/content/decision_points",
        "/semantic_scenario_card/content/causal_graph",
        "/semantic_scenario_card/content/authored_hazards",
        "/semantic_scenario_card/content/common_outcome_obligations",
        "/semantic_scenario_card/content/allowed_variant_dimensions",
    ),
    "variant_content": (
        "/variant_card/schema",
        "/variant_card/schema_version",
        "/variant_card/id",
        "/variant_card/content_digest_sha256",
        "/variant_card/content/semantic_scenario",
        "/variant_card/content/variant_label",
        "/variant_card/content/starts_at",
        "/variant_card/content/expected_terminal",
        "/variant_card/content/human_checkpoint_budget",
        "/variant_card/content/required_checkpoint_types",
        "/variant_card/content/dispatch_failures",
        "/variant_card/content/expected_event_rejections",
        "/variant_card/content/actors",
        "/variant_card/content/hidden_initial_values",
        "/variant_card/content/message_fixtures",
        "/variant_card/content/events",
    ),
    "runtime_contract_content": (
        "/runtime_contract/engine_version",
        "/runtime_contract/artifact_version",
        "/runtime_contract/model_protocol_version",
        "/runtime_contract/model_visible_fields",
        "/runtime_contract/outcome_tool_schema",
        "/runtime_contract/wake_event_types",
        "/runtime_contract/engine_class",
        "/runtime_contract/domain_class",
    ),
    "scaffold_content": (
        "/scaffold/agent_id",
        "/scaffold/agent_kind",
        "/scaffold/registry_factory",
        "/scaffold/supported_scenarios",
        "/scaffold/prompt_envelope_keys",
        "/scaffold/instructions_contract",
        "/scaffold/tool_names",
        "/scaffold/request_mapping",
    ),
    "arm_protocol_content": (
        "/arm_protocol/treatment_label",
        "/arm_protocol/allowed_claim",
        "/arm_protocol/semantic_scenario_id",
        "/arm_protocol/semantic_scenario_digest_sha256",
        "/arm_protocol/common_expected_vector",
        "/arm_protocol/denominator",
        "/arm_protocol/point_ids",
        "/arm_protocol/subject_continuity",
        "/arm_protocol/harness_metadata",
    ),
    "provider_run_plan_content": (
        "/provider_run_plan/execution_mode",
        "/provider_run_plan/provider",
        "/provider_run_plan/api",
        "/provider_run_plan/model",
        "/provider_run_plan/model_profile",
        "/provider_run_plan/settings",
        "/provider_run_plan/controls",
        "/provider_run_plan/pricing_policy",
        "/provider_run_plan/max_output_tokens",
        "/provider_run_plan/retry_policy",
        "/provider_run_plan/ledger_contract_version",
    ),
    "build_provenance_content": (
        "/build_provenance/schema",
        "/build_provenance/schema_version",
        "/build_provenance/source_commit",
        "/build_provenance/source_tree_algorithm",
        "/build_provenance/source_tree_digest_sha256",
        "/build_provenance/tree_clean",
        "/build_provenance/uv_lock_sha256",
        "/build_provenance/build_mode",
        "/build_provenance/distribution_artifacts",
        "/build_provenance/toolchain",
    ),
}

EXPECTED_SOURCE_LEAVES = {
    "operation_core_content": (
        "examples/operatebench/maintenance_v0_1.yaml",
        "operatebench.domains.lettings.maintenance.retrieval.maintenance_retrieval_catalogue",
        "operatebench.domains.lettings.maintenance.spec.OperationSpec.action_schema_view",
        "operatebench.domains.lettings.maintenance.spec.OperationSpec.identity_payload",
        "operatebench.domains.lettings.maintenance.spec.OperationSpec.verify_identity",
        "operatebench.domains.lettings.maintenance.spec.load_spec",
        "operatebench.domains.lettings.maintenance.spec.maintenance_action_evidence_contract",
    ),
    "semantic_scenario_content": (
        "operatebench.domains.lettings.maintenance.semantic_arms_v1.SemanticScenario.semantic_payload",
        "operatebench.domains.lettings.maintenance.semantic_arms_v1.SemanticScenario.verify_identity",
        "operatebench.domains.lettings.maintenance.semantic_arms_v1.maintenance_v1_semantic_scenario",
    ),
    "variant_content": (
        "operatebench.domains.lettings.maintenance.spec.OperationSpec.actor_snapshot",
        "operatebench.domains.lettings.maintenance.spec.OperationSpec.hidden_state_snapshot",
        "operatebench.domains.lettings.maintenance.spec.OperationSpec.message_fixture_snapshot",
        "operatebench.domains.lettings.maintenance.spec.OperationSpec.scenario",
        "operatebench.domains.lettings.maintenance.spec.ScenarioSpec.semantic_payload",
    ),
    "runtime_contract_content": (
        "operatebench.agents.model.MODEL_PROTOCOL_VERSION",
        "operatebench.agents.model.tool_schema",
        "operatebench.artifact.ARTIFACT_VERSION",
        "operatebench.core.engine.Engine",
        "operatebench.core.protocol.MODEL_VISIBLE_FIELDS",
        "operatebench.core.protocol.model_projection",
        "operatebench.domains.lettings.maintenance.operation.MaintenanceOperation",
        "operatebench.domains.lettings.maintenance.spec.MAINTENANCE_WAKE_EVENT_TYPES",
        "operatebench.version.OPERATEBENCH_VERSION",
    ),
    "scaffold_content": (
        "operatebench.agents.lifecycle_contract.INSTRUCTIONS_CONTRACT",
        "operatebench.agents.lifecycle_contract.check_model_request",
        "operatebench.agents.lifecycle_contract.outcome_tools",
        "operatebench.agents.model.ModelAgent.build_request",
        "operatebench.agents.openai_responses.LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION",
        "operatebench.domains.lettings.maintenance.agents.AGENTS",
        "operatebench.domains.lettings.maintenance.agents.AgentEntry",
        "operatebench.domains.lettings.maintenance.agents.agent_entry",
        "operatebench.domains.lettings.maintenance.agents.build_agent",
    ),
    "arm_protocol_content": (
        "operatebench.domains.lettings.maintenance.semantic_arms_v1.CompiledArm",
        "operatebench.domains.lettings.maintenance.semantic_arms_v1.compile_maintenance_v1",
    ),
    "provider_run_plan_content": (
        "operatebench.agents.evidence.provider_identity_for",
        "operatebench.agents.openai_responses.LIFECYCLE_RETRY_POLICY",
        "operatebench.agents.openai_responses.lifecycle_openai_settings",
        "operatebench.agents.transport.in_process_execution",
        "operatebench.execution_bundle.audit_execution_bundle_files",
        "operatebench.execution_ledger.EXECUTION_LEDGER_VERSION",
        "operatebench.execution_ledger.LedgerControls",
        "operatebench.execution_ledger.PRICING_RATE_FIELDS",
        "operatebench.providers.openai_responses.MAX_OUTPUT_TOKENS",
        "operatebench.providers.openai_responses.request_profile_for",
    ),
    "build_provenance_content": (
        "docs/schemas/build-provenance-descriptor-v1.schema.json",
        "docs/schemas/clean-tracked-source-tree-v1.golden.json",
        "uv.lock",
    ),
}

EXPECTED_DEPENDENCY_BINDINGS = {
    "operation_core_content": (),
    "semantic_scenario_content": (
        ("operation_core_content", "/operation_card/id"),
        ("operation_core_content", "/operation_card/content_digest_sha256"),
    ),
    "variant_content": (
        ("semantic_scenario_content", "/semantic_scenario_card/id"),
        ("semantic_scenario_content", "/semantic_scenario_card/content_digest_sha256"),
    ),
    "runtime_contract_content": (
        ("operation_core_content", "/operation_card/content/action_contracts"),
        ("operation_core_content", "/operation_card/content/retrieval_contract"),
    ),
    "scaffold_content": (
        ("runtime_contract_content", "/runtime_contract/model_visible_fields"),
        ("runtime_contract_content", "/runtime_contract/outcome_tool_schema"),
        ("runtime_contract_content", "/runtime_contract/model_protocol_version"),
    ),
    "arm_protocol_content": (
        ("runtime_contract_content", "/runtime_contract/engine_class"),
        ("semantic_scenario_content", "/semantic_scenario_card/content_digest_sha256"),
        ("variant_content", "/variant_card/id"),
    ),
    "provider_run_plan_content": (
        ("runtime_contract_content", "/runtime_contract/model_protocol_version"),
        ("scaffold_content", "/scaffold/request_mapping"),
        ("build_provenance_content", "/build_provenance/source_tree_digest_sha256"),
    ),
    "build_provenance_content": (),
}

EXPECTED_REJECTION_CODES = (
    "adapter_source_unavailable",
    "artifact_digest_mismatch",
    "arm_assignment_missing",
    "arm_assignment_ambiguous",
    "b1_projector_unavailable",
    "build_descriptor_unavailable",
    "build_provenance_mismatch",
    "card_projection_invalid",
    "census_mismatch",
    "dirty_source_tree",
    "freeze_manifest_digest_mismatch",
    "historical_build_descriptor_unavailable",
    "ledger_required",
    "ledger_audit_failed",
    "manifest_input_not_authority",
    "provider_plan_mismatch",
    "runtime_contract_mismatch",
    "scaffold_mismatch",
    "selected_run_mismatch",
    "source_identity_mismatch",
    "unsupported_agent",
    "unsupported_build_mode",
    "unsupported_lane",
    "unsupported_model_profile",
)

EXPLICIT_NULL_ALLOWED = frozenset(
    {
        "/build_provenance/distribution_artifacts",
        "/provider_run_plan/api",
        "/provider_run_plan/controls",
        "/provider_run_plan/ledger_contract_version",
        "/provider_run_plan/max_output_tokens",
        "/provider_run_plan/model",
        "/provider_run_plan/model_profile",
        "/provider_run_plan/pricing_policy",
        "/provider_run_plan/provider",
        "/provider_run_plan/retry_policy",
        "/provider_run_plan/settings",
    }
)


ALL_EXPECTED_PATHS = tuple(
    path for owner in EXPECTED_OWNERS for path in EXPECTED_OWNER_PATHS[owner]
)
ALL_EXPECTED_LEAVES = tuple(
    leaf for owner in EXPECTED_OWNERS for leaf in EXPECTED_SOURCE_LEAVES[owner]
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _leaf_source_path(leaf: str) -> Path:
    if "/" in leaf or leaf in RESOURCE_LEAVES:
        return ROOT / leaf
    parts = leaf.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        candidate = ROOT / "src" / Path(*parts[:cut]).with_suffix(".py")
        if candidate.is_file():
            return candidate
    raise AssertionError(f"no module resolves the source leaf {leaf}")


def _leaf_symbols(leaf: str) -> list[str]:
    if "/" in leaf or leaf in RESOURCE_LEAVES:
        return []
    parts = leaf.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        if (ROOT / "src" / Path(*parts[:cut]).with_suffix(".py")).is_file():
            return parts[cut:]
    raise AssertionError(f"no module resolves the source leaf {leaf}")


def _expected_source_paths(owner: str) -> list[str]:
    return sorted(
        {
            _leaf_source_path(leaf).relative_to(ROOT).as_posix()
            for leaf in EXPECTED_SOURCE_LEAVES[owner]
        }
    )


def _b2_requirements() -> dict[str, list[str]]:
    b2 = _load(PACKAGED / B2_REGISTRY)
    return {item["name"]: item["requires"] for item in b2["components"]}


def _b1_owner_of(pointer: str) -> set[str]:
    entries = _load(DOCS / B1_CENSUS)["entries"]
    return {
        entry["owner"]
        for entry in entries
        if entry["field_path_pattern"] == pointer
        or entry["field_path_pattern"].startswith(pointer + "/")
    }


def _assert_pointer_surface(
    registry: dict[str, Any],
    census: dict[str, Any],
    registry_schema: dict[str, Any],
    census_schema: dict[str, Any],
) -> None:
    """The whole pointer oracle, so an invented pointer can be shown to fail."""
    assert [owner["owner"] for owner in registry["owners"]] == list(EXPECTED_OWNERS)
    for owner in registry["owners"]:
        expected = list(EXPECTED_OWNER_PATHS[owner["owner"]])
        assert owner["projection_pointers"] == expected, owner["owner"]
    assert registry_schema["$defs"]["projectionPointer"]["enum"] == list(
        ALL_EXPECTED_PATHS
    )
    assert [entry["field_path"] for entry in census["entries"]] == list(
        ALL_EXPECTED_PATHS
    )
    entries = census_schema["properties"]["entries"]
    assert entries["items"]["properties"]["field_path"]["enum"] == list(
        ALL_EXPECTED_PATHS
    )
    assert entries["minItems"] == entries["maxItems"] == len(ALL_EXPECTED_PATHS)


def test_adapter_resources_meta_validate_and_packaged_copies_match() -> None:
    for name in (REGISTRY_SCHEMA, CENSUS_SCHEMA):
        Draft202012Validator.check_schema(_load(DOCS / name))
    Draft202012Validator(_load(DOCS / REGISTRY_SCHEMA)).validate(_load(DOCS / REGISTRY))
    Draft202012Validator(_load(DOCS / CENSUS_SCHEMA)).validate(_load(DOCS / CENSUS))
    for name in (REGISTRY_SCHEMA, REGISTRY, CENSUS_SCHEMA, CENSUS):
        assert (DOCS / name).read_bytes() == (PACKAGED / name).read_bytes()


@pytest.mark.parametrize(
    ("schema_name", "property_name", "expected"),
    (
        (REGISTRY_SCHEMA, "dependency_order", EXPECTED_DEPENDENCY_ORDER),
        (CENSUS_SCHEMA, "owners", EXPECTED_OWNERS),
    ),
)
def test_adapter_fixed_vocabularies_require_exact_complete_arrays(
    schema_name: str, property_name: str, expected: tuple[str, ...]
) -> None:
    validator = Draft202012Validator(
        _load(DOCS / schema_name)["properties"][property_name]
    )
    validator.validate(list(expected))
    for prefix_length in range(len(expected)):
        with pytest.raises(ValidationError):
            validator.validate(list(expected[:prefix_length]))
    with pytest.raises(ValidationError):
        validator.validate([*expected, "unexpected"])
    with pytest.raises(ValidationError):
        validator.validate([expected[1], expected[0], *expected[2:]])
    with pytest.raises(ValidationError):
        validator.validate(["unexpected", *expected[1:]])
    for non_array in (True, "unexpected", {}, None):
        with pytest.raises(ValidationError):
            validator.validate(non_array)


def _prefix_item_nodes(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "prefixItems" in value:
            found.append(value)
        for item in value.values():
            found.extend(_prefix_item_nodes(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_prefix_item_nodes(item))
    return found


def test_every_b3_schema_prefix_tuple_has_exact_array_cardinality() -> None:
    seen = 0
    for schema_name in B3_SCHEMAS:
        for node in _prefix_item_nodes(_load(DOCS / schema_name)):
            seen += 1
            length = len(node["prefixItems"])
            assert node.get("type") == "array", schema_name
            assert node.get("minItems") == length, schema_name
            assert node.get("maxItems") == length, schema_name
    assert seen == 5


def _patterns(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "pattern" and isinstance(item, str):
                found.append(item)
            found.extend(_patterns(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_patterns(item))
    return found


def test_adapter_schema_patterns_use_absolute_endings() -> None:
    seen = 0
    for name in (REGISTRY_SCHEMA, CENSUS_SCHEMA):
        for pattern in _patterns(_load(DOCS / name)):
            seen += 1
            assert pattern.startswith("^"), (name, pattern)
            assert not pattern.endswith("$"), (name, pattern)
            assert pattern.endswith(r"(?![\s\S])"), (name, pattern)
    assert seen >= 6


def test_all_eight_runtime_bundle_owners_derive_from_b2_registry() -> None:
    b2 = _load(PACKAGED / B2_REGISTRY)
    runtime_members = next(
        item["members"] for item in b2["composites"] if item["name"] == "runtime_bundle"
    )
    assert runtime_members == list(EXPECTED_OWNERS)
    assert [owner["owner"] for owner in _load(DOCS / REGISTRY)["owners"]] == list(
        EXPECTED_OWNERS
    )
    assert _load(DOCS / CENSUS)["owners"] == list(EXPECTED_OWNERS)


def test_pointer_surface_equals_the_literal_expectation_in_this_file() -> None:
    _assert_pointer_surface(
        _load(DOCS / REGISTRY),
        _load(DOCS / CENSUS),
        _load(DOCS / REGISTRY_SCHEMA),
        _load(DOCS / CENSUS_SCHEMA),
    )
    assert len(ALL_EXPECTED_PATHS) == len(set(ALL_EXPECTED_PATHS)) == 87


def test_a_coherently_invented_pointer_is_refused_by_the_literal_oracle() -> None:
    """The reviewer's escape: rename one pointer everywhere and stay self-consistent."""
    victim = "/build_provenance/toolchain"
    survivor = "/build_provenance/unowned_survivor"

    def rename(value: Any) -> Any:
        if isinstance(value, str):
            return survivor if value == victim else value
        if isinstance(value, list):
            return [rename(item) for item in value]
        if isinstance(value, dict):
            return {key: rename(item) for key, item in value.items()}
        return value

    registry = rename(_load(DOCS / REGISTRY))
    census = rename(_load(DOCS / CENSUS))
    registry_schema = rename(_load(DOCS / REGISTRY_SCHEMA))
    census_schema = rename(_load(DOCS / CENSUS_SCHEMA))
    # The invention is internally coherent: it still validates against its own schemas.
    Draft202012Validator(registry_schema).validate(registry)
    Draft202012Validator(census_schema).validate(census)
    with pytest.raises(AssertionError):
        _assert_pointer_surface(registry, census, registry_schema, census_schema)


def test_census_rows_match_the_literal_owner_source_consumer_and_nullability_policy() -> (
    None
):
    census = _load(DOCS / CENSUS)
    owner_of = {
        path: owner for owner in EXPECTED_OWNERS for path in EXPECTED_OWNER_PATHS[owner]
    }
    for entry in census["entries"]:
        path = entry["field_path"]
        owner = owner_of[path]
        source, consumer = OWNER_POLICY[owner]
        assert entry["owner"] == owner, path
        assert entry["source"] == source, path
        assert entry["consumer"] == consumer, path
        assert entry["source_paths"] == _expected_source_paths(owner), path
        assert entry["cardinality"] == "exact", path
        expected_nullability = (
            "explicit_null_allowed" if path in EXPLICIT_NULL_ALLOWED else "never"
        )
        assert entry["nullability"] == expected_nullability, path
    assert {
        entry["field_path"]
        for entry in census["entries"]
        if entry["nullability"] == "explicit_null_allowed"
    } == set(EXPLICIT_NULL_ALLOWED)


def test_card_pointers_resolve_against_the_frozen_b1_field_census() -> None:
    registry = _load(DOCS / REGISTRY)
    b1_roots = {
        entry["field_path_pattern"].split("/")[1]
        for entry in _load(DOCS / B1_CENSUS)["entries"]
    }
    assert set(B1_CARD_ROOTS) <= b1_roots
    checked = 0
    for owner in registry["owners"]:
        name = owner["owner"]
        for pointer in owner["projection_pointers"]:
            root = pointer.split("/")[1]
            if root not in B1_CARD_ROOTS:
                assert root not in b1_roots, pointer
                continue
            assert B1_CARD_ROOTS[root] == name, pointer
            owners = _b1_owner_of(pointer)
            assert owners == {name}, (pointer, owners)
            checked += 1
    assert checked == 13 + 12 + 16


def test_card_schema_pointers_name_the_shipped_b1_card_schemas() -> None:
    registry = _load(DOCS / REGISTRY)
    for owner in registry["owners"]:
        expected = OWNER_CARD_SCHEMA[owner["owner"]]
        assert owner["card_schema"] == expected, owner["owner"]
        if expected is None:
            continue
        packaged = ROOT / expected
        assert packaged.is_file()
        assert packaged.read_bytes() == (DOCS / packaged.name).read_bytes()


def test_authoritative_source_leaves_are_globally_unique_and_resolve() -> None:
    registry = _load(DOCS / REGISTRY)
    for owner in registry["owners"]:
        assert owner["authoritative_source_leaves"] == list(
            EXPECTED_SOURCE_LEAVES[owner["owner"]]
        ), owner["owner"]
    assert len(ALL_EXPECTED_LEAVES) == len(set(ALL_EXPECTED_LEAVES))
    for leaf in ALL_EXPECTED_LEAVES:
        path = _leaf_source_path(leaf)
        assert path.is_file(), leaf
        symbols = _leaf_symbols(leaf)
        if not symbols:
            continue
        text = path.read_text(encoding="utf-8")
        for symbol in symbols:
            pattern = (
                rf"^\s*(?:def |async def |class )?{re.escape(symbol)}\b\s*(?:\(|[:=])"
            )
            assert re.search(pattern, text, re.MULTILINE), (leaf, symbol)


def test_owned_content_is_isolated_and_no_fact_is_owned_twice() -> None:
    registry = _load(DOCS / REGISTRY)
    roots: dict[str, str] = {}
    for owner in registry["owners"]:
        owner_roots = {pointer.split("/")[1] for pointer in owner["projection_pointers"]}
        assert len(owner_roots) == 1, owner["owner"]
        root = owner_roots.pop()
        assert root not in roots, root
        roots[root] = owner["owner"]
    assert len(roots) == len(EXPECTED_OWNERS)
    # The five facts the B2 DAG already relates must appear as owned content once.
    once = {
        "model_visible_fields": "runtime_contract_content",
        "outcome_tool_schema": "runtime_contract_content",
        "model_protocol_version": "runtime_contract_content",
        "engine_class": "runtime_contract_content",
        "request_mapping": "scaffold_content",
        "action_contracts": "operation_core_content",
        "retrieval_contract": "operation_core_content",
    }
    index: dict[str, list[str]] = {}
    for owner in registry["owners"]:
        for pointer in owner["projection_pointers"]:
            index.setdefault(pointer.rsplit("/", 1)[1], []).append(owner["owner"])
    for leaf, expected_owner in once.items():
        assert index.get(leaf) == [expected_owner], (leaf, index.get(leaf))
    shared = {
        leaf
        for leaf, owners in index.items()
        if len(owners) > 1
        and leaf not in {"schema", "schema_version", "id", "content_digest_sha256"}
    }
    assert shared == set()


def test_dependency_bindings_reproduce_the_frozen_b2_requirement_edges() -> None:
    registry = _load(DOCS / REGISTRY)
    requirements = _b2_requirements()
    owned = {
        owner["owner"]: set(owner["projection_pointers"]) for owner in registry["owners"]
    }
    for owner in registry["owners"]:
        name = owner["owner"]
        actual = [
            (binding["requires_owner"], binding["binds_pointer"])
            for binding in owner["dependency_bindings"]
        ]
        assert actual == list(EXPECTED_DEPENDENCY_BINDINGS[name]), name
        for requires_owner, pointer in actual:
            assert requires_owner != name
            assert pointer in owned[requires_owner], (name, pointer)
            assert pointer not in owned[name], (name, pointer)
            assert requires_owner in requirements[name], (name, requires_owner)
        expected_partners = [
            required for required in requirements[name] if required in EXPECTED_OWNERS
        ]
        assert sorted({partner for partner, _ in actual}) == sorted(expected_partners), (
            name
        )
        assert all(binding["binding"].strip() for binding in owner["dependency_bindings"])


def test_only_two_exact_b31_artifacts_are_assigned_full_lifecycle() -> None:
    registry = _load(DOCS / REGISTRY)
    lanes = registry["lane_registry"]
    expected = {
        "45f2f3bec377230a0fc464c5ea1b64798a053beeb588c3ec4bdb477eaf7cfab5",
        "115bd5c058f4b4057cf7b4cd18d7cf0b367c0234b4367fbac92396aa26a24b3f",
    }
    assert {lane["artifact_sha256"] for lane in lanes} == expected
    assert {lane["arm_assignment"] for lane in lanes} == {"full_lifecycle"}
    assert all(
        lane["assignment_basis"] == "exact_internal_lane_registry" for lane in lanes
    )
    assert registry["arm_default"] is None
    assert registry["outcome_inference_authorized"] is False


def test_fake_agent_authorization_is_jointly_pinned_and_not_supported_yet() -> None:
    registry = _load(DOCS / REGISTRY)
    fake = next(
        lane for lane in registry["lane_registry"] if lane["agent_id"] is not None
    )
    assert fake["agent_id"] == "openai-gpt-5-6-luna"
    assert fake["artifact_sha256"] == (
        "115bd5c058f4b4057cf7b4cd18d7cf0b367c0234b4367fbac92396aa26a24b3f"
    )
    assert fake["ledger_file_sha256"] == (
        "29fc1c33d601529151b9340bc253992c3b55c28ba4c55849f070df13ead7afdb"
    )
    assert fake["freeze_manifest_sha256"] == (
        "6ec8409ca55dc4fbb0f5d9078f25004429edd6525209d1e8080b163b36a2b8c9"
    )
    assert fake["support_status"] == "blocked_historical_build_descriptor"
    assert registry["arbitrary_injected_agents_supported"] is False


def test_support_matrix_is_honest_and_manifest_is_output_only() -> None:
    registry = _load(DOCS / REGISTRY)
    assert registry["manifest_authority"] == "adapter_output_only"
    assert registry["initial_implementation_support"] == (
        "deterministic_reference_fresh_after_projectors_and_build_descriptor"
    )
    assert registry["current_complete_lanes"] == []
    assert registry["source_commit_eligibility"] == "content_addressed_not_commit_pinned"
    assert [row["stage"] for row in registry["staged_status"]] == list(EXPECTED_STAGES)
    assert registry["staged_status"][0]["status"] == "frozen"
    assert {row["status"] for row in registry["staged_status"][1:]} == {"unimplemented"}
    assert {owner["support_status"] for owner in registry["owners"]} == {
        "blocked_projector_unimplemented",
        "blocked_build_descriptor_unimplemented",
    }


def _assert_rejection_vocabulary(
    registry: dict[str, Any], schema: dict[str, Any]
) -> None:
    assert tuple(registry["rejection_codes"]) == EXPECTED_REJECTION_CODES
    assert tuple(schema["$defs"]["rejectionCode"]["enum"]) == EXPECTED_REJECTION_CODES


def test_rejection_vocabulary_is_the_literal_closed_set_and_has_no_decision() -> None:
    _assert_rejection_vocabulary(_load(DOCS / REGISTRY), _load(DOCS / REGISTRY_SCHEMA))
    assert len(EXPECTED_REJECTION_CODES) == len(set(EXPECTED_REJECTION_CODES))
    assert tuple(_load(DOCS / REGISTRY)["dependency_order"]) == EXPECTED_DEPENDENCY_ORDER
    assert "Decision" not in json.dumps(_load(DOCS / REGISTRY))


@pytest.mark.parametrize("drop", ("census_mismatch", "ledger_audit_failed"))
def test_dropping_a_rejection_code_coherently_is_still_refused(drop: str) -> None:
    registry = copy.deepcopy(_load(DOCS / REGISTRY))
    schema = copy.deepcopy(_load(DOCS / REGISTRY_SCHEMA))
    registry["rejection_codes"] = [
        code for code in registry["rejection_codes"] if code != drop
    ]
    schema["$defs"]["rejectionCode"]["enum"] = [
        code for code in schema["$defs"]["rejectionCode"]["enum"] if code != drop
    ]
    # Internally coherent, and still refused by the literal vocabulary.
    Draft202012Validator(schema).validate(registry)
    with pytest.raises(AssertionError):
        _assert_rejection_vocabulary(registry, schema)


def test_inventing_a_rejection_code_coherently_is_still_refused() -> None:
    registry = copy.deepcopy(_load(DOCS / REGISTRY))
    schema = copy.deepcopy(_load(DOCS / REGISTRY_SCHEMA))
    registry["rejection_codes"].append("invented_code")
    schema["$defs"]["rejectionCode"]["enum"].append("invented_code")
    Draft202012Validator(schema).validate(registry)
    with pytest.raises(AssertionError):
        _assert_rejection_vocabulary(registry, schema)


def test_fixture_pins_recompute_from_unchanged_artifact8_bytes() -> None:
    registry = _load(DOCS / REGISTRY)
    fixtures = ROOT / "tests" / "fixtures" / "b3"
    for lane in registry["lane_registry"]:
        assert (
            hashlib.sha256((fixtures / lane["artifact"]).read_bytes()).hexdigest()
            == lane["artifact_sha256"]
        )
        if lane["ledger"] is not None:
            assert (
                hashlib.sha256((fixtures / lane["ledger"]).read_bytes()).hexdigest()
                == lane["ledger_file_sha256"]
            )
    assert (
        hashlib.sha256(
            (fixtures / "artifact8-evidence-freeze-v1.json").read_bytes()
        ).hexdigest()
        == registry["freeze_manifest_sha256"]
    )
