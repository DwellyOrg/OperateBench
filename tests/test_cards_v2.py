# SPDX-License-Identifier: Apache-2.0
"""Independent executable contract checks for the additive Card v2 foundation."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import weakref
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator

import operatebench
from boundarybench.jsonsafe import canonical_json_bytes
from operatebench import cards
from operatebench.version import (
    CARD_SCHEMA_VERSION,
    OPERATION_CARD_V1_SCHEMA_VERSION,
    OPERATION_CARD_V2_SCHEMA_VERSION,
    SEMANTIC_SCENARIO_CARD_V1_SCHEMA_VERSION,
    SEMANTIC_SCENARIO_CARD_V2_SCHEMA_VERSION,
    VARIANT_CARD_V1_SCHEMA_VERSION,
    VARIANT_CARD_V2_SCHEMA_VERSION,
)

ROOT = Path(__file__).parents[1]
_V1_SPEC = importlib.util.spec_from_file_location(
    "_card_v1_test_helpers", ROOT / "tests/test_cards.py"
)
assert _V1_SPEC is not None and _V1_SPEC.loader is not None
_V1 = importlib.util.module_from_spec(_V1_SPEC)
_V1_SPEC.loader.exec_module(_V1)
_operation = cast(Any, _V1._operation)
_scenario = cast(Any, _V1._scenario)
_variant = cast(Any, _V1._variant)
DOCS = ROOT / "docs" / "schemas"
PACKAGED = ROOT / "src" / "operatebench" / "resources" / "cards"
V2_RESOURCES = (
    "card-defs-v2.schema.json",
    "operation-card-v2.schema.json",
    "semantic-scenario-card-v2.schema.json",
    "variant-card-v2.schema.json",
    "card-registry-v2.json",
    "identity-field-census-v2.schema.json",
    "identity-field-census-v2.registry.json",
)
V1_RESOURCE_SHA256 = {
    "card-defs-v1.schema.json": (
        "b2448b853f0b5844190a40030edc444ffcd2a11e1c63625ec83d3954dba836fd"
    ),
    "card-registry-v1.json": (
        "c6a4f67e422adcd25f36b300ffe4e26080220006bb5b2bfd8376417e695f5bbd"
    ),
    "identity-field-census-v1.registry.json": (
        "056b7acf26a0fcdc01b04abe7552b3f0c35a0ad70935434d01d94ef70ef1ef61"
    ),
    "identity-field-census-v1.schema.json": (
        "ab301b186bc6ff52abac8855cbff781713fb54dcf01a77d1409eddcda64b7c33"
    ),
    "operation-card-v1.schema.json": (
        "30d0d860125e28f80176ae65eb7c4283a800f15b8123c6134d25153be03c680a"
    ),
    "semantic-scenario-card-v1.schema.json": (
        "bf76011d58f2b2a4d6956b26324bedbcdb3b803b31daee66cef29dcfe71c664e"
    ),
    "variant-card-v1.schema.json": (
        "443e5a07566b8a8d659bf3f49cca28e604bf7dad32f1eca330b8f7fad7acb03b"
    ),
}
V2_RESOURCE_SHA256 = {
    "card-defs-v2.schema.json": (
        "b659b7032f0e0d7bf6a9b1f7dde4ab10db5328885a210222a48d9c11a22cdd32"
    ),
    "card-registry-v2.json": (
        "8dece73a16630d5164cd83cde0893f10a88ab47d973aa39c63d64caf13239b7d"
    ),
    "identity-field-census-v2.registry.json": (
        "278d324cb721693fe6f7cd3b59c8357d3c71c3a3ed8c5ae82e6b4bd256122046"
    ),
    "identity-field-census-v2.schema.json": (
        "f06fbbba730cdcbe73c8e8241ec7069600eb13f761772b2bea9e6584135d925c"
    ),
    "operation-card-v2.schema.json": (
        "f107e4446312d2a872209e1f12444f71251f95558f060f95ccfbe27550a15156"
    ),
    "semantic-scenario-card-v2.schema.json": (
        "d2f66dfd54c2a37d01380a597dabf46f7595e51a6f144236cb1ab85ecd948d2d"
    ),
    "variant-card-v2.schema.json": (
        "dac508e852107b9cca64ea0e22928add98350e8fed69bf1c6aa016d09c5f876e"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_card_v2_exact_dispatch_symbols_are_additive() -> None:
    assert cards.CARD_SCHEMAS == (
        "operatebench.operation_card.v1",
        "operatebench.semantic_scenario_card.v1",
        "operatebench.variant_card.v1",
    )
    assert cards.SUPPORTED_CARD_SCHEMA_VERSIONS == {
        "operatebench.operation_card.v1": 1,
        "operatebench.semantic_scenario_card.v1": 1,
        "operatebench.variant_card.v1": 1,
        "operatebench.operation_card.v2": 2,
        "operatebench.semantic_scenario_card.v2": 2,
        "operatebench.variant_card.v2": 2,
    }
    assert cards.OPERATION_CARD_V2_SCHEMA == "operatebench.operation_card.v2"
    assert cards.SEMANTIC_SCENARIO_CARD_V2_SCHEMA == (
        "operatebench.semantic_scenario_card.v2"
    )
    assert cards.VARIANT_CARD_V2_SCHEMA == "operatebench.variant_card.v2"
    assert CARD_SCHEMA_VERSION == 1
    assert operatebench.CARD_SCHEMA_VERSION == 1
    for name in (
        "OPERATION_CARD_V1_SCHEMA_VERSION",
        "OPERATION_CARD_V2_SCHEMA_VERSION",
        "SEMANTIC_SCENARIO_CARD_V1_SCHEMA_VERSION",
        "SEMANTIC_SCENARIO_CARD_V2_SCHEMA_VERSION",
        "VARIANT_CARD_V1_SCHEMA_VERSION",
        "VARIANT_CARD_V2_SCHEMA_VERSION",
    ):
        assert name not in operatebench.__all__
        assert not hasattr(operatebench, name)
    assert (
        OPERATION_CARD_V1_SCHEMA_VERSION,
        SEMANTIC_SCENARIO_CARD_V1_SCHEMA_VERSION,
        VARIANT_CARD_V1_SCHEMA_VERSION,
    ) == (1, 1, 1)
    assert (
        OPERATION_CARD_V2_SCHEMA_VERSION,
        SEMANTIC_SCENARIO_CARD_V2_SCHEMA_VERSION,
        VARIANT_CARD_V2_SCHEMA_VERSION,
    ) == (2, 2, 2)


def _schema_leaf_paths(schema: Mapping[str, Any], path: str) -> set[str]:
    """Derive card leaves from schema structure, not from the census under test."""
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.rsplit("/", 1)[-1] in {
        "operationCardRef",
        "semanticScenarioCardRef",
    }:
        definitions = json.loads((DOCS / "card-defs-v2.schema.json").read_text())["$defs"]
        return _schema_leaf_paths(definitions[ref.rsplit("/", 1)[-1]], path)
    if "$ref" in schema or path.endswith("/policy_contract"):
        return {path}
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        leaves: set[str] = set()
        for name, child in properties.items():
            assert isinstance(name, str) and isinstance(child, Mapping)
            leaves.update(_schema_leaf_paths(child, f"{path}/{name}"))
        return leaves
    items = schema.get("items")
    if isinstance(items, Mapping):
        return _schema_leaf_paths(items, path)
    branches = schema.get("oneOf")
    if isinstance(branches, list):
        object_leaves: set[str] = set()
        for branch in branches:
            assert isinstance(branch, Mapping)
            if isinstance(branch.get("properties"), Mapping):
                object_leaves.update(_schema_leaf_paths(branch, path))
        if object_leaves:
            return object_leaves
    return {path}


def _audit_v2_census(census: Mapping[str, Any]) -> None:
    prefixes = {
        "operation-card-v2.schema.json": ("/operation_card_v2", "operation_core_content"),
        "semantic-scenario-card-v2.schema.json": (
            "/semantic_scenario_card_v2",
            "semantic_scenario_content",
        ),
        "variant-card-v2.schema.json": ("/variant_card_v2", "variant_content"),
    }
    consumers = {
        "card_schema_content": ("package_loader", "card_validator"),
        "operation_core_content": (
            "card_loader",
            "card_validator",
            "identity_manifest_input",
        ),
        "semantic_scenario_content": (
            "card_loader",
            "card_validator",
            "identity_manifest_input",
        ),
        "variant_content": (
            "card_loader",
            "card_validator",
            "identity_manifest_input",
        ),
    }
    expected: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {}
    for name in V2_RESOURCES:
        expected[f"/contract_resources/{name}"] = (
            "card_schema_content",
            consumers["card_schema_content"],
            (f"docs/schemas/{name}",),
        )
    for name, (prefix, owner) in prefixes.items():
        schema = json.loads((DOCS / name).read_text())
        for path in _schema_leaf_paths(schema, prefix):
            expected[path] = (
                owner,
                consumers[owner],
                ("src/operatebench/cards.py", f"docs/schemas/{name}"),
            )

    entries = census["entries"]
    assert isinstance(entries, (list, tuple))
    actual_paths = [entry["field_path_pattern"] for entry in entries]
    assert len(expected) == 106
    assert Counter(actual_paths) == Counter(dict.fromkeys(expected, 1))
    for entry in entries:
        owner, expected_consumers, source_paths = expected[entry["field_path_pattern"]]
        assert entry["owner"] == owner
        assert tuple(entry["consumers"]) == expected_consumers
        assert tuple(entry["source_paths"]) == source_paths


def test_v2_registry_and_census_are_explicit_independent_resources() -> None:
    registry = cards.load_card_registry_v2()
    assert registry["schema"] == "operatebench.card_registry.v2"
    assert [row["schema_literal"] for row in registry["schema_documents"]] == [
        "operatebench.operation_card.v2",
        "operatebench.semantic_scenario_card.v2",
        "operatebench.variant_card.v2",
        "operatebench.card_defs.v2",
    ]
    assert registry["predicate_ast"]["quantifiers"] == [
        "ALL",
        "ANY",
        "NONE",
        "EXACTLY_ONE",
    ]
    assert "NULLABLE_STRING" in registry["field_shape_tags"]

    census = cards.load_field_census_v2()
    validated = cards.validate_field_census_v2(census, source_root=ROOT)
    assert validated["artifact_schema"] == "operatebench.b1_card_contract.v2"
    _audit_v2_census(validated)
    for mutation in (
        "omission",
        "invention",
        "duplicate",
        "owner",
        "consumer",
        "source",
    ):
        bad = deepcopy(census)
        if mutation == "omission":
            bad["entries"].pop()
        elif mutation == "invention":
            bad["entries"].append(deepcopy(bad["entries"][-1]))
            bad["entries"][-1]["field_path_pattern"] = "/invented"
        elif mutation == "duplicate":
            bad["entries"].append(deepcopy(bad["entries"][-1]))
        elif mutation == "owner":
            bad["entries"][-1]["owner"] = "operation_core_content"
        elif mutation == "consumer":
            bad["entries"][-1]["consumers"] = ["invented"]
        else:
            bad["entries"][-1]["source_paths"] = ["src/absent.py"]
        with pytest.raises(cards.CardValidationError):
            cards.validate_field_census_v2(bad, source_root=ROOT)
        with pytest.raises(AssertionError):
            _audit_v2_census(bad)


def _all_refs(value: object) -> list[str]:
    if isinstance(value, dict):
        refs = [value["$ref"]] if isinstance(value.get("$ref"), str) else []
        return refs + [ref for child in value.values() for ref in _all_refs(child)]
    if isinstance(value, list):
        return [ref for child in value for ref in _all_refs(child)]
    return []


def test_card_v2_resources_are_offline_meta_valid_and_byte_identical() -> None:
    for name, digest in V1_RESOURCE_SHA256.items():
        assert _sha256(DOCS / name) == digest
        assert _sha256(PACKAGED / name) == digest
    for name in V2_RESOURCES:
        assert (DOCS / name).read_bytes() == (PACKAGED / name).read_bytes()
        assert _sha256(DOCS / name) == V2_RESOURCE_SHA256[name]
    for name in (*V2_RESOURCES[:4], "identity-field-census-v2.schema.json"):
        Draft202012Validator.check_schema(json.loads((DOCS / name).read_text()))
    for schema_literal in cards.SUPPORTED_CARD_SCHEMA_VERSIONS:
        loaded = cards.load_schema(schema_literal)
        assert loaded["properties"]["schema"]["const"] == schema_literal
    definitions = json.loads((DOCS / "card-defs-v2.schema.json").read_text())["$defs"]
    for name in V2_RESOURCES[:4]:
        for ref in _all_refs(json.loads((DOCS / name).read_text())):
            prefixes = ("card-defs-v2.schema.json#/$defs/", "#/$defs/")
            prefix = next(
                (candidate for candidate in prefixes if ref.startswith(candidate)), None
            )
            assert prefix is not None
            assert ref.removeprefix(prefix) in definitions


def _prefix_item_nodes(value: object) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    if isinstance(value, dict):
        if "prefixItems" in value:
            found.append(value)
        for child in value.values():
            found.extend(_prefix_item_nodes(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_prefix_item_nodes(child))
    return found


def test_every_v2_prefix_items_array_is_exact_cardinality() -> None:
    discovered = 0
    for name in V2_RESOURCES:
        value = json.loads((DOCS / name).read_text())
        for node in _prefix_item_nodes(value):
            discovered += 1
            count = len(node["prefixItems"])  # type: ignore[arg-type]
            assert node["type"] == "array"
            assert node["minItems"] == node["maxItems"] == count
            assert node["items"] is False
    assert discovered == 0


def _repin(card: dict[str, Any]) -> dict[str, Any]:
    card["content_digest_sha256"] = cards.recompute_card_digest(card)
    return card


def _operation_v2() -> dict[str, Any]:
    operation = _operation()
    operation["schema"] = cards.OPERATION_CARD_V2_SCHEMA
    operation["schema_version"] = 2
    shape = operation["content"]["event_contracts"][0]["required"]["summary"]["shape"]
    shape["kind"] = "NULLABLE_STRING"
    return _repin(operation)


def _scenario_v2(operation: object) -> dict[str, Any]:
    scenario = _scenario(operation)
    scenario["schema"] = cards.SEMANTIC_SCENARIO_CARD_V2_SCHEMA
    scenario["schema_version"] = 2
    scenario["content"]["decision_points"][0]["reach_predicate"] = {
        "kind": "quantified",
        "quantifier": "EXACTLY_ONE",
        "collection": "/status",
        "binding": "record",
        "predicate": {
            "kind": "scalar",
            "left": {"kind": "binding", "binding": "record", "pointer": "/value"},
            "operator": "EQ",
            "right": {"kind": "literal", "value": "open"},
        },
    }
    return _repin(scenario)


def _variant_v2(scenario: object) -> dict[str, Any]:
    variant = _variant(scenario)
    variant["schema"] = cards.VARIANT_CARD_V2_SCHEMA
    variant["schema_version"] = 2
    return _repin(variant)


def test_v1_and_v2_canonical_card_goldens_are_literal_oracles() -> None:
    v1_cards = (_operation(), _scenario(_operation()), _variant(_scenario(_operation())))
    v1_expected = (
        (2154, "240e4678ba33db942344b84533f5cd6e5b0feb06cafb633836d2b41522f5ced0"),
        (971, "c7678d7ea9e93618003e31d0880a56008cba6eac070300353e22d4b5e38bfca6"),
        (1070, "036d691888b4adcca77ffde013623620c1bf8c54423755aecf635f5633710487"),
    )
    operation = cards.validate_card(_operation_v2())
    scenario = cards.validate_card(_scenario_v2(operation), references=[operation])
    variant = cards.validate_card(_variant_v2(scenario), references=[operation, scenario])
    v2_expected = (
        (2163, "601746f664a78307afb8d7b5717efcb1a38e70e123cab5ce397e36090a7c46b7"),
        (1092, "f0f8ecb53832cc90fef43c135982b8c5505ebeef8608c135aaf4149cf0b52052"),
        (1070, "06438e7468366af30a1d9eeab1987a04053f361ba8f734c4e0fe7b12cea9646c"),
    )
    for card, (byte_count, digest) in zip(
        (*v1_cards, operation, scenario, variant),
        (*v1_expected, *v2_expected),
        strict=True,
    ):
        encoded = canonical_json_bytes(cards.card_to_json(card), "golden Card")
        assert len(encoded) == byte_count
        assert hashlib.sha256(encoded).hexdigest() == digest


def test_v2_resource_loads_are_detached_cached_and_isolated_from_v1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cards._clear_resource_caches()
    reads: list[str] = []
    original = cards._read_resource_bytes

    def counted(name: str) -> bytes:
        reads.append(name)
        return original(name)

    monkeypatch.setattr(cards, "_read_resource_bytes", counted)
    try:
        for _ in range(2):
            cards.validate_card(_operation())
            cards.validate_card(_operation_v2())
        assert reads.count("operation-card-v1.schema.json") == 1
        assert reads.count("card-defs-v1.schema.json") == 1
        assert reads.count("operation-card-v2.schema.json") == 1
        assert reads.count("card-defs-v2.schema.json") == 1

        registry = cards.load_card_registry_v2()
        census = cards.load_field_census_v2()
        schema = cards.load_schema(cards.OPERATION_CARD_V2_SCHEMA)
        registry["schema"] = census["artifact_schema"] = schema["type"] = "poisoned"
        assert cards.load_card_registry_v2()["schema"] == "operatebench.card_registry.v2"
        assert cards.load_field_census_v2()["artifact_schema"] == (
            "operatebench.b1_card_contract.v2"
        )
        assert cards.load_schema(cards.OPERATION_CARD_V2_SCHEMA)["type"] == "object"
    finally:
        cards._clear_resource_caches()


def test_exact_dispatch_accepts_only_two_complete_triplets() -> None:
    v1 = (_operation(),)
    v2_operation = cards.validate_card(_operation_v2())
    v2_scenario = cards.validate_card(
        _scenario_v2(v2_operation), references=[v2_operation]
    )
    v2_variant = cards.validate_card(
        _variant_v2(v2_scenario), references=[v2_operation, v2_scenario]
    )
    assert [
        v2_operation["schema_version"],
        v2_scenario["schema_version"],
        v2_variant["schema_version"],
    ] == [2, 2, 2]

    for original in (*v1, _operation_v2()):
        for bad_version in (True, 0, 1 if original["schema_version"] == 2 else 2, 3):
            bad = deepcopy(original)
            bad["schema_version"] = bad_version
            with pytest.raises(cards.CardValidationError, match="schema/version"):
                cards.recompute_card_digest(bad)


def test_mixed_version_reference_chains_and_downgrade_attacks_fail_closed() -> None:
    v1_operation = cards.validate_card(_operation())
    v1_scenario = cards.validate_card(_scenario(v1_operation), references=[v1_operation])
    v2_operation = cards.validate_card(_operation_v2())
    v2_scenario = cards.validate_card(
        _scenario_v2(v2_operation), references=[v2_operation]
    )

    mixed_scenario = _scenario_v2(v1_operation)
    with pytest.raises(cards.CardValidationError):
        cards.validate_card(mixed_scenario, references=[v1_operation])

    mixed_variant = _variant_v2(v1_scenario)
    with pytest.raises(cards.CardValidationError):
        cards.validate_card(mixed_variant, references=[v1_operation, v1_scenario])

    downgraded = cards.card_to_json(v2_operation)
    downgraded["schema"] = cards.OPERATION_CARD_SCHEMA
    downgraded["schema_version"] = 1
    downgraded["content_digest_sha256"] = cards.recompute_card_digest(downgraded)
    with pytest.raises(cards.CardValidationError):
        cards.validate_card(downgraded)

    variant = _variant_v2(v2_scenario)
    with pytest.raises(cards.CardValidationError, match="ambiguous"):
        cards.validate_card(
            variant,
            references=[v2_operation, v2_scenario, cards.card_to_json(v2_scenario)],
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "operation_unknown_role",
        "operation_unknown_tool",
        "scenario_duplicate_order",
        "scenario_unresolved_causal_edge",
        "variant_unknown_actor_role",
        "variant_bad_payload",
    ),
)
def test_v2_chain_reuses_all_frozen_semantic_constraints(mutation: str) -> None:
    operation = _operation_v2()
    if mutation == "operation_unknown_role":
        operation["content"]["authority_contract"]["event_authority"][0][
            "required_authority"
        ] = "absent"
    elif mutation == "operation_unknown_tool":
        operation["content"]["retrieval_contract"]["evidence_requirements"][0][
            "required_tools"
        ] = ["absent"]
    operation = _repin(operation)
    if mutation.startswith("operation_"):
        with pytest.raises(cards.CardValidationError):
            cards.validate_card(operation)
        return

    validated_operation = cards.validate_card(operation)
    scenario = _scenario_v2(validated_operation)
    if mutation == "scenario_duplicate_order":
        duplicate = deepcopy(scenario["content"]["decision_points"][0])
        duplicate["point_id"] = "p2"
        scenario["content"]["decision_points"].append(duplicate)
    elif mutation == "scenario_unresolved_causal_edge":
        scenario["content"]["causal_graph"]["edges"] = [
            {
                "cause_node_id": "absent",
                "effect_node_id": "p1",
                "relation": "PRECEDES",
            }
        ]
    scenario = _repin(scenario)
    if mutation.startswith("scenario_"):
        with pytest.raises(cards.CardValidationError):
            cards.validate_card(scenario, references=[validated_operation])
        return

    validated_scenario = cards.validate_card(scenario, references=[validated_operation])
    variant = _variant_v2(validated_scenario)
    if mutation == "variant_unknown_actor_role":
        variant["content"]["actors"][0]["role_id"] = "absent"
    else:
        variant["content"]["events"][0]["payload"]["summary"] = 42
    with pytest.raises(cards.CardValidationError):
        cards.validate_card(
            _repin(variant), references=[validated_operation, validated_scenario]
        )


def test_nullable_string_contract_accepts_null_and_constrained_strings() -> None:
    operation = cards.validate_card(_operation_v2())
    scenario = cards.validate_card(_scenario_v2(operation), references=[operation])
    shape = _operation_v2()["content"]["event_contracts"][0]["required"]["summary"][
        "shape"
    ]
    shape.update(min_length=1, max_length=4, pattern="^[a-z]+$", enum=["leak"])
    operation_card = _operation_v2()
    operation_card["content"]["event_contracts"][0]["required"]["summary"]["shape"] = (
        shape
    )
    operation = cards.validate_card(_repin(operation_card))
    scenario = cards.validate_card(_scenario_v2(operation), references=[operation])

    for accepted in (None, "leak"):
        variant = _variant_v2(scenario)
        variant["content"]["events"][0]["payload"]["summary"] = accepted
        assert cards.validate_card(_repin(variant), references=[operation, scenario])
    for refused in ("", "toolong", "LEAK", "other", True, 1, []):
        variant = _variant_v2(scenario)
        variant["content"]["events"][0]["payload"]["summary"] = refused
        with pytest.raises(cards.CardValidationError):
            cards.validate_card(_repin(variant), references=[operation, scenario])

    v1 = _operation()
    v1["content"]["event_contracts"][0]["required"]["summary"]["shape"] = shape
    with pytest.raises(cards.CardValidationError):
        cards.validate_card(_repin(v1))


def _exactly_one(value: object) -> dict[str, Any]:
    return {
        "kind": "quantified",
        "quantifier": "EXACTLY_ONE",
        "collection": "/records",
        "binding": "record",
        "predicate": {
            "kind": "scalar",
            "left": {"kind": "binding", "binding": "record", "pointer": "/value"},
            "operator": "EQ",
            "right": {"kind": "literal", "value": value},
        },
    }


def test_exactly_one_truth_binding_malformed_and_type_sensitive_vectors() -> None:
    for records, expected, bound_key in (
        ({}, False, None),
        ({"b": {"value": 1}, "a": {"value": 2}}, True, "b"),
        ({"b": {"value": 1}, "a": {"value": 1}}, False, None),
        ({"a": 1, "b": {"value": 1}}, True, "b"),
    ):
        truth, bindings = cards._evaluate_predicate_v2(
            _exactly_one(1), {"records": records}
        )
        assert truth is expected
        if bound_key is None:
            assert bindings is None
        else:
            assert bindings is not None
            assert bindings["record"] is records[bound_key]

    for malformed in (None, [], "records", 1, True):
        assert cards._evaluate_predicate_v2(_exactly_one(1), {"records": malformed}) == (
            False,
            None,
        )

    # Exact JSON scalar equality: bool is not integer one.
    assert cards._evaluate_predicate_v2(
        _exactly_one(1), {"records": {"a": {"value": True}}}
    ) == (False, None)

    nested = {
        "kind": "all",
        "operands": [
            _exactly_one("open"),
            {
                "kind": "scalar",
                "left": {
                    "kind": "binding",
                    "binding": "record",
                    "pointer": "/name",
                },
                "operator": "EQ",
                "right": {"kind": "literal", "value": "first"},
            },
        ],
    }
    truth, bindings = cards._evaluate_predicate_v2(
        nested,
        {
            "records": {
                "z": {"value": "closed", "name": "last"},
                "a": {"value": "open", "name": "first"},
            }
        },
    )
    assert truth is True and bindings is not None
    assert bindings["record"]["name"] == "first"


# Reviewer-blocker regressions: these are intentionally independent boundary oracles.
def _walk_patterns(value: object) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        pattern = value.get("pattern")
        if isinstance(pattern, str):
            found.append(pattern)
        for child in value.values():
            found.extend(_walk_patterns(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_patterns(child))
    return found


def test_all_v2_exact_languages_use_portable_absolute_end_and_reject_suffixes() -> None:
    names = (*V2_RESOURCES[:4], "identity-field-census-v2.schema.json")
    patterns = [
        pattern
        for name in names
        for pattern in _walk_patterns(json.loads((DOCS / name).read_text()))
    ]
    assert len(patterns) == 94
    assert len(set(patterns)) == 10
    assert all(pattern.endswith("$(?![\\s\\S])") for pattern in patterns)
    samples: dict[str, str] = {}
    for pattern in set(patterns):
        if "0-9a-f" in pattern and "{64}" in pattern:
            sample = "a" * 64
        elif "pyproject" in pattern:
            sample = "src/x.py"
        elif ("~0|~1" in pattern and "~*" in pattern) or "~0|~1" in pattern:
            sample = "/field"
        elif "A-Z]{3}" in pattern:
            sample = "USD"
        elif "T(?:[01]" in pattern:
            sample = "2026-08-25T12:34:56.123456Z"
        elif "0|[1-9]" in pattern and "A-Za-z" in pattern:
            sample = "1.2.3"
        elif "@" in pattern:
            sample = "name@1.2.3"
        elif "x20" in pattern:
            sample = "^a$"
        else:
            sample = "name"
        samples[pattern] = sample
    assert len(samples) == 10
    for pattern, sample in samples.items():
        validator = Draft202012Validator({"type": "string", "pattern": pattern})
        assert not list(validator.iter_errors(sample)), pattern
        assert cards._pattern_fullmatch(pattern, sample, "$", trusted_pattern=True)
        for suffix in ("\n", "\r", "\u2028"):
            assert list(validator.iter_errors(sample + suffix)), (pattern, suffix)
            assert not cards._pattern_fullmatch(
                pattern, sample + suffix, "$", trusted_pattern=True
            )


def _nested_quantifier(outer_binding: str, inner_binding: str) -> dict[str, Any]:
    return {
        "kind": "quantified",
        "quantifier": "EXACTLY_ONE",
        "collection": "/records",
        "binding": outer_binding,
        "predicate": {
            "kind": "quantified",
            "quantifier": "EXACTLY_ONE",
            "collection": "/children",
            "binding": inner_binding,
            "predicate": {
                "kind": "scalar",
                "left": {"kind": "binding", "binding": inner_binding, "pointer": "/ok"},
                "operator": "EQ",
                "right": {"kind": "literal", "value": True},
            },
        },
    }


def test_nested_binding_redeclaration_refused_and_distinct_bindings_preserved() -> None:
    with pytest.raises(cards.CardValidationError, match="redeclare"):
        cards._validate_predicate(
            _nested_quantifier("record", "record"), {"/records", "/children"}, set()
        )
    outer = {"outer": True}
    child = {"ok": True}
    truth, bindings = cards._evaluate_predicate_v2(
        _nested_quantifier("record", "child"),
        {"records": {"outer": outer}, "children": {"child": child}},
    )
    assert truth is True and bindings is not None
    assert bindings["record"] is outer
    assert bindings["child"] is child
    for records in ({}, {"a": outer, "b": outer}):
        truth, bindings = cards._evaluate_predicate_v2(
            _nested_quantifier("record", "child"),
            {"records": records, "children": {"child": child}},
        )
        assert (truth, bindings) == (False, None)


def test_quantifier_sibling_scopes_bool_int_and_malformed_keys_are_deterministic() -> (
    None
):
    sibling = {
        "kind": "any",
        "operands": [_exactly_one(1), _exactly_one(2)],
    }
    cards._validate_predicate(sibling, {"/records"}, set())
    assert cards._evaluate_predicate_v2(sibling, {"records": {"a": {"value": 2}}})[0]
    assert cards._evaluate_predicate_v2(
        _exactly_one(1), {"records": {"a": {"value": True}}}
    ) == (False, None)
    assert cards._evaluate_predicate_v2(
        _exactly_one(True), {"records": {"a": {"value": True}}}
    )[0]
    for malformed in ({"a": {"value": 1}, 1: {"value": 1}}, {True: {"value": 1}}):
        assert cards._evaluate_predicate_v2(_exactly_one(1), {"records": malformed}) == (
            False,
            None,
        )


def test_v2_card_ref_fragment_is_v2_only() -> None:
    definitions = json.loads((DOCS / "card-defs-v2.schema.json").read_text())["$defs"]
    fragment = definitions["cardRef"]
    assert ".v1" not in json.dumps(definitions, sort_keys=True)
    for schema in (
        cards.OPERATION_CARD_V2_SCHEMA,
        cards.SEMANTIC_SCENARIO_CARD_V2_SCHEMA,
        cards.VARIANT_CARD_V2_SCHEMA,
    ):
        cards._validate_schema(
            {"schema": schema, "id": "x", "content_digest_sha256": "a" * 64},
            fragment,
            "$",
            defs_file="card-defs-v2.schema.json",
        )
    with pytest.raises(cards.CardValidationError):
        cards._validate_schema(
            {
                "schema": cards.OPERATION_CARD_SCHEMA,
                "id": "x",
                "content_digest_sha256": "a" * 64,
            },
            fragment,
            "$",
            defs_file="card-defs-v2.schema.json",
        )


class _CountingSequence(Sequence[object]):
    def __init__(self, count: int | None, item: object = None) -> None:
        self.count = count
        self.item = item
        self.reads = 0

    def __len__(self) -> int:
        raise RuntimeError("len must not be called")

    def __getitem__(self, index: int) -> object:
        self.reads += 1
        if self.count is not None and index >= self.count:
            raise IndexError
        return self.item


class _ItemsMapping(Mapping[object, object]):
    def __init__(self, pairs: list[tuple[object, object]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        raise RuntimeError("len must not be called")

    def __iter__(self):  # type: ignore[no-untyped-def]
        raise RuntimeError("mapping iteration must not be called")

    def __getitem__(self, key: object) -> object:
        raise RuntimeError("getitem must not be called")

    def items(self):  # type: ignore[no-untyped-def]
        return iter(self.pairs)


@pytest.mark.parametrize("hook", ("items", "iter", "getitem"))
def test_projection_translates_hostile_protocol_exceptions(hook: str) -> None:
    class Hostile(_ItemsMapping):
        def items(self):  # type: ignore[no-untyped-def]
            if hook == "items":
                raise RuntimeError("hostile")
            return super().items()

    class HostileSequence(_CountingSequence):
        def __iter__(self):  # type: ignore[no-untyped-def]
            if hook == "iter":
                raise RuntimeError("hostile")
            return super().__iter__()

        def __getitem__(self, index: int) -> object:
            if hook == "getitem":
                raise TypeError("hostile")
            return super().__getitem__(index)

    value: object = HostileSequence(0) if hook in {"iter", "getitem"} else Hostile([])
    with pytest.raises(cards.CardValidationError, match="malformed") as caught:
        cards.card_to_json(value)
    if hook != "getitem":
        assert caught.value.__cause__ is None


def test_projection_duplicate_mixed_keys_and_width_are_bounded() -> None:
    for pairs in (
        [("x", 1), ("x", 2)],
        [("x", 1), (1, 2)],
        [(1, 2), ("x", 1)],
    ):
        with pytest.raises(cards.CardValidationError):
            cards.card_to_json(_ItemsMapping(pairs))
    assert len(cards.card_to_json({str(i): None for i in range(4096)})) == 4096
    with pytest.raises(cards.CardValidationError, match="4096"):
        cards.card_to_json(_ItemsMapping([(str(i), None) for i in range(4097)]))
    sequence = _CountingSequence(None)
    with pytest.raises(cards.CardValidationError, match="4096"):
        cards.card_to_json(sequence)
    assert sequence.reads <= 4097

    class CountingItems(_ItemsMapping):
        def __init__(self) -> None:
            super().__init__([])
            self.reads = 0

        def items(self):  # type: ignore[no-untyped-def]
            while True:
                self.reads += 1
                yield str(self.reads), None

    mapping = CountingItems()
    with pytest.raises(cards.CardValidationError, match="4096"):
        cards.card_to_json(mapping)
    assert mapping.reads <= 4097


def test_projection_exact_string_aggregate_node_depth_and_canonical_limits() -> None:
    mib = 1_048_576
    aggregate = 8_388_608
    assert cards.card_to_json("x" * mib) == "x" * mib
    with pytest.raises(cards.CardValidationError):
        cards.card_to_json("x" * (mib + 1))
    exact_aggregate = ["x" * mib for _ in range(8)]
    assert cards.card_to_json(exact_aggregate) == exact_aggregate
    with pytest.raises(cards.CardValidationError):
        cards.card_to_json([*exact_aggregate, "x"])

    exact_nodes: list[object] = [[None] * 4096 for _ in range(24)]
    exact_nodes.append([None] * 1670)
    assert cards.card_to_json(exact_nodes) == exact_nodes
    exact_nodes[-1].append(None)  # type: ignore[union-attr]
    with pytest.raises(cards.CardValidationError, match="100000"):
        cards.card_to_json(exact_nodes)

    depth: object = None
    for _ in range(64):
        depth = [depth]
    assert cards.card_to_json(depth) == depth
    with pytest.raises(cards.CardValidationError, match="64"):
        cards.card_to_json([depth])

    # Eight strings make canonical punctuation independently decisive.
    overhead = 1 + 8 * 2 + 7 + 1
    lengths = [mib] * 7 + [aggregate - overhead - 7 * mib]
    exact_bytes = ["x" * length for length in lengths]
    with pytest.raises(cards.CardValidationError) as exact_error:
        cards.validate_card(exact_bytes)  # type: ignore[arg-type]
    assert "canonicalized" not in str(exact_error.value)
    exact_bytes[-1] += "x"
    with pytest.raises(cards.CardValidationError, match="canonicalized"):
        cards.validate_card(exact_bytes)  # type: ignore[arg-type]


def test_versioned_card_json_loaders_are_generation_exact() -> None:
    operation_v1 = cards.validate_card(_operation())
    scenario_v1 = cards.validate_card(_scenario(operation_v1), references=[operation_v1])
    variant_v1 = cards.validate_card(
        _variant(scenario_v1), references=[operation_v1, scenario_v1]
    )
    operation_v2 = cards.validate_card(_operation_v2())
    scenario_v2 = cards.validate_card(
        _scenario_v2(operation_v2), references=[operation_v2]
    )
    variant_v2 = cards.validate_card(
        _variant_v2(scenario_v2), references=[operation_v2, scenario_v2]
    )

    for card, references in (
        (operation_v1, []),
        (scenario_v1, [operation_v1]),
        (variant_v1, [operation_v1, scenario_v1]),
    ):
        encoded = json.dumps(cards.card_to_json(card))
        assert cards.load_card_json(encoded, references=references) == card
        with pytest.raises(cards.CardValidationError, match="Card v2 schema/version"):
            cards.load_card_json_v2(encoded, references=references)
    for card, references in (
        (operation_v2, []),
        (scenario_v2, [operation_v2]),
        (variant_v2, [operation_v2, scenario_v2]),
    ):
        encoded = json.dumps(cards.card_to_json(card))
        assert cards.load_card_json_v2(encoded, references=references) == card
        with pytest.raises(cards.CardValidationError, match="Card v1 schema/version"):
            cards.load_card_json(encoded, references=references)

    for bad in (
        {"schema": "operatebench.operation_card.v3", "schema_version": 3},
        {"schema": cards.OPERATION_CARD_SCHEMA, "schema_version": 2},
        {"schema": cards.OPERATION_CARD_V2_SCHEMA, "schema_version": 1},
    ):
        with pytest.raises(cards.CardValidationError, match="Card v1 schema/version"):
            cards.load_card_json(json.dumps(bad))
        with pytest.raises(cards.CardValidationError, match="Card v2 schema/version"):
            cards.load_card_json_v2(json.dumps(bad))


def test_load_card_json_v2_raw_utf8_limit_and_surrogate_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = 8_388_608
    raw = json.dumps(_operation_v2()).encode("utf-8")
    exact = raw + b" " * (limit - len(raw))
    assert cards.load_card_json_v2(exact)["schema"] == cards.OPERATION_CARD_V2_SCHEMA

    def parser_sentinel(_raw: bytes, _context: str) -> object:
        raise AssertionError("over-limit v2 reached the JSON parser")

    monkeypatch.setattr(cards, "_load_json_bytes", parser_sentinel)
    with pytest.raises(cards.CardValidationError, match="raw UTF-8") as over:
        cards.load_card_json_v2(exact + b" ")
    assert over.value.__cause__ is None
    with pytest.raises(cards.CardValidationError) as surrogate:
        cards.load_card_json_v2("\ud800")
    assert surrogate.value.__cause__ is None


def test_cards_module_all_is_exact_and_wildcard_importable() -> None:
    expected = [
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
    ]
    assert cards.__all__ == expected
    namespace: dict[str, object] = {}
    exec("from operatebench.cards import *", {}, namespace)
    assert {name for name in namespace if not name.startswith("__")} == set(expected)


def test_load_card_json_v2_source_and_path_errors_are_domain_bounded(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-v2-card.json"
    with pytest.raises(
        cards.CardValidationError, match="Card v2 JSON path"
    ) as path_error:
        cards.load_card_json_v2(missing)
    assert isinstance(path_error.value.__cause__, FileNotFoundError)

    with pytest.raises(
        cards.CardValidationError, match="malformed Card v2 JSON source"
    ) as source:
        cards.load_card_json_v2(None)  # type: ignore[arg-type]
    assert type(source.value.__cause__) is TypeError


def test_reference_graph_validates_every_target_and_refuses_extraneous() -> None:
    operation = cards.validate_card(_operation_v2())
    scenario = cards.validate_card(_scenario_v2(operation), references=[operation])
    variant = cards.validate_card(_variant_v2(scenario), references=[operation, scenario])
    malformed_variant = {"schema": cards.VARIANT_CARD_V2_SCHEMA, "id": "bad"}
    with pytest.raises(cards.CardValidationError, match="invalid card reference target"):
        cards.validate_card(scenario, references=[operation, malformed_variant])
    with pytest.raises(cards.CardValidationError, match="extraneous"):
        cards.validate_card(operation, references=[variant, scenario, operation])
    stale = cards.card_to_json(operation)
    stale["content_digest_sha256"] = "0" * 64
    with pytest.raises(cards.CardValidationError, match="invalid card reference target"):
        cards.validate_card(scenario, references=[stale])
    assert cards.validate_card(_variant_v2(scenario), references=[operation, scenario])


def test_v2_census_has_independent_literal_nullability_oracle() -> None:
    census = cards.load_field_census_v2()
    expected = {
        (entry["owner"], entry["field_path_pattern"]): "never"
        for entry in census["entries"]
    }
    assert len(expected) == 106
    assert frozenset() == cards._V2_EXPLICIT_NULL_ALLOWED
    assert set(expected.values()) == {"never"}
    for owner in {entry["owner"] for entry in census["entries"]}:
        bad = deepcopy(census)
        row = next(entry for entry in bad["entries"] if entry["owner"] == owner)
        row["nullability"] = "explicit_null_allowed"
        with pytest.raises(cards.CardValidationError, match="nullability"):
            cards.validate_field_census_v2(bad)


class _RetentionBacking:
    def __init__(self) -> None:
        self.storage = bytearray(4 * 1024 * 1024)


class _RetentionMapping(Mapping[str, object]):
    def __init__(self) -> None:
        self.backing = _RetentionBacking()

    def __getitem__(self, key: str) -> object:
        raise RuntimeError("hostile mapping getitem")

    def __iter__(self):  # type: ignore[no-untyped-def]
        raise RuntimeError("hostile mapping iteration")

    def __len__(self) -> int:
        raise RuntimeError("hostile mapping length")

    def items(self):  # type: ignore[no-untyped-def]
        raise RuntimeError("hostile mapping items")


class _RetentionSequence(Sequence[object]):
    def __init__(self) -> None:
        self.backing = _RetentionBacking()

    def __len__(self) -> int:
        raise RuntimeError("hostile sequence length")

    def __getitem__(self, index: int) -> object:
        raise RuntimeError("hostile sequence getitem")


def _invoke_retention_boundary(
    boundary: str,
) -> tuple[
    cards.CardValidationError,
    weakref.ReferenceType[_RetentionMapping | _RetentionSequence],
    weakref.ReferenceType[_RetentionBacking],
    set[int],
]:
    """Invoke one public boundary without returning a caller-owned raw input root."""
    hostile: _RetentionMapping | _RetentionSequence | None = (
        _RetentionSequence()
        if "sequence" in boundary
        or boundary in {"reference_iterable", "load_json_unvisited_references"}
        else _RetentionMapping()
    )
    root_ref = weakref.ref(hostile)
    backing_ref = weakref.ref(hostile.backing)
    forbidden = {id(hostile), id(hostile.backing), id(hostile.backing.storage)}
    root: object = None
    child: object = None
    references: object = None
    operation: object = None
    error: cards.CardValidationError | None = None
    try:
        child = hostile

        if boundary in {"card_to_json_root_mapping", "card_to_json_root_sequence"}:
            root = child
            cards.card_to_json(root)
        elif boundary.startswith("card_to_json_nested"):
            root = {"nested": child}
            cards.card_to_json(root)
        elif boundary in {
            "validate_card_nested_mapping",
            "recompute_digest_nested_sequence",
        }:
            root = _operation_v2()
            root["content"]["summary"] = child  # type: ignore[index]
            if boundary == "validate_card_nested_mapping":
                cards.validate_card(root)  # type: ignore[arg-type]
            else:
                cards.recompute_card_digest(root)  # type: ignore[arg-type]
        elif boundary in {"reference_iterable", "reference_node"}:
            operation = cards.validate_card(_operation_v2())
            root = _scenario_v2(operation)
            references = child if boundary == "reference_iterable" else [child]
            cards.validate_card(root, references=references)  # type: ignore[arg-type]
        elif boundary in {
            "field_census_nested_mapping",
            "field_census_v2_nested_sequence",
        }:
            root = (
                cards.load_field_census()
                if boundary == "field_census_nested_mapping"
                else cards.load_field_census_v2()
            )
            root["entries"] = child  # type: ignore[index]
            if boundary == "field_census_nested_mapping":
                cards.validate_field_census(root)  # type: ignore[arg-type]
            else:
                cards.validate_field_census_v2(root)  # type: ignore[arg-type]
        else:
            references = child
            cards.load_card_json("{", references=references)  # type: ignore[arg-type]
    except cards.CardValidationError as caught:
        error = caught
    finally:
        hostile = None
        root = None
        child = None
        references = None
        operation = None

    assert error is not None
    return error, root_ref, backing_ref, forbidden


def _assert_detached_error_graph(error: BaseException, forbidden: set[int]) -> None:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        assert not isinstance(current, RuntimeError)
        assert all(id(argument) not in forbidden for argument in current.args)
        assert (
            sum(len(argument) for argument in current.args if type(argument) is str)
            < 2_000_000
        )
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        pending.extend(getattr(current, "exceptions", ()))
        traceback = current.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            if frame.f_code.co_filename == cards.__file__ or (
                frame.f_code.co_filename == __file__
                and frame.f_code.co_name == "_invoke_retention_boundary"
            ):
                assert all(
                    id(value) not in forbidden for value in frame.f_locals.values()
                )
            traceback = traceback.tb_next


@pytest.mark.parametrize(
    "boundary",
    (
        "card_to_json_root_mapping",
        "card_to_json_nested_mapping",
        "card_to_json_root_sequence",
        "card_to_json_nested_sequence",
        "validate_card_nested_mapping",
        "recompute_digest_nested_sequence",
        "reference_iterable",
        "reference_node",
        "field_census_nested_mapping",
        "field_census_v2_nested_sequence",
        "load_json_unvisited_references",
    ),
)
def test_hostile_protocol_failures_do_not_retain_library_graphs(boundary: str) -> None:
    error, root_ref, backing_ref, forbidden = _invoke_retention_boundary(boundary)

    assert error.__cause__ is None
    assert error.__context__ is None
    _assert_detached_error_graph(error, forbidden)
    gc.collect()
    assert root_ref() is None
    assert backing_ref() is None


@pytest.mark.parametrize("exception_type", (RuntimeError, TypeError, KeyboardInterrupt))
def test_trusted_projection_helper_failures_propagate_raw(
    monkeypatch: pytest.MonkeyPatch, exception_type: type[BaseException]
) -> None:
    failure = exception_type("trusted helper sentinel")

    def fail_node(_budget: cards._ProjectionBudget) -> None:
        raise failure

    monkeypatch.setattr(cards._ProjectionBudget, "node", fail_node)
    with pytest.raises(exception_type) as caught:
        cards.card_to_json({})
    assert caught.value is failure
    assert caught.value.__traceback__ is not None
