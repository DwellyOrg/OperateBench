from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy
from enum import IntEnum
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import pytest

import operatebench
import operatebench.cards as cards_module
from operatebench.cards import (
    CardValidationError,
    card_to_json,
    load_card_json,
    load_card_registry,
    load_field_census,
    load_schema,
    recompute_card_digest,
    validate_card,
    validate_field_census,
)
from operatebench.jsonsafe import MAX_JSON_DEPTH


def _operation() -> dict[str, object]:
    content = {
        "operation_type": "lettings.maintenance.synthetic",
        "operation_version": "0.5.0",
        "state_contract": {
            "schema_id": "maintenance.state.v1",
            "initial_state_schema": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "minLength": 0,
                        "maxLength": 255,
                        "pattern": "^[a-z]+$",
                        "enum": ["open"],
                    }
                },
                "required": ["status"],
                "additionalProperties": False,
            },
            "canonical_state_projection": ["/status"],
            "replay_final_pointer": "/status",
            "terminal_pointer": "/status",
        },
        "authority_contract": {
            "roles": [{"role_id": "tenant", "authorities": ["report.issue"]}],
            "event_authority": [
                {"event_type": "issue.reported", "required_authority": "report.issue"}
            ],
            "action_authority": [
                {"action_type": "case.close", "required_authority": "report.issue"}
            ],
        },
        "policy_contract": {
            "schema": "operatebench.policy.lettings.maintenance.v1",
            "currency": "GBP",
            "issue_classification": "repair",
            "approval_threshold_minor": 1,
            "approval_reminder_after_minutes": 1,
            "approval_deadline_after_minutes": 2,
            "approval_validity_minutes": 3,
            "provisional_close_minutes": 4,
            "visit_followup_after_minutes": 5,
            "completion_notice_within_minutes": 6,
            "max_turns_per_invocation": 7,
            "horizon_minutes": 8,
            "max_retrieval_batches_per_invocation": 9,
        },
        "event_contracts": [
            {
                "event_type": "issue.reported",
                "required": {
                    "summary": {
                        "required": True,
                        "shape": {
                            "kind": "string",
                            "min_length": 0,
                            "max_length": 255,
                            "pattern": None,
                            "enum": None,
                        },
                    }
                },
                "optional": {},
            }
        ],
        "action_contracts": [
            {
                "action_type": "case.close",
                "required": {},
                "optional": {},
                "evidence_refs": "forbidden",
            }
        ],
        "retrieval_contract": {
            "catalogue": [
                {
                    "tool_id": "case.read",
                    "source": "case",
                    "authority": "report.issue",
                    "schema_id": "case.record.v1",
                    "arguments": {},
                }
            ],
            "record_identity_algorithm": "sha256.canonical-json.v1",
            "evidence_requirements": [
                {
                    "outcome_type": "closed",
                    "selector_id": "closure",
                    "required_tools": ["case.read"],
                }
            ],
        },
        "transition_contract": {
            "reducer_ids": ["maintenance.reduce.v1"],
            "obligation_types": ["close.case"],
            "terminal_kinds": ["closed"],
            "common_predicate_inputs": ["/status"],
            "declared_max_domain_generated_audit_events": 0,
        },
    }
    card: dict[str, object] = {
        "schema": "operatebench.operation_card.v1",
        "schema_version": 1,
        "id": "maintenance.core@0.5.0",
        "content": content,
        "content_digest_sha256": "0" * 64,
    }
    card["content_digest_sha256"] = recompute_card_digest(card)
    return card


def _scenario(operation: object) -> dict[str, object]:
    operation_card = card_to_json(operation)
    card: dict[str, object] = {
        "schema": "operatebench.semantic_scenario_card.v1",
        "schema_version": 1,
        "id": "maintenance.scenario",
        "content": {
            "operation_core": {
                key: operation_card[key]
                for key in ("schema", "id", "content_digest_sha256")
            },
            "construct_label": "lifecycle",
            "common_estimand_id": "common.y.v1",
            "decision_points": [
                {
                    "point_id": "p1",
                    "order": 0,
                    "reach_predicate": {
                        "kind": "scalar",
                        "left": {"kind": "pointer", "pointer": "/status"},
                        "operator": "EQ",
                        "right": {"kind": "literal", "value": "open"},
                    },
                    "required_prior_point_ids": [],
                    "common_obligation_id": "o1",
                    "admissible_effect_classes": ["close"],
                }
            ],
            "causal_graph": {
                "nodes": [{"node_id": "p1", "semantic_type": "decision"}],
                "edges": [],
            },
            "authored_hazards": [],
            "common_outcome_obligations": [
                {
                    "obligation_id": "o1",
                    "predicate_id": "closed",
                    "terminal_required": True,
                }
            ],
            "allowed_variant_dimensions": [],
        },
        "content_digest_sha256": "0" * 64,
    }
    card["content_digest_sha256"] = recompute_card_digest(card)
    return card


def _variant(scenario: object) -> dict[str, object]:
    scenario_card = card_to_json(scenario)
    card: dict[str, object] = {
        "schema": "operatebench.variant_card.v1",
        "schema_version": 1,
        "id": "maintenance.variant",
        "content": {
            "semantic_scenario": {
                key: scenario_card[key]
                for key in ("schema", "id", "content_digest_sha256")
            },
            "variant_label": "baseline",
            "starts_at": "2026-08-21T00:00:00.000000Z",
            "expected_terminal": "closed",
            "human_checkpoint_budget": 0,
            "required_checkpoint_types": [],
            "dispatch_failures": [],
            "expected_event_rejections": [],
            "actors": [{"actor_id": "tenant.one", "role_id": "tenant"}],
            "hidden_initial_values": {"status": "open"},
            "message_fixtures": [
                {
                    "fixture_id": "message.one",
                    "surface_content_digest_sha256": "a" * 64,
                }
            ],
            "events": [
                {
                    "event_id": "event.one",
                    "event_type": "issue.reported",
                    "actor_id": "tenant.one",
                    "authored_sequence": 0,
                    "at": "2026-08-21T00:00:00.000000Z",
                    "trigger": None,
                    "delay_minutes": 0,
                    "triggers_agent": True,
                    "payload": {"summary": "leak"},
                }
            ],
        },
        "content_digest_sha256": "0" * 64,
    }
    card["content_digest_sha256"] = recompute_card_digest(card)
    return card


def test_packaged_resources_are_available_and_match_normative_docs() -> None:
    registry = load_card_registry()
    assert registry["schema"] == "operatebench.card_registry.v1"
    packaged = files("operatebench.resources.cards").joinpath("card-registry-v1.json")
    normative = files("operatebench").joinpath("../../docs/schemas/card-registry-v1.json")
    assert packaged.read_bytes() == normative.read_bytes()
    assert load_schema("operatebench.operation_card.v1")["type"] == "object"


def test_public_resource_loads_are_detached_and_internal_reads_are_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cards_module._clear_resource_caches()
    reads: list[str] = []
    original_read = cards_module._read_resource_bytes

    def counted_read(name: str) -> bytes:
        reads.append(name)
        return original_read(name)

    monkeypatch.setattr(cards_module, "_read_resource_bytes", counted_read)

    for _ in range(2):
        validate_card(_operation())
    assert reads.count("operation-card-v1.schema.json") == 1
    assert reads.count("card-defs-v1.schema.json") == 1

    registry = load_card_registry()
    schema = load_schema("operatebench.operation_card.v1")
    census = load_field_census()
    registry["schema"] = "poisoned"
    schema["type"] = "poisoned"
    census["artifact_schema"] = "poisoned"
    assert load_card_registry()["schema"] == "operatebench.card_registry.v1"
    assert load_schema("operatebench.operation_card.v1")["type"] == "object"
    assert load_field_census()["artifact_schema"] == "operatebench.b1_card_contract.v1"

    cards_module._clear_resource_caches()


def test_malformed_packaged_defs_fail_closed_through_public_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cards_module._clear_resource_caches()
    original_read = cards_module._read_resource_bytes

    def malformed_defs(name: str) -> bytes:
        if name == "card-defs-v1.schema.json":
            return b'{"$defs": []}'
        return original_read(name)

    monkeypatch.setattr(cards_module, "_read_resource_bytes", malformed_defs)
    try:
        with pytest.raises(CardValidationError, match=r"card-defs.*\$defs"):
            validate_card(_operation())
    finally:
        cards_module._clear_resource_caches()


def test_card_trust_paths_do_not_depend_on_optimization_sensitive_asserts() -> None:
    source = Path(cards_module.__file__).read_text(encoding="utf-8")
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(ast.parse(source)))


@pytest.mark.parametrize(
    ("loader", "resource_name"),
    (
        (
            lambda: load_schema("operatebench.operation_card.v1"),
            "operation-card-v1.schema.json",
        ),
        (load_card_registry, "card-registry-v1.json"),
        (load_field_census, "identity-field-census-v1.registry.json"),
    ),
)
@pytest.mark.parametrize("error_type", (OSError, FileNotFoundError))
def test_public_resource_loaders_translate_read_errors(
    monkeypatch: pytest.MonkeyPatch,
    loader: Callable[[], object],
    resource_name: str,
    error_type: type[OSError],
) -> None:
    class UnreadableResource:
        def joinpath(self, name: str) -> UnreadableResource:
            assert name == resource_name
            return self

        def read_bytes(self) -> bytes:
            raise error_type("simulated unreadable package data")

    cards_module._clear_resource_caches()
    monkeypatch.setattr(cards_module, "files", lambda _package: UnreadableResource())
    try:
        with pytest.raises(CardValidationError, match=resource_name) as caught:
            loader()
        assert isinstance(caught.value.__cause__, error_type)
    finally:
        cards_module._clear_resource_caches()


def test_operation_card_round_trip_is_immutable_and_projection_is_detached() -> None:
    source = _operation()
    validated = validate_card(source)
    source["id"] = "changed"
    assert validated["id"] == "maintenance.core@0.5.0"
    with pytest.raises(TypeError):
        validated["id"] = "changed"  # type: ignore[index]
    plain = card_to_json(validated)
    plain["id"] = "detached"
    assert validated["id"] == "maintenance.core@0.5.0"


def test_load_card_json_translates_utf8_encoding_failures() -> None:
    with pytest.raises(CardValidationError, match="UTF-8"):
        load_card_json(chr(0xD800))

    assert load_card_json(json.dumps(_operation()))["id"] == "maintenance.core@0.5.0"


def test_load_card_json_translates_missing_path_read_failure(tmp_path: Path) -> None:
    source = tmp_path / "missing-card.json"

    with pytest.raises(CardValidationError, match=str(source)) as caught:
        load_card_json(source)

    assert isinstance(caught.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize("error_type", (OSError, PermissionError))
def test_load_card_json_translates_path_read_os_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error_type: type[OSError],
) -> None:
    source = tmp_path / "unreadable-card.json"
    failure = error_type("simulated card read failure")

    def fail_read(_source: Path) -> bytes:
        raise failure

    monkeypatch.setattr(Path, "read_bytes", fail_read)

    with pytest.raises(CardValidationError, match=str(source)) as caught:
        load_card_json(source)

    assert caught.value.__cause__ is failure


def test_load_card_json_does_not_translate_programmer_read_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "card.json"
    failure = TypeError("simulated programmer error")

    def fail_read(_source: Path) -> bytes:
        raise failure

    monkeypatch.setattr(Path, "read_bytes", fail_read)

    with pytest.raises(TypeError, match="simulated programmer error") as caught:
        load_card_json(source)

    assert caught.value is failure


def test_load_card_json_path_and_direct_input_controls_are_unchanged(
    tmp_path: Path,
) -> None:
    raw = json.dumps(_operation()).encode()
    source = tmp_path / "card.json"
    source.write_bytes(raw)

    assert load_card_json(source)["id"] == "maintenance.core@0.5.0"
    assert load_card_json(raw)["id"] == "maintenance.core@0.5.0"
    assert load_card_json(raw.decode())["id"] == "maintenance.core@0.5.0"

    with pytest.raises(CardValidationError, match="Card v1 JSON is not UTF-8"):
        load_card_json(b"\xff")
    with pytest.raises(CardValidationError, match="invalid Card v1 JSON"):
        load_card_json(b"{")


def test_load_card_json_preserves_whitespace_padded_v1_compatibility() -> None:
    raw = json.dumps(_operation()).encode("utf-8")
    padded = raw + b" " * (8_388_609 - len(raw))

    assert load_card_json(padded)["schema"] == "operatebench.operation_card.v1"


def test_duplicate_json_keys_and_unsafe_scalars_are_refused() -> None:
    raw = json.dumps(_operation()).replace(
        '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
    )
    with pytest.raises(CardValidationError, match="duplicate"):
        load_card_json(raw)
    for bad in (True, 1.5, 2**63, "\ud800"):
        card = _operation()
        card["content"]["policy_contract"]["approval_threshold_minor"] = bad  # type: ignore[index]
        with pytest.raises(CardValidationError):
            validate_card(card)


def test_unknown_keys_bad_digest_unsorted_sets_and_ordered_arrays() -> None:
    card = _operation()
    card["content"]["authority_contract"]["roles"][0]["extra"] = 1  # type: ignore[index]
    with pytest.raises(CardValidationError):
        validate_card(card)
    card = _operation()
    card["content_digest_sha256"] = "f" * 64
    with pytest.raises(CardValidationError, match="digest"):
        validate_card(card)
    card = _operation()
    card["content"]["transition_contract"]["terminal_kinds"] = ["z", "a"]  # type: ignore[index]
    card["content_digest_sha256"] = recompute_card_digest(card)
    with pytest.raises(CardValidationError, match="sorted"):
        validate_card(card)


def test_validator_enforces_every_all_of_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = load_schema("operatebench.operation_card.v1")
    schema["allOf"] = [{"required": ["required_by_all_of"]}]
    monkeypatch.setattr(cards_module, "load_schema", lambda _name: schema)
    with pytest.raises(CardValidationError, match="required_by_all_of"):
        validate_card(_operation())


def test_exact_timestamp_calendar_and_reference_resolution() -> None:
    operation = validate_card(_operation())
    op_ref = {k: operation[k] for k in ("schema", "id", "content_digest_sha256")}
    scenario = {
        "schema": "operatebench.semantic_scenario_card.v1",
        "schema_version": 1,
        "id": "maintenance.scenario",
        "content": {
            "operation_core": op_ref,
            "construct_label": "lifecycle",
            "common_estimand_id": "common.y.v1",
            "decision_points": [
                {
                    "point_id": "p1",
                    "order": 0,
                    "reach_predicate": {
                        "kind": "scalar",
                        "left": {"kind": "pointer", "pointer": "/status"},
                        "operator": "EQ",
                        "right": {"kind": "literal", "value": "open"},
                    },
                    "required_prior_point_ids": [],
                    "common_obligation_id": "o1",
                    "admissible_effect_classes": ["close"],
                }
            ],
            "causal_graph": {
                "nodes": [{"node_id": "p1", "semantic_type": "decision"}],
                "edges": [],
            },
            "authored_hazards": [],
            "common_outcome_obligations": [
                {
                    "obligation_id": "o1",
                    "predicate_id": "closed",
                    "terminal_required": True,
                }
            ],
            "allowed_variant_dimensions": [
                {
                    "dimension_id": "surface",
                    "value_type": "STRING",
                    "changes_semantics": False,
                }
            ],
        },
        "content_digest_sha256": "0" * 64,
    }
    scenario["content_digest_sha256"] = recompute_card_digest(scenario)
    valid_scenario = validate_card(scenario, references=[operation])
    bad = deepcopy(scenario)
    bad["content"]["operation_core"]["content_digest_sha256"] = "f" * 64
    bad["content_digest_sha256"] = recompute_card_digest(bad)
    with pytest.raises(CardValidationError, match="reference"):
        validate_card(bad, references=[operation])
    assert valid_scenario["id"] == "maintenance.scenario"


def test_scenario_decision_obligations_resolve_exactly() -> None:
    operation = validate_card(_operation())
    scenario = _scenario(operation)
    scenario["content"]["decision_points"][0]["common_obligation_id"] = "absent"
    scenario["content_digest_sha256"] = recompute_card_digest(scenario)
    with pytest.raises(CardValidationError, match="obligation"):
        validate_card(scenario, references=[operation])


def _scenario_with_causal_chain(
    operation: object, node_count: int, *, back_edge: bool = False
) -> dict[str, object]:
    scenario = _scenario(operation)
    content = cast(dict[str, Any], scenario["content"])
    point_template = content["decision_points"][0]
    points = []
    nodes = []
    edges = []
    for index in range(node_count):
        point_id = f"p{index:04d}"
        point = deepcopy(point_template)
        point.update(
            point_id=point_id,
            order=index,
            required_prior_point_ids=[] if index == 0 else [f"p{index - 1:04d}"],
        )
        points.append(point)
        nodes.append({"node_id": point_id, "semantic_type": "decision"})
        if index:
            edges.append(
                {
                    "cause_node_id": f"p{index - 1:04d}",
                    "effect_node_id": point_id,
                    "relation": "PRECEDES",
                }
            )
    if back_edge:
        edges.append(
            {
                "cause_node_id": f"p{node_count - 1:04d}",
                "effect_node_id": "p0000",
                "relation": "PRECEDES",
            }
        )
    content["decision_points"] = points
    content["causal_graph"] = {"nodes": nodes, "edges": edges}
    scenario["content_digest_sha256"] = recompute_card_digest(scenario)
    return scenario


def test_scenario_causal_chain_above_python_recursion_limit_validates() -> None:
    operation = validate_card(_operation())
    scenario = _scenario_with_causal_chain(operation, 2000)
    original = deepcopy(scenario)

    assert validate_card(scenario, references=[operation])
    assert scenario == original


def test_scenario_large_causal_cycle_is_domain_rejected() -> None:
    operation = validate_card(_operation())
    scenario = _scenario_with_causal_chain(operation, 2000, back_edge=True)

    with pytest.raises(CardValidationError, match="causal graph contains a cycle"):
        validate_card(scenario, references=[operation])


def test_scenario_causal_self_loop_is_rejected() -> None:
    operation = validate_card(_operation())
    scenario = _scenario(operation)
    scenario["content"]["causal_graph"]["edges"] = [
        {"cause_node_id": "p1", "effect_node_id": "p1", "relation": "PRECEDES"}
    ]
    scenario["content_digest_sha256"] = recompute_card_digest(scenario)

    with pytest.raises(CardValidationError, match="causal graph contains a cycle"):
        validate_card(scenario, references=[operation])


def test_scenario_disconnected_causal_components_validate() -> None:
    operation = validate_card(_operation())
    scenario = _scenario_with_causal_chain(operation, 3)
    scenario["content"]["causal_graph"]["edges"] = scenario["content"]["causal_graph"][
        "edges"
    ][:1]
    scenario["content_digest_sha256"] = recompute_card_digest(scenario)

    assert validate_card(scenario, references=[operation])


def test_scenario_missing_causal_endpoint_precedes_topology_validation() -> None:
    operation = validate_card(_operation())
    scenario = _scenario(operation)
    scenario["content"]["causal_graph"]["edges"] = [
        {
            "cause_node_id": "missing",
            "effect_node_id": "missing",
            "relation": "PRECEDES",
        },
        {"cause_node_id": "p1", "effect_node_id": "p1", "relation": "PRECEDES"},
    ]
    scenario["content"]["causal_graph"]["edges"].sort(
        key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
    )
    scenario["content_digest_sha256"] = recompute_card_digest(scenario)

    with pytest.raises(CardValidationError, match="causal edge is unresolved"):
        validate_card(scenario, references=[operation])


@pytest.mark.parametrize("duplicate_kind", ("node_id", "edge"))
def test_scenario_duplicate_causal_declarations_remain_rejected(
    duplicate_kind: str,
) -> None:
    operation = validate_card(_operation())
    scenario = _scenario(operation)
    graph = scenario["content"]["causal_graph"]
    if duplicate_kind == "node_id":
        graph["nodes"].append({"node_id": "p1", "semantic_type": "hazard"})
        graph["nodes"].sort(
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
        )
        expected = "causal node identities collide"
    else:
        edge = {
            "cause_node_id": "p1",
            "effect_node_id": "p1",
            "relation": "PRECEDES",
        }
        graph["edges"] = [edge, deepcopy(edge)]
        expected = "duplicate set items"
    scenario["content_digest_sha256"] = recompute_card_digest(scenario)

    with pytest.raises(CardValidationError, match=expected):
        validate_card(scenario, references=[operation])


def test_scenario_decision_point_and_hazard_ids_are_disjoint() -> None:
    operation = validate_card(_operation())
    scenario = _scenario(operation)
    scenario["content"]["authored_hazards"] = [
        {
            "hazard_id": "p1",
            "predicate": deepcopy(
                scenario["content"]["decision_points"][0]["reach_predicate"]
            ),
            "expected_common_finding_code": "shared.identity",
        }
    ]
    scenario["content_digest_sha256"] = recompute_card_digest(scenario)

    with pytest.raises(
        CardValidationError, match="scenario hazard and decision point IDs collide"
    ):
        validate_card(scenario, references=[operation])


def test_variant_payload_values_are_validated_against_operation_contract() -> None:
    operation = validate_card(_operation())
    scenario = validate_card(_scenario(operation), references=[operation])
    variant = _variant(scenario)
    valid = validate_card(variant, references=[operation, scenario])
    assert valid["id"] == "maintenance.variant"

    invalid = deepcopy(variant)
    invalid["content"]["events"][0]["payload"]["summary"] = 42
    invalid["content_digest_sha256"] = recompute_card_digest(invalid)
    with pytest.raises(CardValidationError, match="payload"):
        validate_card(invalid, references=[operation, scenario])


@pytest.mark.parametrize(
    "dangerous_pattern",
    [
        "(a+)+$",
        "^(a|aa)+$",
        "^a*a*$",
        "^a*aa*$",
        "^a*(?:)a*(?:)a*(?:)a*$",
        "^" + "(?:a|aa)" * 28 + "$",
        "^(?:a)$",
        "^a|b$",
        r"^(a)\1$",
        "^(?=a)a$",
        "(?i)^a$",
        "^(?(1)a|b)$",
        "^(?>a)$",
        "[a-z",
        "[z-a]",
        "[a--z]",
        "^*a$",
        "^a**$",
        "^a+?$",
        "^a{2}{3}$",
        "a^$",
        "^$a",
        "a\\",
        "a" * 257,
    ],
)
def test_operation_refuses_unsafe_field_patterns_before_matching(
    dangerous_pattern: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    operation = _operation()
    operation["content"]["event_contracts"][0]["required"]["summary"]["shape"][
        "pattern"
    ] = dangerous_pattern
    operation["content_digest_sha256"] = recompute_card_digest(operation)
    original_fullmatch = cards_module.re.fullmatch

    def guarded_fullmatch(pattern: str, value: str) -> object:
        assert pattern != dangerous_pattern, "unsafe card pattern reached re.fullmatch"
        return original_fullmatch(pattern, value)

    monkeypatch.setattr(cards_module.re, "fullmatch", guarded_fullmatch)
    with pytest.raises(CardValidationError, match=r"safe regex|pattern|allowed shape"):
        validate_card(operation)


@pytest.mark.parametrize("location", ["root", "properties", "items"])
@pytest.mark.parametrize(
    "dangerous_pattern",
    [
        "(a+)+$",
        "^a*(?:)a*(?:)a*(?:)a*$",
        "^" + "(?:a|aa)" * 28 + "$",
        "^(?:a)$",
        "^a|b$",
    ],
)
def test_operation_refuses_unsafe_initial_state_patterns_recursively(
    location: str, dangerous_pattern: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    operation = _operation()
    schema = operation["content"]["state_contract"]["initial_state_schema"]
    if location == "root":
        operation["content"]["state_contract"]["initial_state_schema"] = {
            "type": "string",
            "minLength": 0,
            "maxLength": 255,
            "pattern": dangerous_pattern,
            "enum": None,
        }
    elif location == "properties":
        schema["properties"]["status"]["pattern"] = dangerous_pattern
    else:
        schema["properties"]["status"] = {
            "type": "array",
            "items": {
                "type": "string",
                "minLength": 0,
                "maxLength": 255,
                "pattern": dangerous_pattern,
                "enum": None,
            },
            "minItems": 0,
            "maxItems": 1,
            "uniqueItems": False,
        }
        schema["required"] = ["status"]
    operation["content_digest_sha256"] = recompute_card_digest(operation)
    original_fullmatch = cards_module.re.fullmatch

    def guarded_fullmatch(pattern: str, value: str) -> object:
        assert pattern != dangerous_pattern, "unsafe card pattern reached re.fullmatch"
        return original_fullmatch(pattern, value)

    monkeypatch.setattr(cards_module.re, "fullmatch", guarded_fullmatch)
    with pytest.raises(CardValidationError, match=r"safe regex|pattern|allowed shape"):
        validate_card(operation)


def test_packaged_schema_enforces_the_authored_safe_regex_grammar() -> None:
    normative_path = Path(__file__).parents[1] / "docs/schemas/card-defs-v1.schema.json"
    packaged_path = (
        Path(cards_module.__file__).parent / "resources/cards/card-defs-v1.schema.json"
    )
    normative_bytes = normative_path.read_bytes()
    assert packaged_path.read_bytes() == normative_bytes
    definitions = json.loads(normative_bytes)["$defs"]
    pattern_schemas = (
        definitions["fieldShape"]["oneOf"][0]["properties"]["pattern"],
        definitions["jsonSchemaSubset"]["oneOf"][3]["properties"]["pattern"],
    )

    for pattern_schema in pattern_schemas:
        cards_module._validate_schema(None, pattern_schema, "$.pattern")
        cards_module._validate_schema("~" * 256, pattern_schema, "$.pattern")
        for invalid in (
            "é" * 128,
            "a" * 257,
            "a\n",
            "\ud800",
            "a|b",
            "(a)",
            ".",
            "a*a*",
            "a**",
            "[a-z",
            "[]",
            "[z-a]",
            "a\\",
            r"\q",
            r"\1",
            "a{4097}",
            "a{2,1}",
            "a{1,4097}",
            "a{,2}",
        ):
            with pytest.raises(CardValidationError, match=r"allowed shape|lexical|long"):
                cards_module._validate_schema(invalid, pattern_schema, "$.pattern")


@pytest.mark.parametrize("authored_path", ("field_shape", "initial_state_schema"))
@pytest.mark.parametrize(
    "invalid_pattern",
    (
        "a|b",
        "(a)",
        ".",
        "a*a*",
        "a**",
        "[a-z",
        "[]",
        "[^]",
        "[z-a]",
        "[a--z]",
        "a\\",
        r"\q",
        r"\1",
        "a{4097}",
        "a{2,1}",
        "a{1,4097}",
        "a{,2}",
        "a{2}b{3}",
        "a*b+",
    ),
)
def test_schema_rejects_runtime_invalid_patterns_before_safe_parser(
    authored_path: str,
    invalid_pattern: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation: dict[str, Any] = _operation()
    if authored_path == "field_shape":
        operation["content"]["event_contracts"][0]["required"]["summary"]["shape"][
            "pattern"
        ] = invalid_pattern
    else:
        operation["content"]["state_contract"]["initial_state_schema"]["properties"][
            "status"
        ]["pattern"] = invalid_pattern
    operation["content_digest_sha256"] = recompute_card_digest(operation)

    def safe_parser_must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("schema-invalid authored pattern reached the safe parser")

    monkeypatch.setattr(cards_module, "_validate_safe_pattern", safe_parser_must_not_run)
    with pytest.raises(CardValidationError, match="allowed shape"):
        validate_card(operation)


@pytest.mark.parametrize("authored_path", ("field_shape", "initial_state_schema"))
def test_multibyte_authored_patterns_fail_in_packaged_schema_before_safe_parser(
    authored_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    operation: dict[str, Any] = _operation()
    if authored_path == "field_shape":
        operation["content"]["event_contracts"][0]["required"]["summary"]["shape"][
            "pattern"
        ] = "é" * 128
    else:
        operation["content"]["state_contract"]["initial_state_schema"]["properties"][
            "status"
        ]["pattern"] = "é" * 128
    operation["content_digest_sha256"] = recompute_card_digest(operation)

    def safe_parser_must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("schema-invalid authored pattern reached the safe parser")

    monkeypatch.setattr(cards_module, "_validate_safe_pattern", safe_parser_must_not_run)
    with pytest.raises(CardValidationError, match="allowed shape"):
        validate_card(operation)


def test_direct_safe_pattern_gate_enforces_printable_ascii_and_exact_boundary() -> None:
    cards_module._validate_safe_pattern("~" * 256, "$.pattern")

    for invalid in ("é", "\ud800", "\x1f", "\x7f"):
        with pytest.raises(CardValidationError, match=r"safe regex.*printable ASCII"):
            cards_module._validate_safe_pattern(invalid, "$.pattern")
    with pytest.raises(CardValidationError, match=r"safe regex.*256 characters"):
        cards_module._validate_safe_pattern("a" * 257, "$.pattern")


def test_safe_card_patterns_remain_valid_and_regex_inputs_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation_card = _operation()
    shape = operation_card["content"]["event_contracts"][0]["required"]["summary"][
        "shape"
    ]
    shape["pattern"] = r"^[a-z]+-[A-Z\d]$"
    shape["max_length"] = 5000
    operation_card["content_digest_sha256"] = recompute_card_digest(operation_card)
    operation = validate_card(operation_card)
    scenario = validate_card(_scenario(operation), references=[operation])

    accepted = _variant(scenario)
    accepted["content"]["events"][0]["payload"]["summary"] = "leak-A"
    accepted["content_digest_sha256"] = recompute_card_digest(accepted)
    assert validate_card(accepted, references=[operation, scenario])

    oversized = _variant(scenario)
    oversized["content"]["events"][0]["payload"]["summary"] = "a" * 4097
    oversized["content_digest_sha256"] = recompute_card_digest(oversized)
    original_fullmatch = cards_module.re.fullmatch

    def guarded_fullmatch(pattern: str, value: str) -> object:
        if pattern == shape["pattern"]:
            raise AssertionError("oversized card input reached re.fullmatch")
        return original_fullmatch(pattern, value)

    monkeypatch.setattr(cards_module.re, "fullmatch", guarded_fullmatch)
    with pytest.raises(CardValidationError, match=r"regex input|too long"):
        validate_card(oversized, references=[operation, scenario])


@pytest.mark.parametrize(
    ("pattern", "value"),
    [
        ("", ""),
        ("^$", ""),
        (r"^[a-z]+$", "leak"),
        (r"^\(a\+\)\{2\}\|\.$", "(a+){2}|."),
        (r"^[(){}+*?.|\\]+$", "(){}+*?.|\\"),
        (r"^[A-Z\d]{1,8}$", "A12"),
        (r"^a{2}ba?$", "aaba"),
    ],
)
def test_safe_literal_class_range_and_bounded_patterns_are_accepted(
    pattern: str, value: str
) -> None:
    operation_card = _operation()
    shape = operation_card["content"]["event_contracts"][0]["required"]["summary"][
        "shape"
    ]
    shape["pattern"] = pattern
    operation_card["content_digest_sha256"] = recompute_card_digest(operation_card)
    operation = validate_card(operation_card)
    scenario = validate_card(_scenario(operation), references=[operation])
    variant = _variant(scenario)
    variant["content"]["events"][0]["payload"]["summary"] = value
    variant["content_digest_sha256"] = recompute_card_digest(variant)
    assert validate_card(variant, references=[operation, scenario])


def test_nullable_authored_state_enum_means_no_restriction() -> None:
    operation_card = _operation()
    status_schema = operation_card["content"]["state_contract"]["initial_state_schema"][
        "properties"
    ]["status"]
    status_schema["enum"] = None
    operation_card["content_digest_sha256"] = recompute_card_digest(operation_card)
    operation = validate_card(operation_card)
    scenario = validate_card(_scenario(operation), references=[operation])
    variant = _variant(scenario)
    variant["content"]["hidden_initial_values"]["status"] = "pending"
    variant["content_digest_sha256"] = recompute_card_digest(variant)
    assert validate_card(variant, references=[operation, scenario])


def test_authored_state_enum_enforces_value_and_generic_enums_use_exact_types() -> None:
    operation = validate_card(_operation())
    scenario = validate_card(_scenario(operation), references=[operation])

    matching = _variant(scenario)
    assert validate_card(matching, references=[operation, scenario])

    nonmatching = _variant(scenario)
    nonmatching["content"]["hidden_initial_values"]["status"] = "pending"
    nonmatching["content_digest_sha256"] = recompute_card_digest(nonmatching)
    with pytest.raises(CardValidationError, match=r"hidden_initial_values|enum"):
        validate_card(nonmatching, references=[operation, scenario])

    with pytest.raises(CardValidationError, match="enum"):
        cards_module._validate_schema(True, {"enum": [1]}, "$")
    with pytest.raises(CardValidationError, match="enum"):
        cards_module._validate_schema(1, {"enum": [True]}, "$")


def test_schema_validation_can_apply_unique_items_without_card_set_ordering() -> None:
    schema = {"type": "array", "uniqueItems": True, "items": {"type": "string"}}

    cards_module._validate_schema(["z", "a"], schema, "$", enforce_canonical_sets=False)
    with pytest.raises(CardValidationError, match="duplicate"):
        cards_module._validate_schema(
            ["a", "a"], schema, "$", enforce_canonical_sets=False
        )
    with pytest.raises(CardValidationError, match="sorted"):
        cards_module._validate_schema(["z", "a"], schema, "$")


def test_census_is_b1_complete_and_artifact9_dependency_fails_closed() -> None:
    census = load_field_census()
    validated = validate_field_census(census)
    assert validated["artifact_schema"] == "operatebench.b1_card_contract.v1"
    assert all(entry["cardinality"] == "exact" for entry in validated["entries"])
    with pytest.raises(CardValidationError, match="Artifact 9"):
        validate_field_census(census, artifact_schema="operatebench.artifact.v9")
    duplicate = card_to_json(validated)
    duplicate["entries"].append(deepcopy(duplicate["entries"][0]))
    with pytest.raises(CardValidationError, match=r"multiple owners|duplicate"):
        validate_field_census(duplicate)


def test_census_frozen_output_revalidates_idempotently_three_times() -> None:
    validated = validate_field_census(load_field_census())
    projection = card_to_json(validated)
    digest = hashlib.sha256(
        cards_module.canonical_json_bytes(projection, "field census projection")
    ).hexdigest()

    for _ in range(3):
        validated = validate_field_census(validated)
        assert card_to_json(validated) == projection
        assert (
            hashlib.sha256(
                cards_module.canonical_json_bytes(
                    card_to_json(validated), "field census projection"
                )
            ).hexdigest()
            == digest
        )


def test_census_revalidated_output_remains_immutable_and_detached() -> None:
    source = load_field_census()
    validated = validate_field_census(validate_field_census(source))

    with pytest.raises(TypeError):
        validated["artifact_schema"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        validated["entries"][0]["source_paths"][0] = "changed"  # type: ignore[index]

    source["entries"][0]["source_paths"][0] = "changed"
    detached = card_to_json(validated)
    detached["entries"][0]["source_paths"][0] = "also-changed"
    assert validated["entries"][0]["source_paths"][0] not in {
        "changed",
        "also-changed",
    }


def test_forged_frozen_census_is_still_fully_validated() -> None:
    malformed = load_field_census()
    malformed["entries"][0]["source_paths"] = ("../escape",)
    malformed_frozen = MappingProxyType(
        {**malformed, "entries": tuple(malformed["entries"])}
    )
    with pytest.raises(CardValidationError, match="source path"):
        validate_field_census(malformed_frozen)

    unknown = load_field_census()
    unknown_frozen = MappingProxyType(
        {**unknown, "entries": tuple(unknown["entries"]), "fabricated": True}
    )
    with pytest.raises(CardValidationError):
        validate_field_census(unknown_frozen)


def test_census_consumers_match_normative_owner_oracle_exactly() -> None:
    census = load_field_census()
    assert validate_field_census(census)

    for wrong_consumers in (["fabricated.consumer"], ["identity_manifest_input"]):
        changed = deepcopy(census)
        changed["entries"][0]["consumers"] = wrong_consumers
        with pytest.raises(CardValidationError, match="consumer"):
            validate_field_census(changed)

    absent = deepcopy(census)
    absent["entries"][0]["consumers"] = []
    with pytest.raises(CardValidationError, match="consumer"):
        validate_field_census(absent)


def test_census_rejects_zero_owner_missing_owner_stale_source_and_patterns() -> None:
    census = load_field_census()
    zero_owner = deepcopy(census)
    zero_owner["entries"][0]["owner"] = ""
    with pytest.raises(CardValidationError, match="owner"):
        validate_field_census(zero_owner)

    missing_owner = deepcopy(census)
    removed_owner = missing_owner["entries"][-1]["owner"]
    missing_owner["entries"] = [
        entry for entry in missing_owner["entries"] if entry["owner"] != removed_owner
    ]
    with pytest.raises(CardValidationError, match="missing owner"):
        validate_field_census(missing_owner)

    absent_consumer = deepcopy(census)
    absent_consumer["entries"][0]["consumers"] = []
    with pytest.raises(CardValidationError, match="consumer"):
        validate_field_census(absent_consumer)

    wrong_owner = deepcopy(census)
    wrong_owner["entries"][-1]["owner"] = "operation_core_content"
    with pytest.raises(CardValidationError, match="owner"):
        validate_field_census(wrong_owner)

    stale_source = deepcopy(census)
    stale_source["entries"][0]["source_paths"] = ["src/operatebench/absent.py"]
    with pytest.raises(CardValidationError, match="stale"):
        validate_field_census(stale_source, source_root=Path(__file__).parents[1])

    unexpanded = deepcopy(census)
    unexpanded["entries"][0]["field_path_pattern"] = "/card_schema/*"
    with pytest.raises(CardValidationError, match="unexpanded"):
        validate_field_census(unexpanded)


def test_operation_validation_does_not_advance_a_raising_reference_generator() -> None:
    advances = 0

    def raising_references() -> Any:
        nonlocal advances
        advances += 1
        raise AssertionError("operation validation advanced references")
        yield  # pragma: no cover

    assert validate_card(_operation(), references=raising_references())
    assert advances == 0


def test_operation_validation_does_not_request_a_reference_iterator() -> None:
    class SentinelReferences:
        def __iter__(self) -> Any:
            raise AssertionError("operation validation requested an iterator")

    assert validate_card(_operation(), references=SentinelReferences())


def test_operation_validation_ignores_malformed_unrelated_reference() -> None:
    stale_unrelated = _operation()
    stale_unrelated["content"]["operation_version"] = "9.9.9"

    assert validate_card(_operation(), references=[stale_unrelated])


def test_malformed_operation_is_rejected_before_reference_access() -> None:
    class SentinelReferences:
        def __iter__(self) -> Any:
            raise AssertionError("malformed operation accessed references")

    operation = _operation()
    operation["content"].pop("operation_type")
    operation["content_digest_sha256"] = recompute_card_digest(operation)

    with pytest.raises(CardValidationError, match=r"operation_type|required"):
        validate_card(operation, references=SentinelReferences())


@pytest.mark.parametrize("card_kind", ["scenario", "variant"])
def test_dependent_card_validation_materializes_references_exactly_once(
    card_kind: str,
) -> None:
    operation = validate_card(_operation())
    scenario = validate_card(_scenario(operation), references=[operation])
    candidate = _scenario(operation) if card_kind == "scenario" else _variant(scenario)
    targets = [operation] if card_kind == "scenario" else [operation, scenario]
    iterations = 0

    class CountingReferences:
        def __iter__(self) -> Any:
            nonlocal iterations
            iterations += 1
            return (target for target in targets)

    assert validate_card(candidate, references=CountingReferences())
    assert iterations == 1


@pytest.mark.parametrize("card_kind", ["scenario", "variant"])
def test_invalid_dependent_card_is_rejected_before_reference_access(
    card_kind: str,
) -> None:
    class SentinelReferences:
        def __iter__(self) -> Any:
            raise AssertionError("invalid dependent card accessed references")

    operation = validate_card(_operation())
    scenario = validate_card(_scenario(operation), references=[operation])
    candidate = _scenario(operation) if card_kind == "scenario" else _variant(scenario)
    candidate["content_digest_sha256"] = "0" * 64

    with pytest.raises(CardValidationError, match="digest"):
        validate_card(candidate, references=SentinelReferences())


@pytest.mark.parametrize("card_kind", ["scenario", "variant"])
@pytest.mark.parametrize("failure", ["stale", "forged", "missing", "ambiguous"])
def test_dependent_cards_fail_closed_for_untrusted_reference_sets(
    card_kind: str, failure: str
) -> None:
    operation = validate_card(_operation())
    scenario = validate_card(_scenario(operation), references=[operation])
    candidate = _scenario(operation) if card_kind == "scenario" else _variant(scenario)
    target = operation if card_kind == "scenario" else scenario
    references = [operation] if card_kind == "scenario" else [operation, scenario]

    if failure == "missing":
        references.remove(target)
    elif failure == "ambiguous":
        references.append(target)
    else:
        replacement = card_to_json(target)
        field = "construct_label" if card_kind == "variant" else "operation_type"
        replacement["content"][field] = "forged.value"
        if failure == "forged":
            replacement["content_digest_sha256"] = recompute_card_digest(replacement)
        references[references.index(target)] = replacement

    with pytest.raises(CardValidationError, match=r"ambiguous|digest|reference|target"):
        validate_card(candidate, references=(item for item in references))


def test_untrusted_reference_targets_are_revalidated_and_generators_are_safe() -> None:
    operation = validate_card(_operation())
    scenario = _scenario(operation)
    forged = card_to_json(operation)
    forged["content"]["operation_type"] = "attacker.controlled"
    with pytest.raises(CardValidationError, match=r"target|digest"):
        validate_card(scenario, references=[forged])

    stale = card_to_json(operation)
    stale["content"]["operation_version"] = "9.9.9"
    with pytest.raises(CardValidationError, match=r"target|digest"):
        validate_card(scenario, references=[stale])

    validated_scenario = validate_card(scenario, references=[operation])
    variant = _variant(validated_scenario)
    variant["content"]["hidden_initial_values"] = {"status": "INVALID"}
    variant["content_digest_sha256"] = recompute_card_digest(variant)
    with pytest.raises(CardValidationError, match=r"hidden_initial_values|lexical|enum"):
        validate_card(
            variant, references=(item for item in [operation, validated_scenario])
        )


def test_variant_resolves_operation_through_its_scenario_reference() -> None:
    operation = validate_card(_operation())
    scenario = validate_card(_scenario(operation), references=[operation])
    unrelated = _operation()
    unrelated["id"] = "unrelated.core@0.5.0"
    unrelated["content"]["state_contract"]["initial_state_schema"]["properties"][
        "status"
    ]["enum"] = ["different"]
    unrelated["content_digest_sha256"] = recompute_card_digest(unrelated)
    unrelated = validate_card(unrelated)
    assert validate_card(_variant(scenario), references=[unrelated, scenario, operation])

    with pytest.raises(CardValidationError, match=r"operation|reference"):
        validate_card(_variant(scenario), references=[unrelated, scenario])


def test_scenario_predicate_pointers_resolve_to_operation_projection() -> None:
    operation = validate_card(_operation())
    scenario = _scenario(operation)
    scenario["content"]["decision_points"][0]["reach_predicate"]["left"]["pointer"] = (
        "/unknown"
    )
    scenario["content_digest_sha256"] = recompute_card_digest(scenario)
    with pytest.raises(CardValidationError, match="pointer"):
        validate_card(scenario, references=[operation])


def test_variant_triggers_resolve_contract_and_correlation_shapes() -> None:
    operation = validate_card(_operation())
    scenario = validate_card(_scenario(operation), references=[operation])
    variant = _variant(scenario)
    event = variant["content"]["events"][0]
    event["at"] = None
    event["trigger"] = {
        "kind": "action",
        "type_name": "totally.unknown",
        "correlation": {},
    }
    variant["content_digest_sha256"] = recompute_card_digest(variant)
    with pytest.raises(CardValidationError, match="trigger"):
        validate_card(variant, references=[scenario, operation])

    event["trigger"] = {
        "kind": "event",
        "type_name": "issue.reported",
        "correlation": {"summary": 42},
    }
    variant["content_digest_sha256"] = recompute_card_digest(variant)
    with pytest.raises(CardValidationError, match="correlation"):
        validate_card(variant, references=[scenario, operation])

    event["trigger"]["correlation"] = {"undeclared": "value"}
    variant["content_digest_sha256"] = recompute_card_digest(variant)
    with pytest.raises(CardValidationError, match="correlation"):
        validate_card(variant, references=[scenario, operation])


@pytest.mark.parametrize(
    ("malformation", "expected_path"),
    (
        ("timestamp", r"\$/content/events/1/at"),
        ("payload", r"\$/content/events/1/payload/summary"),
        ("trigger_correlation", r"\$/content/events/1/trigger/correlation/summary"),
    ),
)
def test_variant_second_event_errors_include_exact_index(
    malformation: str, expected_path: str
) -> None:
    operation = validate_card(_operation())
    scenario = validate_card(_scenario(operation), references=[operation])
    variant: Any = _variant(scenario)
    second = deepcopy(variant["content"]["events"][0])
    second["event_id"] = "event.two"
    second["authored_sequence"] = 1
    second["at"] = "2026-08-21T00:01:00.000000Z"
    variant["content"]["events"].append(second)

    if malformation == "timestamp":
        second["at"] = "2026-02-31T00:01:00.000000Z"
    elif malformation == "payload":
        second["payload"]["summary"] = 42
    else:
        second["at"] = None
        second["trigger"] = {
            "kind": "event",
            "type_name": "issue.reported",
            "correlation": {"summary": 42},
        }

    variant["content_digest_sha256"] = recompute_card_digest(variant)
    with pytest.raises(CardValidationError, match=expected_path):
        validate_card(variant, references=[scenario, operation])


def _variant_with_events(
    scenario: object, events: list[dict[str, Any]]
) -> dict[str, object]:
    variant = _variant(scenario)
    content = cast(dict[str, Any], variant["content"])
    content["events"] = events
    variant["content_digest_sha256"] = recompute_card_digest(variant)
    return variant


def test_second_absolute_event_is_reported_as_canonical_order_offender() -> None:
    operation = validate_card(_operation())
    scenario = validate_card(_scenario(operation), references=[operation])
    variant = _variant(scenario)
    earlier = deepcopy(cast(dict[str, Any], variant["content"])["events"][0])
    later = deepcopy(earlier)
    later.update(
        event_id="event.later",
        authored_sequence=1,
        at="2026-08-21T00:01:00.000000Z",
    )
    earlier.update(event_id="event.earlier", authored_sequence=0)

    candidate = _variant_with_events(scenario, [later, earlier])

    with pytest.raises(CardValidationError, match=r"\$/content/events/1 absolute"):
        validate_card(candidate, references=[scenario, operation])


def test_third_absolute_event_is_reported_when_it_creates_first_inversion() -> None:
    operation = validate_card(_operation())
    scenario = validate_card(_scenario(operation), references=[operation])
    variant = _variant(scenario)
    template = deepcopy(cast(dict[str, Any], variant["content"])["events"][0])
    first = deepcopy(template)
    first.update(event_id="event.first", authored_sequence=0)
    second = deepcopy(template)
    second.update(
        event_id="event.third",
        authored_sequence=2,
        at="2026-08-21T00:02:00.000000Z",
    )
    third = deepcopy(template)
    third.update(
        event_id="event.second",
        authored_sequence=1,
        at="2026-08-21T00:01:00.000000Z",
    )

    candidate = _variant_with_events(scenario, [first, second, third])

    with pytest.raises(CardValidationError, match=r"\$/content/events/2 absolute"):
        validate_card(candidate, references=[scenario, operation])


def test_interleaved_conditional_event_preserves_absolute_offender_index() -> None:
    operation = validate_card(_operation())
    scenario = validate_card(_scenario(operation), references=[operation])
    variant = _variant(scenario)
    earlier = deepcopy(cast(dict[str, Any], variant["content"])["events"][0])
    later = deepcopy(earlier)
    later.update(
        event_id="event.later",
        authored_sequence=1,
        at="2026-08-21T00:01:00.000000Z",
    )
    conditional = deepcopy(earlier)
    conditional.update(
        event_id="event.conditional",
        authored_sequence=2,
        at=None,
        trigger={
            "kind": "event",
            "type_name": "issue.reported",
            "correlation": {"summary": "leak"},
        },
    )
    earlier.update(event_id="event.earlier", authored_sequence=0)

    candidate = _variant_with_events(scenario, [later, conditional, earlier])

    with pytest.raises(CardValidationError, match=r"\$/content/events/2 absolute"):
        validate_card(candidate, references=[scenario, operation])


def test_absolute_events_in_canonical_order_are_accepted() -> None:
    operation = validate_card(_operation())
    scenario = validate_card(_scenario(operation), references=[operation])
    variant = _variant(scenario)
    first = deepcopy(cast(dict[str, Any], variant["content"])["events"][0])
    second = deepcopy(first)
    second.update(
        event_id="event.second",
        authored_sequence=1,
        at="2026-08-21T00:01:00.000000Z",
    )

    candidate = _variant_with_events(scenario, [first, second])

    assert validate_card(candidate, references=[scenario, operation])


def test_absolute_timestamp_ties_use_sequence_before_event_id_for_offender() -> None:
    operation = validate_card(_operation())
    scenario = validate_card(_scenario(operation), references=[operation])
    variant = _variant(scenario)
    earlier = deepcopy(cast(dict[str, Any], variant["content"])["events"][0])
    later = deepcopy(earlier)
    later.update(event_id="event.a", authored_sequence=2)
    earlier.update(event_id="event.z", authored_sequence=1)

    candidate = _variant_with_events(scenario, [later, earlier])

    with pytest.raises(CardValidationError, match=r"\$/content/events/1 absolute"):
        validate_card(candidate, references=[scenario, operation])


def test_conditional_events_with_distinct_owners_allow_both_permutations() -> None:
    operation = validate_card(_operation())
    scenario = validate_card(_scenario(operation), references=[operation])
    variant = _variant(scenario)
    action_owned = variant["content"]["events"][0]
    action_owned["at"] = None
    action_owned["trigger"] = {
        "kind": "action",
        "type_name": "case.close",
        "correlation": {},
    }
    action_owned["event_id"] = "event.action"
    action_owned["authored_sequence"] = 1
    event_owned = deepcopy(action_owned)
    event_owned["event_id"] = "event.event"
    event_owned["authored_sequence"] = 2
    event_owned["trigger"] = {
        "kind": "event",
        "type_name": "issue.reported",
        "correlation": {"summary": "leak"},
    }

    for events in ([action_owned, event_owned], [event_owned, action_owned]):
        candidate = deepcopy(variant)
        candidate["content"]["events"] = deepcopy(events)
        candidate["content_digest_sha256"] = recompute_card_digest(candidate)
        assert validate_card(candidate, references=[scenario, operation])


def test_conditional_events_are_canonically_ordered_within_causal_owner() -> None:
    operation = validate_card(_operation())
    scenario = validate_card(_scenario(operation), references=[operation])
    variant = _variant(scenario)
    first = variant["content"]["events"][0]
    first["at"] = None
    first["trigger"] = {
        "kind": "action",
        "type_name": "case.close",
        "correlation": {},
    }
    first["event_id"] = "event.two"
    first["authored_sequence"] = 2
    second = deepcopy(first)
    second["event_id"] = "event.one"
    second["authored_sequence"] = 1
    variant["content"]["events"].append(second)
    variant["content_digest_sha256"] = recompute_card_digest(variant)
    with pytest.raises(CardValidationError, match=r"conditional.*order"):
        validate_card(variant, references=[scenario, operation])


def _variant_rejection_context() -> tuple[
    Mapping[str, Any], Mapping[str, Any], dict[str, Any]
]:
    operation = validate_card(_operation())
    scenario = validate_card(_scenario(operation), references=[operation])
    return operation, scenario, cast(dict[str, Any], _variant(scenario))


@pytest.mark.parametrize("second_code", ["declined", "superseded"])
def test_variant_rejects_duplicate_expected_rejection_event_id(
    second_code: str,
) -> None:
    operation, scenario, variant = _variant_rejection_context()
    variant["content"]["expected_event_rejections"] = [
        {"event_id": "event.one", "code": "declined"},
        {"event_id": "event.one", "code": second_code},
    ]
    variant["content_digest_sha256"] = recompute_card_digest(variant)

    with pytest.raises(
        CardValidationError,
        match=(
            r"^\$/content/expected_event_rejections/1/event_id duplicates event_id "
            r"'event\.one' first declared at "
            r"\$/content/expected_event_rejections/0/event_id$"
        ),
    ):
        validate_card(variant, references=[scenario, operation])


def test_variant_rejects_non_adjacent_rejection_duplicate_at_later_index() -> None:
    operation, scenario, variant = _variant_rejection_context()
    second_event = deepcopy(variant["content"]["events"][0])
    second_event.update(
        event_id="event.two",
        authored_sequence=1,
        at="2026-08-21T00:01:00.000000Z",
    )
    variant["content"]["events"].append(second_event)
    variant["content"]["expected_event_rejections"] = [
        {"event_id": "event.one", "code": "declined"},
        {"event_id": "event.two", "code": "deferred"},
        {"event_id": "event.one", "code": "superseded"},
    ]
    variant["content_digest_sha256"] = recompute_card_digest(variant)

    with pytest.raises(
        CardValidationError,
        match=r"^\$/content/expected_event_rejections/2/event_id duplicates event_id",
    ):
        validate_card(variant, references=[scenario, operation])


def test_variant_accepts_distinct_expected_rejection_ids_in_authored_order() -> None:
    operation, scenario, variant = _variant_rejection_context()
    second_event = deepcopy(variant["content"]["events"][0])
    second_event.update(
        event_id="event.two",
        authored_sequence=1,
        at="2026-08-21T00:01:00.000000Z",
    )
    variant["content"]["events"].append(second_event)
    authored = [
        {"event_id": "event.two", "code": "deferred"},
        {"event_id": "event.one", "code": "declined"},
    ]
    variant["content"]["expected_event_rejections"] = authored
    variant["content_digest_sha256"] = recompute_card_digest(variant)
    original = deepcopy(variant)

    validated = validate_card(variant, references=[scenario, operation])

    assert card_to_json(validated)["content"]["expected_event_rejections"] == authored
    assert card_to_json(validated)["content_digest_sha256"] == recompute_card_digest(
        validated
    )
    assert variant == original


def test_variant_unresolved_expected_rejection_keeps_exact_index_and_precedence() -> None:
    operation, scenario, variant = _variant_rejection_context()
    variant["content"]["expected_event_rejections"] = [
        {"event_id": "event.one", "code": "declined"},
        {"event_id": "event.missing", "code": "missing"},
        {"event_id": "event.one", "code": "superseded"},
    ]
    variant["content_digest_sha256"] = recompute_card_digest(variant)

    with pytest.raises(
        CardValidationError,
        match=(
            r"^\$/content/expected_event_rejections/1/event_id "
            r"'event\.missing' is unresolved$"
        ),
    ):
        validate_card(variant, references=[scenario, operation])


@pytest.mark.parametrize("identifier", ["actor_id", "fixture_id"])
def test_variant_lookup_identifiers_are_unique(identifier: str) -> None:
    operation_card = _operation()
    operation_card["content"]["authority_contract"]["roles"].append(
        {"role_id": "other", "authorities": ["report.issue"]}
    )
    operation_card["content"]["authority_contract"]["roles"].sort(
        key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
    )
    operation_card["content_digest_sha256"] = recompute_card_digest(operation_card)
    operation = validate_card(operation_card)
    scenario = validate_card(_scenario(operation), references=[operation])
    variant = _variant(scenario)
    if identifier == "actor_id":
        variant["content"]["actors"].append(
            {"actor_id": "tenant.one", "role_id": "other"}
        )
        variant["content"]["actors"].sort(
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
        )
    else:
        variant["content"]["message_fixtures"].append(
            {"fixture_id": "message.one", "surface_content_digest_sha256": "b" * 64}
        )
        variant["content"]["message_fixtures"].sort(
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
        )
    variant["content_digest_sha256"] = recompute_card_digest(variant)
    with pytest.raises(CardValidationError, match=identifier.replace("_", " ")):
        validate_card(variant, references=[scenario, operation])


def test_variant_rejects_unknown_role_for_unused_actor_with_actor_context() -> None:
    operation = validate_card(_operation())
    scenario = validate_card(_scenario(operation), references=[operation])
    variant = _variant(scenario)
    variant["content"]["actors"].append(
        {"actor_id": "unused.actor", "role_id": "unknown.role"}
    )
    variant["content"]["actors"].sort(
        key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
    )
    invalid_index = next(
        index
        for index, actor in enumerate(variant["content"]["actors"])
        if actor["actor_id"] == "unused.actor"
    )
    variant["content_digest_sha256"] = recompute_card_digest(variant)

    with pytest.raises(
        CardValidationError,
        match=rf"\$/content/actors/{invalid_index}/role_id.*operation-declared",
    ):
        validate_card(variant, references=[scenario, operation])


def test_variant_accepts_unused_actor_with_operation_declared_role() -> None:
    operation = validate_card(_operation())
    scenario = validate_card(_scenario(operation), references=[operation])
    variant = _variant(scenario)
    variant["content"]["actors"].append({"actor_id": "unused.actor", "role_id": "tenant"})
    variant["content"]["actors"].sort(
        key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
    )
    variant["content_digest_sha256"] = recompute_card_digest(variant)

    assert validate_card(variant, references=[scenario, operation])


def test_variant_used_actor_still_requires_event_authority() -> None:
    operation_card = _operation()
    operation_card["content"]["authority_contract"]["roles"].append(
        {"role_id": "observer", "authorities": ["case.observe"]}
    )
    operation_card["content"]["authority_contract"]["roles"].sort(
        key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
    )
    operation_card["content_digest_sha256"] = recompute_card_digest(operation_card)
    operation = validate_card(operation_card)
    scenario = validate_card(_scenario(operation), references=[operation])
    variant = _variant(scenario)
    variant["content"]["actors"][0]["role_id"] = "observer"
    variant["content_digest_sha256"] = recompute_card_digest(variant)

    with pytest.raises(
        CardValidationError, match=r"\$/content/events/0/actor_id authority mismatch"
    ):
        validate_card(variant, references=[scenario, operation])


def test_variant_rejects_one_invalid_role_among_multiple_unused_actors() -> None:
    operation = validate_card(_operation())
    scenario = validate_card(_scenario(operation), references=[operation])
    variant = _variant(scenario)
    variant["content"]["actors"].extend(
        [
            {"actor_id": "unused.invalid", "role_id": "unknown.role"},
            {"actor_id": "unused.valid", "role_id": "tenant"},
        ]
    )
    variant["content"]["actors"].sort(
        key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
    )
    invalid_index = next(
        index
        for index, actor in enumerate(variant["content"]["actors"])
        if actor["actor_id"] == "unused.invalid"
    )
    variant["content_digest_sha256"] = recompute_card_digest(variant)

    with pytest.raises(
        CardValidationError,
        match=rf"\$/content/actors/{invalid_index}/role_id.*unknown\.role",
    ):
        validate_card(variant, references=[scenario, operation])


def test_public_validators_translate_malformed_in_memory_values() -> None:
    cyclic = _operation()
    cycle: list[object] = []
    cycle.append(cycle)
    cyclic["content"]["policy_contract"]["currency"] = cycle
    with pytest.raises(CardValidationError):
        validate_card(cyclic)

    census = load_field_census()
    census_cycle: list[object] = []
    census_cycle.append(census_cycle)
    census["entries"] = census_cycle
    with pytest.raises(CardValidationError):
        validate_field_census(census)

    for malformed_census in (
        {**load_field_census(), "entries": 1},
        {**load_field_census(), "schema_version": True},
        {**load_field_census(), "artifact_schema": "\ud800"},
    ):
        with pytest.raises(CardValidationError):
            validate_field_census(malformed_census)

    malformed_values = [
        {"schema": True},
        {"schema": "operatebench.operation_card.v1", "content": 1},
        {"schema": "\ud800"},
    ]
    for malformed in malformed_values:
        with pytest.raises(CardValidationError):
            validate_card(malformed)


@pytest.mark.parametrize(
    "malformed_content",
    (
        float("nan"),
        float("inf"),
        {"\ud800": "value"},
        {"key": "\ud800"},
        {1: "value"},
    ),
)
def test_recompute_card_digest_closes_malformed_json_value_errors(
    malformed_content: object,
) -> None:
    card = _operation()
    card["content"] = malformed_content

    with pytest.raises(CardValidationError):
        recompute_card_digest(card)


def test_recompute_card_digest_translates_cycles_with_preserved_cause() -> None:
    card = _operation()
    cycle: list[object] = []
    cycle.append(cycle)
    card["content"] = cycle

    with pytest.raises(CardValidationError) as caught:
        recompute_card_digest(card)

    assert isinstance(caught.value.__cause__, cards_module.JsonSafetyError)


@pytest.mark.parametrize(
    "failure",
    (
        cards_module.JsonSafetyError("simulated safety refusal"),
        UnicodeError("simulated text refusal"),
        RecursionError("simulated recursion refusal"),
    ),
)
def test_recompute_card_digest_translates_declared_encoder_malformed_families(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    card = _operation()

    def refuse_encoding(_value: object, _context: str) -> bytes:
        raise failure

    monkeypatch.setattr(cards_module, "canonical_json_bytes", refuse_encoding)

    with pytest.raises(CardValidationError, match="digest wrapper") as caught:
        recompute_card_digest(card)

    assert caught.value.__cause__ is failure


def test_recompute_card_digest_reraises_existing_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = _operation()
    failure = CardValidationError("already classified")

    def refuse_encoding(_value: object, _context: str) -> bytes:
        raise failure

    monkeypatch.setattr(cards_module, "canonical_json_bytes", refuse_encoding)

    with pytest.raises(CardValidationError) as caught:
        recompute_card_digest(card)

    assert caught.value is failure


def test_recompute_card_digest_keeps_programmer_encoder_type_errors_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = _operation()
    failure = TypeError("simulated programmer error")

    def fail_encoding(_value: object, _context: str) -> bytes:
        raise failure

    monkeypatch.setattr(cards_module, "canonical_json_bytes", fail_encoding)

    with pytest.raises(TypeError, match="simulated programmer error") as caught:
        recompute_card_digest(card)

    assert caught.value is failure


@pytest.mark.parametrize(
    "malformed",
    (
        bytearray(b"bytes"),
        memoryview(b"bytes"),
        b"bytes",
        {"set-item"},
        frozenset({"set-item"}),
    ),
)
def test_non_json_python_types_are_rejected_at_every_in_memory_boundary(
    malformed: object,
) -> None:
    card = _operation()
    card["content"] = malformed
    census = load_field_census()
    census["entries"] = malformed

    for probe in (
        lambda: card_to_json(malformed),
        lambda: recompute_card_digest(card),
        lambda: validate_card(card),
        lambda: validate_field_census(census),
    ):
        with pytest.raises(CardValidationError) as caught:
            probe()
        assert isinstance(caught.value.__cause__, cards_module.JsonSafetyError)


def test_hostile_numeric_subclasses_are_rejected_before_their_protocols_run() -> None:
    class HostileInt(int):
        def __lt__(self, _other: object) -> bool:
            raise TypeError("hostile integer comparison ran")

        def __le__(self, _other: object) -> bool:
            raise TypeError("hostile integer comparison ran")

        def __eq__(self, _other: object) -> bool:
            raise TypeError("hostile integer comparison ran")

        def __add__(self, _other: object) -> int:
            raise TypeError("hostile integer arithmetic ran")

    class HostileFloat(float):
        def __float__(self) -> float:
            raise TypeError("hostile float conversion ran")

        def __lt__(self, _other: object) -> bool:
            raise TypeError("hostile float comparison ran")

        def __eq__(self, _other: object) -> bool:
            raise TypeError("hostile float comparison ran")

    int_content_card = _operation()
    int_content_card["content"] = HostileInt(1)
    int_version_card = _operation()
    int_version_card["schema_version"] = HostileInt(1)
    float_content_card = _operation()
    float_content_card["content"] = HostileFloat(1.0)

    for probe in (
        lambda: recompute_card_digest(int_content_card),
        lambda: validate_card(int_version_card),
        lambda: card_to_json(HostileFloat(1.0)),
        lambda: recompute_card_digest(float_content_card),
    ):
        with pytest.raises(CardValidationError) as caught:
            probe()
        assert isinstance(caught.value.__cause__, cards_module.JsonSafetyError)


def test_hostile_string_subclass_is_rejected_before_text_protocols_run() -> None:
    class HostileStr(str):
        def __str__(self) -> str:
            raise TypeError("hostile string conversion ran")

        def encode(self, *_args: object, **_kwargs: object) -> bytes:
            raise TypeError("hostile string encoding ran")

    hostile = HostileStr("hostile")
    card = _operation()
    card["content"] = hostile
    census = load_field_census()
    census["entries"] = hostile

    for probe in (
        lambda: card_to_json(hostile),
        lambda: recompute_card_digest(card),
        lambda: validate_card(card),
        lambda: validate_field_census(census),
    ):
        with pytest.raises(CardValidationError) as caught:
            probe()
        assert isinstance(caught.value.__cause__, cards_module.JsonSafetyError)


def test_hostile_string_subclass_key_is_rejected_before_hash_or_equality() -> None:
    class HostileKey(str):
        def __hash__(self) -> int:
            raise TypeError("hostile key hash ran")

        def __eq__(self, _other: object) -> bool:
            raise TypeError("hostile key equality ran")

    class HostileKeyMapping(Mapping[str, object]):
        def __getitem__(self, _key: str) -> object:
            raise AssertionError("mapping lookup must not run")

        def __iter__(self) -> Iterator[str]:
            return iter(())

        def __len__(self) -> int:
            return 1

        def items(self) -> Any:
            return ((HostileKey("hostile"), "value"),)

    with pytest.raises(CardValidationError) as caught:
        card_to_json(HostileKeyMapping())

    assert isinstance(caught.value.__cause__, cards_module.JsonSafetyError)


@pytest.mark.parametrize(
    "subclass_value",
    (
        type("BenignStr", (str,), {})("value"),
        type("BenignInt", (int,), {})(1),
        type("BenignFloat", (float,), {})(1.0),
        IntEnum("BenignIntEnum", {"ONE": 1}).ONE,
    ),
)
def test_benign_json_primitive_subclasses_are_rejected(
    subclass_value: object,
) -> None:
    with pytest.raises(CardValidationError) as caught:
        card_to_json(subclass_value)

    assert isinstance(caught.value.__cause__, cards_module.JsonSafetyError)


def test_card_to_json_keeps_internal_json_safety_type_errors_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = TypeError("internal ensure_json_safe programmer error")

    def fail_safety(_value: object, _context: str) -> object:
        raise failure

    monkeypatch.setattr(cards_module, "ensure_json_safe", fail_safety)

    with pytest.raises(TypeError) as caught:
        card_to_json({"valid": "projection"})

    assert caught.value is failure


def test_validate_card_keeps_internal_projection_type_errors_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = _operation()
    failure = TypeError("internal card_to_json programmer error")

    def fail_projection(_value: object) -> object:
        raise failure

    monkeypatch.setattr(cards_module, "card_to_json", fail_projection)

    with pytest.raises(TypeError) as caught:
        validate_card(card)

    assert caught.value is failure


def test_validate_field_census_keeps_internal_validator_type_errors_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = TypeError("internal census validator programmer error")

    def fail_validation(*_args: object, **_kwargs: object) -> Mapping[str, Any]:
        raise failure

    monkeypatch.setattr(cards_module, "_validate_field_census", fail_validation)

    with pytest.raises(TypeError) as caught:
        validate_field_census(load_field_census())

    assert caught.value is failure


def test_card_census_public_callable_export_audit_is_complete() -> None:
    card_callables = {
        name for name in cards_module.__all__ if callable(getattr(cards_module, name))
    }
    assert card_callables == {
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
    }
    assert {name for name in operatebench.__all__ if name in card_callables} == {
        "CardValidationError",
        "load_card_json",
        "load_field_census",
        "recompute_card_digest",
        "validate_card",
        "validate_field_census",
    }


@pytest.mark.parametrize(
    "probe",
    (
        lambda: load_schema([]),
        lambda: load_card_json(None),
        lambda: validate_card(None),
        lambda: validate_field_census(None),
        lambda: recompute_card_digest(None),
        lambda: card_to_json(item for item in ()),
    ),
)
def test_card_census_public_boundaries_close_wrong_root_types(
    probe: Callable[[], object],
) -> None:
    with pytest.raises(CardValidationError):
        probe()


def test_public_text_and_byte_sources_require_exact_builtin_types() -> None:
    class HostileStr(str):
        def __str__(self) -> str:
            raise OSError("hostile string conversion ran")

        def encode(self, *_args: object, **_kwargs: object) -> bytes:
            raise OSError("hostile string encoding ran")

        def __hash__(self) -> int:
            raise OSError("hostile string hash ran")

        def __eq__(self, _other: object) -> bool:
            raise OSError("hostile string equality ran")

    class HostileBytes(bytes):
        def decode(self, *_args: object, **_kwargs: object) -> str:
            raise OSError("hostile bytes decoding ran")

    for probe in (
        lambda: load_schema(HostileStr("operatebench.operation_card.v1")),
        lambda: load_card_json(HostileStr("{}")),
        lambda: load_card_json(HostileBytes(b"{}")),
        lambda: validate_field_census(
            load_field_census(), artifact_schema=HostileStr("unused")
        ),
    ):
        with pytest.raises(CardValidationError) as caught:
            probe()
        assert type(caught.value.__cause__) is TypeError

    assert load_schema("operatebench.operation_card.v1")["type"] == "object"
    assert load_card_json(json.dumps(_operation()))["schema"] == (
        "operatebench.operation_card.v1"
    )
    assert load_card_json(json.dumps(_operation()).encode())["schema"] == (
        "operatebench.operation_card.v1"
    )


def test_public_path_sources_require_exact_platform_path_type(tmp_path: Path) -> None:
    class HostilePath(type(Path())):
        def read_bytes(self) -> bytes:
            raise AssertionError("path subclass read ran")

        def resolve(self, *args: object, **kwargs: object) -> Path:
            raise AssertionError("path subclass resolve ran")

    hostile = HostilePath(tmp_path / "card.json")
    for probe in (
        lambda: load_card_json(hostile),
        lambda: validate_field_census(load_field_census(), source_root=hostile),
    ):
        with pytest.raises(CardValidationError) as caught:
            probe()
        assert type(caught.value.__cause__) is TypeError


def test_hostile_census_mapping_key_is_rejected_before_key_protocols_run() -> None:
    class HostileKey(str):
        def __str__(self) -> str:
            raise OSError("hostile key conversion ran")

        def __repr__(self) -> str:
            raise OSError("hostile key representation ran")

        def __hash__(self) -> int:
            raise OSError("hostile key hash ran")

        def __eq__(self, _other: object) -> bool:
            raise OSError("hostile key equality ran")

    class HostileCensus(Mapping[str, object]):
        def __getitem__(self, _key: str) -> object:
            raise AssertionError("mapping lookup ran")

        def __iter__(self) -> Iterator[str]:
            return iter(())

        def __len__(self) -> int:
            return 1

        def items(self) -> Any:
            return ((HostileKey("entries"), []),)

    with pytest.raises(CardValidationError) as caught:
        validate_field_census(HostileCensus())

    assert isinstance(caught.value.__cause__, cards_module.JsonSafetyError)


def test_raw_json_safety_errors_are_never_formatted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HostileJsonSafetyError(cards_module.JsonSafetyError):
        @property
        def args(self) -> tuple[object, ...]:  # type: ignore[override]
            raise AssertionError("hostile exception args read")

        def __str__(self) -> str:
            raise AssertionError("hostile exception formatting ran")

        def __repr__(self) -> str:
            raise AssertionError("hostile exception representation ran")

    failure = HostileJsonSafetyError()

    def fail_depth_check(_text: str, _context: str) -> None:
        raise failure

    monkeypatch.setattr(cards_module, "ensure_raw_json_depth", fail_depth_check)

    with pytest.raises(CardValidationError) as caught:
        load_card_json(b"{}")

    assert str(caught.value) == "invalid Card v1 JSON"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_packaged_resource_read_errors_are_never_formatted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HostileOSError(OSError):
        @property
        def args(self) -> tuple[object, ...]:  # type: ignore[override]
            raise AssertionError("hostile exception args read")

        def __str__(self) -> str:
            raise AssertionError("hostile exception formatting ran")

        def __repr__(self) -> str:
            raise AssertionError("hostile exception representation ran")

    failure = HostileOSError()

    class UnreadableResource:
        def joinpath(self, _name: str) -> UnreadableResource:
            return self

        def read_bytes(self) -> bytes:
            raise failure

    cards_module._clear_resource_caches()
    monkeypatch.setattr(cards_module, "files", lambda _package: UnreadableResource())
    try:
        with pytest.raises(CardValidationError) as caught:
            load_card_registry()
    finally:
        cards_module._clear_resource_caches()

    assert str(caught.value) == ("cannot read packaged resource 'card-registry-v1.json'")
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize(
    "failure",
    (
        type(
            "HostileJsonSafetyError",
            (cards_module.JsonSafetyError,),
            {
                "args": property(
                    lambda _self: (_ for _ in ()).throw(
                        AssertionError("hostile exception args read")
                    )
                ),
                "__str__": lambda _self: (_ for _ in ()).throw(
                    AssertionError("hostile exception formatting ran")
                ),
                "__repr__": lambda _self: (_ for _ in ()).throw(
                    AssertionError("hostile exception representation ran")
                ),
            },
        )(),
        type(
            "HostileUnicodeError",
            (UnicodeError,),
            {
                "args": property(
                    lambda _self: (_ for _ in ()).throw(
                        AssertionError("hostile exception args read")
                    )
                ),
                "__str__": lambda _self: (_ for _ in ()).throw(
                    AssertionError("hostile exception formatting ran")
                ),
                "__repr__": lambda _self: (_ for _ in ()).throw(
                    AssertionError("hostile exception representation ran")
                ),
            },
        )(),
        type(
            "HostileRecursionError",
            (RecursionError,),
            {
                "args": property(
                    lambda _self: (_ for _ in ()).throw(
                        AssertionError("hostile exception args read")
                    )
                ),
                "__str__": lambda _self: (_ for _ in ()).throw(
                    AssertionError("hostile exception formatting ran")
                ),
                "__repr__": lambda _self: (_ for _ in ()).throw(
                    AssertionError("hostile exception representation ran")
                ),
            },
        )(),
    ),
)
def test_digest_encoder_errors_are_never_formatted(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    card = _operation()

    def fail_encoding(_value: object, _context: str) -> bytes:
        raise failure

    monkeypatch.setattr(cards_module, "canonical_json_bytes", fail_encoding)

    with pytest.raises(CardValidationError) as caught:
        recompute_card_digest(card)

    assert str(caught.value) == "malformed in-memory card digest wrapper"
    assert caught.value.__cause__ is failure


def test_regex_errors_from_card_values_are_never_formatted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HostileRegexError(re.error):
        @property
        def args(self) -> tuple[object, ...]:  # type: ignore[override]
            raise AssertionError("hostile exception args read")

        def __str__(self) -> str:
            raise AssertionError("hostile exception formatting ran")

        def __repr__(self) -> str:
            raise AssertionError("hostile exception representation ran")

    compile_failure = HostileRegexError("hostile")
    monkeypatch.setattr(
        cards_module,
        "re",
        SimpleNamespace(
            error=re.error,
            compile=lambda _pattern: (_ for _ in ()).throw(compile_failure),
            fullmatch=re.fullmatch,
        ),
    )
    with pytest.raises(CardValidationError) as compile_caught:
        cards_module._validate_safe_pattern("literal", "$/pattern")
    assert compile_caught.value.__cause__ is compile_failure

    match_failure = HostileRegexError("hostile")
    monkeypatch.setattr(
        cards_module,
        "re",
        SimpleNamespace(
            error=re.error,
            compile=re.compile,
            fullmatch=lambda _pattern, _value: (_ for _ in ()).throw(match_failure),
        ),
    )
    with pytest.raises(CardValidationError) as match_caught:
        cards_module._pattern_fullmatch(
            "literal", "literal", "$/value", trusted_pattern=True
        )
    assert match_caught.value.__cause__ is match_failure


def test_error_wrapping_never_formats_hostile_protocol_exceptions() -> None:
    class HostileTypeError(TypeError):
        @property
        def args(self) -> tuple[object, ...]:  # type: ignore[override]
            raise OSError("hostile exception args read")

        def __str__(self) -> str:
            raise OSError("hostile exception formatting ran")

        def __repr__(self) -> str:
            raise OSError("hostile exception representation ran")

    failures = [HostileTypeError() for _ in range(3)]

    class BadMapping(dict[str, object]):
        def items(self) -> Any:
            raise failures[0]

    class BadSequence(list[object]):
        def __iter__(self) -> Iterator[object]:
            raise failures[1]

    def bad_references() -> Any:
        raise failures[2]
        yield  # pragma: no cover

    operation = validate_card(_operation())
    scenario = _scenario(operation)
    for probe, _failure in (
        (lambda: card_to_json(BadMapping()), failures[0]),
        (lambda: card_to_json(BadSequence()), failures[1]),
        (lambda: validate_card(scenario, references=bad_references()), failures[2]),
    ):
        with pytest.raises(CardValidationError) as caught:
            probe()
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None


def test_projection_errors_do_not_read_hostile_metaclass_names() -> None:
    class HostileMeta(type):
        def __getattribute__(self, name: str) -> object:
            if name == "__name__":
                raise OSError("hostile metaclass name read")
            return super().__getattribute__(name)

    class Unsupported(metaclass=HostileMeta):
        pass

    for probe in (
        lambda: card_to_json(Unsupported()),
        lambda: load_card_json(cast(Any, Unsupported())),
    ):
        with pytest.raises(CardValidationError) as caught:
            probe()
        assert isinstance(
            caught.value.__cause__, (cards_module.JsonSafetyError, TypeError)
        )


def test_exact_path_oserror_translation_does_not_format_hostile_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class HostileOSError(OSError):
        @property
        def args(self) -> tuple[object, ...]:  # type: ignore[override]
            raise OSError("hostile OSError args read")

        def __str__(self) -> str:
            raise OSError("hostile OSError formatting ran")

        def __repr__(self) -> str:
            raise OSError("hostile OSError representation ran")

    source = tmp_path / "card.json"
    failure = HostileOSError()

    def fail_read(_source: Path) -> bytes:
        raise failure

    monkeypatch.setattr(Path, "read_bytes", fail_read)
    with pytest.raises(
        CardValidationError, match="cannot read Card v1 JSON path"
    ) as caught:
        load_card_json(source)

    assert caught.value.__cause__ is failure


def test_exact_source_root_resolve_errors_preserve_boundary_taxonomy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class HostileOSError(OSError):
        @property
        def args(self) -> tuple[object, ...]:  # type: ignore[override]
            raise OSError("hostile resolve error args read")

        def __str__(self) -> str:
            raise OSError("hostile resolve error formatting ran")

        def __repr__(self) -> str:
            raise OSError("hostile resolve error representation ran")

    for failure, expected in (
        (HostileOSError(), CardValidationError),
        (TypeError("programmer resolve error"), TypeError),
    ):

        def fail_resolve(
            _source: Path, *, strict: bool = False, _failure: Exception = failure
        ) -> Path:
            raise _failure

        with monkeypatch.context() as patch_context:
            patch_context.setattr(Path, "resolve", fail_resolve)
            with pytest.raises(expected) as caught:
                validate_field_census(load_field_census(), source_root=tmp_path)

        if isinstance(failure, OSError):
            assert caught.value.__cause__ is failure
        else:
            assert caught.value is failure


@pytest.mark.parametrize(
    "failure",
    (
        AssertionError("malformed mapping assertion"),
        OSError("malformed mapping access"),
        TypeError("malformed mapping protocol"),
        UnicodeError("malformed mapping text"),
    ),
)
def test_in_memory_boundaries_translate_custom_mapping_protocol_failures(
    failure: Exception,
) -> None:
    class MalformedMapping(dict[str, object]):
        def items(self) -> Any:
            raise failure

    malformed = MalformedMapping(
        schema="unused",
        schema_version=1,
        id="unused",
    )
    malformed["content"] = malformed
    probes = (
        lambda: card_to_json(malformed),
        lambda: validate_card(malformed),
        lambda: validate_field_census(malformed),
        lambda: recompute_card_digest(malformed),
    )

    for probe in probes:
        with pytest.raises(CardValidationError) as caught:
            probe()
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "failure",
    (
        AssertionError("malformed sequence assertion"),
        OSError("malformed sequence access"),
        TypeError("malformed sequence protocol"),
        UnicodeError("malformed sequence text"),
    ),
)
def test_in_memory_boundaries_translate_custom_sequence_protocol_failures(
    failure: Exception,
) -> None:
    class MalformedSequence(list[object]):
        def __iter__(self) -> Iterator[object]:
            raise failure

    malformed = MalformedSequence()
    card = _operation()
    card["content"] = malformed
    census = load_field_census()
    census["entries"] = malformed

    for probe in (
        lambda: card_to_json(malformed),
        lambda: validate_card(card),
        lambda: validate_field_census(census),
        lambda: recompute_card_digest(card),
    ):
        with pytest.raises(CardValidationError) as caught:
            probe()
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "failure_type", (AssertionError, OSError, TypeError, UnicodeError)
)
def test_dependent_card_reference_generators_close_malformed_iteration_errors(
    failure_type: type[Exception],
) -> None:
    operation = validate_card(_operation())
    scenario = _scenario(operation)
    failure = failure_type("malformed reference iteration")

    def malformed_references() -> Any:
        raise failure
        yield  # pragma: no cover

    with pytest.raises(CardValidationError) as caught:
        validate_card(scenario, references=malformed_references())

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_card_census_boundaries_close_deep_recursive_values() -> None:
    deep: object = "leaf"
    for _ in range(MAX_JSON_DEPTH + 1):
        deep = [deep]
    card = _operation()
    card["content"] = deep
    census = load_field_census()
    census["entries"] = deep

    for probe in (
        lambda: card_to_json(deep),
        lambda: validate_card(card),
        lambda: validate_field_census(census),
        lambda: recompute_card_digest(card),
    ):
        with pytest.raises(CardValidationError) as caught:
            probe()
        assert isinstance(caught.value.__cause__, cards_module.JsonSafetyError)


def test_card_validation_errors_from_untrusted_protocols_are_detached() -> None:
    failure = CardValidationError("already classified")

    class ClassifiedMapping(dict[str, object]):
        def items(self) -> Any:
            raise failure

    with pytest.raises(CardValidationError) as caught:
        card_to_json(ClassifiedMapping())

    assert caught.value is not failure
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_valid_card_digest_bytes_remain_stable() -> None:
    card = _operation()

    assert recompute_card_digest(card) == (
        "81ceec044170b7a23f904504f0f208ccb5c6cd9cf8e45c874d5561330840c6bf"
    )


@pytest.mark.parametrize(
    "options",
    (
        {"artifact_schema": True},
        {"artifact_schema": 1},
        {"source_root": "not-a-path"},
        {"source_root": 1},
    ),
)
def test_validate_field_census_closes_malformed_optional_argument_types(
    options: dict[str, object],
) -> None:
    with pytest.raises(CardValidationError):
        validate_field_census(load_field_census(), **options)


@pytest.mark.parametrize(
    "malformed_source_paths",
    (
        "src",
        ("docs/schemas/card-defs-v1.schema.json",),
        [],
        ["src/operatebench/cards.py", "src/operatebench/cards.py"],
        [1],
        [True],
        ["src/operatebench/cards.py", None],
    ),
)
def test_census_source_paths_match_normative_array_shape(
    malformed_source_paths: object,
) -> None:
    census = load_field_census()
    census["entries"][0]["source_paths"] = malformed_source_paths

    with pytest.raises(CardValidationError):
        validate_field_census(census)


@pytest.mark.parametrize(
    ("field", "malformed"),
    (
        ("field_path_pattern", 1),
        ("field_path_pattern", "/bad~escape"),
        ("field_path_pattern", "/bad/*"),
        ("owner", True),
        ("owner", 1),
        ("owner", "fabricated_owner"),
        ("consumers", "identity_manifest_input"),
        ("consumers", ("package_loader", "card_validator")),
        ("consumers", []),
        ("consumers", ["identity_manifest_input", "identity_manifest_input"]),
        ("consumers", [1]),
        ("consumers", [True]),
        ("nullability", 1),
        ("nullability", "sometimes"),
        ("cardinality", True),
        ("cardinality", "pattern"),
    ),
)
def test_census_entries_match_normative_schema_shape(
    field: str, malformed: object
) -> None:
    census = load_field_census()
    census["entries"][0][field] = malformed

    with pytest.raises(CardValidationError):
        validate_field_census(census)


def test_census_schema_rejects_unknown_entry_and_top_level_keys() -> None:
    unknown_entry = load_field_census()
    unknown_entry["entries"][0]["fabricated"] = True
    with pytest.raises(CardValidationError):
        validate_field_census(unknown_entry)

    unknown_top_level = load_field_census()
    unknown_top_level["fabricated"] = True
    with pytest.raises(CardValidationError):
        validate_field_census(unknown_top_level)


def test_census_packaged_schema_patterns_use_trusted_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbid_untrusted_pattern(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("packaged census schema used the untrusted regex gate")

    monkeypatch.setattr(cards_module, "_validate_safe_pattern", forbid_untrusted_pattern)
    assert validate_field_census(load_field_census())


def test_census_source_paths_reject_drive_forms_and_all_control_characters() -> None:
    census = load_field_census()
    for unsafe in ("C:/absolute/file", "C:drive-relative", "C/relative/file"):
        changed = deepcopy(census)
        changed["entries"][0]["source_paths"] = [unsafe]
        with pytest.raises(CardValidationError, match="source path"):
            validate_field_census(changed)

    for code_point in (*range(0x20), 0x7F):
        changed = deepcopy(census)
        changed["entries"][0]["source_paths"] = [
            f"docs/schemas/prefix{chr(code_point)}suffix.json"
        ]
        with pytest.raises(CardValidationError, match="source path"):
            validate_field_census(changed)

    for neighboring_control in (
        "docs/schemas/prefix suffix.json",
        "docs/schemas/prefix~suffix.json",
    ):
        changed = deepcopy(census)
        changed["entries"][0]["source_paths"] = [neighboring_control]
        assert validate_field_census(changed)


def test_draft_2020_12_census_schema_rejects_del_in_source_paths() -> None:
    normative_path = (
        Path(__file__).parents[1] / "docs/schemas/identity-field-census-v1.schema.json"
    )
    packaged_path = (
        Path(cards_module.__file__).parent
        / "resources/cards/identity-field-census-v1.schema.json"
    )
    normative_bytes = normative_path.read_bytes()
    assert packaged_path.read_bytes() == normative_bytes
    schema = json.loads(normative_bytes)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    source_path_schema = schema["properties"]["entries"]["items"]["properties"][
        "source_paths"
    ]["items"]

    cards_module._validate_schema(
        "docs/schemas/prefix~suffix.json", source_path_schema, "$"
    )
    with pytest.raises(CardValidationError, match="lexical"):
        cards_module._validate_schema(
            "docs/schemas/prefix\x7fsuffix.json", source_path_schema, "$"
        )


def test_census_source_paths_are_portable_and_optionally_checked_under_trusted_root(
    tmp_path: Path,
) -> None:
    census = load_field_census()
    old_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        validate_field_census(census)
    finally:
        os.chdir(old_cwd)

    for unsafe in (
        "/etc/passwd",
        "../escape",
        "docs/../escape",
        "docs\\schemas\\card.json",
        "docs//schemas/card.json",
        "docs/./schemas/card.json",
        "docs/schemas/card.json\n",
    ):
        changed = deepcopy(census)
        changed["entries"][0]["source_paths"] = [unsafe]
        with pytest.raises(CardValidationError, match="source path"):
            validate_field_census(changed)

    with pytest.raises(CardValidationError, match="stale"):
        validate_field_census(census, source_root=tmp_path)
    assert validate_field_census(census, source_root=Path(__file__).parents[1])


def test_normative_census_is_fully_expanded_unique_and_byte_identical() -> None:
    census = load_field_census()
    paths = [entry["field_path_pattern"] for entry in census["entries"]]
    assert len(paths) == len(set(paths))
    assert len(paths) == 106
    assert all(entry["cardinality"] == "exact" for entry in census["entries"])
    assert all("*" not in path and "[]" not in path for path in paths)
    assert all(entry["consumers"] for entry in census["entries"])
    packaged = files("operatebench.resources.cards").joinpath(
        "identity-field-census-v1.registry.json"
    )
    normative = files("operatebench").joinpath(
        "../../docs/schemas/identity-field-census-v1.registry.json"
    )
    assert packaged.read_bytes() == normative.read_bytes()


def test_census_rejects_missing_or_fabricated_contract_leaves() -> None:
    census = load_field_census()
    missing = deepcopy(census)
    missing["entries"].pop()
    with pytest.raises(CardValidationError, match="contract leaves"):
        validate_field_census(missing)

    fabricated = deepcopy(census)
    extra = deepcopy(fabricated["entries"][-1])
    extra["field_path_pattern"] = "/variant_card/content/fabricated"
    fabricated["entries"].append(extra)
    with pytest.raises(CardValidationError, match="contract leaves"):
        validate_field_census(fabricated)
