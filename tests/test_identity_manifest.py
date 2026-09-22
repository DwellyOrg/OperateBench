# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import hashlib
import json
import math
import shutil
import subprocess
import time
import types
import typing
from collections import ChainMap, UserDict
from collections.abc import Mapping, Sequence
from copy import copy, deepcopy
from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType

import pytest

from operatebench.jsonsafe import MAX_JSON_DEPTH, canonical_json_bytes

COMPONENTS = (
    "card_schema_content",
    "build_provenance_content",
    "operation_core_content",
    "semantic_scenario_content",
    "variant_content",
    "runtime_contract_content",
    "scaffold_content",
    "arm_protocol_content",
    "provider_run_plan_content",
    "evaluator_bundle_content",
    "analysis_contract_content",
)
EDGES = {
    "card_schema_content": (),
    "build_provenance_content": (),
    "operation_core_content": ("card_schema_content",),
    "semantic_scenario_content": ("operation_core_content", "card_schema_content"),
    "variant_content": ("semantic_scenario_content", "card_schema_content"),
    "runtime_contract_content": ("operation_core_content", "card_schema_content"),
    "scaffold_content": ("runtime_contract_content",),
    "arm_protocol_content": (
        "runtime_contract_content",
        "semantic_scenario_content",
        "variant_content",
    ),
    "provider_run_plan_content": (
        "runtime_contract_content",
        "scaffold_content",
        "build_provenance_content",
    ),
    "evaluator_bundle_content": (
        "operation_core_content",
        "card_schema_content",
        "runtime_contract_content",
    ),
    "analysis_contract_content": ("evaluator_bundle_content",),
}
BUNDLES = {
    "runtime_bundle": (
        "operation_core_content",
        "semantic_scenario_content",
        "variant_content",
        "runtime_contract_content",
        "scaffold_content",
        "arm_protocol_content",
        "provider_run_plan_content",
        "build_provenance_content",
    ),
    "experiment_bundle": (
        "runtime_bundle",
        "evaluator_bundle_content",
        "analysis_contract_content",
    ),
}


def _identity_module():
    import operatebench.identity as identity

    return identity


def _assert_tracebacks_do_not_retain(
    exc: BaseException, hostile: object, *, max_local_size: int
) -> None:
    pending_exceptions = [exc]
    seen_exceptions: set[int] = set()
    seen_objects: set[int] = set()
    retained_bytes = 0

    def inspect(root: object) -> None:
        nonlocal retained_bytes
        pending_objects = [root]
        while pending_objects:
            value = pending_objects.pop()
            assert value is not hostile
            if id(value) in seen_objects:
                continue
            seen_objects.add(id(value))
            if type(value) in (bytes, bytearray):
                size = len(value)  # type: ignore[arg-type]
                assert size <= max_local_size
                retained_bytes += size
            elif type(value) is str:
                assert len(value) <= max_local_size
            elif type(value) is dict:
                pending_objects.extend(value.keys())
                pending_objects.extend(value.values())
            elif type(value) in (list, tuple, set, frozenset):
                pending_objects.extend(value)  # type: ignore[arg-type]
            elif type(value) is MappingProxyType:
                pending_objects.extend(value.keys())
                pending_objects.extend(value.values())

    while pending_exceptions:
        current = pending_exceptions.pop()
        if id(current) in seen_exceptions:
            continue
        seen_exceptions.add(id(current))
        traceback = current.__traceback__
        while traceback is not None:
            for value in traceback.tb_frame.f_locals.values():
                inspect(value)
            traceback = traceback.tb_next
        if current.__cause__ is not None:
            pending_exceptions.append(current.__cause__)
        if current.__context__ is not None:
            pending_exceptions.append(current.__context__)
    assert retained_bytes <= max_local_size


def _assert_resource_exception_graph_is_clean(
    exc: BaseException, *, forbidden_ids: set[int]
) -> None:
    """Inspect every exception frame and recursively census retained raw bytes."""
    exceptions = [exc]
    seen_exceptions: set[int] = set()
    seen_objects: set[int] = set()
    retained_bytes = 0

    def inspect(root: object, location: str) -> None:
        nonlocal retained_bytes
        pending = [root]
        while pending:
            value = pending.pop()
            assert id(value) not in forbidden_ids, location
            if id(value) in seen_objects:
                continue
            seen_objects.add(id(value))
            if type(value) in (bytes, bytearray):
                assert len(value) <= 64 * 1024, location
                retained_bytes += len(value)
            elif type(value) is dict:
                pending.extend(value.keys())
                pending.extend(value.values())
            elif type(value) in (list, tuple, set, frozenset):
                pending.extend(value)
            elif type(value) is MappingProxyType:
                pending.extend(value.keys())
                pending.extend(value.values())

    while exceptions:
        current = exceptions.pop()
        if id(current) in seen_exceptions:
            continue
        seen_exceptions.add(id(current))
        traceback = current.__traceback__
        while traceback is not None:
            frame_name = traceback.tb_frame.f_code.co_name
            for local_name, value in traceback.tb_frame.f_locals.items():
                inspect(value, f"{frame_name}.{local_name}")
            traceback = traceback.tb_next
        if current.__cause__ is not None:
            exceptions.append(current.__cause__)
        if current.__context__ is not None:
            exceptions.append(current.__context__)

    assert retained_bytes <= 64 * 1024


def _requirements(name: str, identities: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "component": dependency,
            "compatible": {
                "kind": "exact_digest",
                "content_digest_sha256": identities[dependency].content_digest_sha256,
            },
        }
        for dependency in EDGES[name]
    ]


def _all_identities():
    identity = _identity_module()
    result = {}
    for index, name in enumerate(COMPONENTS):
        result[name] = identity.recompute_component(
            name,
            {f"/fixture/{name}": {"index": index}},
            _requirements(name, result),
        )
    return result


def _bundle_members(name: str, identities: dict[str, object]) -> dict[str, object]:
    return {member: identities[member] for member in BUNDLES[name]}


def _bundle_closure(name: str, identities: dict[str, object]) -> dict[str, object]:
    if name == "runtime_bundle":
        return _bundle_members(name, identities)
    return {
        member: identities[member]
        for member in (*BUNDLES["runtime_bundle"], *BUNDLES["experiment_bundle"][1:])
    }


def test_public_contract_literals_and_exports_are_closed() -> None:
    identity = _identity_module()
    assert identity.IDENTITY_MANIFEST_SCHEMA == "operatebench.identity_manifest.v1"
    assert identity.COMPONENT_IDENTITY_SCHEMA == "operatebench.component_identity.v1"
    assert identity.COMPOSITE_IDENTITY_SCHEMA == "operatebench.composite_identity.v1"
    assert (
        identity.COMPATIBILITY_CONTRACT_SCHEMA == "operatebench.compatibility_contract.v1"
    )
    assert identity.MANIFEST_DIGEST_DOMAIN == "operatebench.digest.identity_manifest.v1"
    assert identity.MAX_COMPONENT_IDENTITY_BYTES == 336 * 1024
    assert identity.MAX_COMPONENT_IDENTITY_NODES == 5_600
    assert identity.MAX_IDENTITY_RESOURCE_BYTES == 4 * 1024 * 1024
    assert identity.MAX_IDENTITY_RESOURCE_READ_CALLS == 128
    assert set(identity.__all__) == {
        "IDENTITY_MANIFEST_SCHEMA",
        "IDENTITY_MANIFEST_PROJECTION_CENSUS_SCHEMA",
        "COMPONENT_IDENTITY_SCHEMA",
        "COMPOSITE_IDENTITY_SCHEMA",
        "COMPATIBILITY_CONTRACT_SCHEMA",
        "MANIFEST_DIGEST_DOMAIN",
        "OWNED_MUTATION_CONTRACT_VERSION",
        "IDENTITY_SCHEMA_VERSION",
        "IdentityValidationError",
        "OwnedMutation",
        "RequirementV1",
        "RequirementEvaluationV1",
        "ComponentIdentityV1",
        "CompositeIdentityV1",
        "ValidatedIdentityManifest",
        "load_identity_schema",
        "load_identity_manifest_projection_census_schema",
        "load_component_registry",
        "load_compatibility_registry",
        "load_identity_golden_vectors",
        "identity_to_json",
        "identity_diff",
        "evaluate_component_requirements",
        "recompute_component",
        "recompute_bundle",
        "recompute_manifest_digest",
        "validate_identity_manifest",
    }


def test_strict_container_annotations_match_the_runtime_contract() -> None:
    identity = _identity_module()
    inspected_names = (
        "RequirementV1",
        "ComponentIdentityV1",
        "recompute_component",
        "recompute_bundle",
        "identity_to_json",
        "recompute_manifest_digest",
        "_cached_resource",
        "_exact_keys",
        "_validate_component_registry",
        "_validate_compatibility_registry",
        "_normalize_compatible",
        "_normalize_requirements",
        "_validate_component_identity",
        "_validate_projectable_composite",
        "_compute_bundle",
        "_compute_experiment_bundle",
        "_snapshot_resolved_member_identities",
    )
    hints = {
        name: typing.get_type_hints(getattr(identity, name)) for name in inspected_names
    }

    assert typing.get_origin(hints["recompute_component"]["owned_projection"]) is dict
    requires = hints["recompute_component"]["requires"]
    assert typing.get_origin(requires) is list
    assert typing.get_origin(typing.get_args(requires)[0]) is dict
    assert hints["recompute_manifest_digest"]["manifest"] == dict[str, typing.Any]

    resolver_types = typing.get_args(
        hints["recompute_bundle"]["resolved_member_identities"]
    )
    assert {typing.get_origin(candidate) for candidate in resolver_types} == {
        dict,
        types.MappingProxyType,
    }
    projected_resolver = hints["identity_to_json"]["resolved_member_identities"]
    projected_resolver_types = {
        candidate
        for candidate in typing.get_args(projected_resolver)
        if candidate is not type(None)
    }
    assert projected_resolver_types == set(resolver_types)
    for record_name, field_name in (
        ("RequirementV1", "compatible"),
        ("ComponentIdentityV1", "owned_projection"),
    ):
        assert {
            typing.get_origin(candidate)
            for candidate in typing.get_args(hints[record_name][field_name])
        } == {dict, types.MappingProxyType}

    def contains_broad_container(annotation: object) -> bool:
        return annotation in (Mapping, Sequence) or any(
            contains_broad_container(argument) for argument in typing.get_args(annotation)
        )

    strict_annotations = tuple(
        annotation for item in hints.values() for annotation in item.values()
    )
    assert all(
        not contains_broad_container(annotation) for annotation in strict_annotations
    )


def test_external_strict_mypy_accepts_natural_resolver_dictionary_shapes(
    tmp_path: Path,
) -> None:
    probe = tmp_path / "identity_resolver_consumer.py"
    probe.write_text(
        """\
from types import MappingProxyType

from operatebench.identity import (
    ComponentIdentityV1,
    CompositeIdentityV1,
    identity_to_json,
    recompute_bundle,
)


def accept_component_dict(
    resolver: dict[str, ComponentIdentityV1],
    composite: CompositeIdentityV1,
) -> None:
    recompute_bundle("runtime_bundle", resolver)
    identity_to_json(composite, resolved_member_identities=resolver)


def accept_composite_dict(
    resolver: dict[str, CompositeIdentityV1],
    composite: CompositeIdentityV1,
) -> None:
    recompute_bundle("runtime_bundle", resolver)
    identity_to_json(composite, resolved_member_identities=resolver)


def accept_mixed_dict(
    resolver: dict[str, ComponentIdentityV1 | CompositeIdentityV1],
    composite: CompositeIdentityV1,
) -> None:
    recompute_bundle("runtime_bundle", resolver)
    identity_to_json(composite, resolved_member_identities=resolver)


def accept_exact_frozen_equivalents(
    components: MappingProxyType[str, ComponentIdentityV1],
    composites: MappingProxyType[str, CompositeIdentityV1],
    mixed: MappingProxyType[
        str, ComponentIdentityV1 | CompositeIdentityV1
    ],
    composite: CompositeIdentityV1,
) -> None:
    recompute_bundle("runtime_bundle", components)
    recompute_bundle("runtime_bundle", composites)
    recompute_bundle("runtime_bundle", mixed)
    identity_to_json(composite, resolved_member_identities=components)
    identity_to_json(composite, resolved_member_identities=composites)
    identity_to_json(composite, resolved_member_identities=mixed)
""",
        encoding="utf-8",
    )
    mypy = shutil.which("mypy")
    assert mypy is not None
    result = subprocess.run(
        [mypy, "--strict", "--no-incremental", str(probe)],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_external_strict_mypy_rejects_nonexact_resolver_containers(
    tmp_path: Path,
) -> None:
    probe = tmp_path / "invalid_identity_resolver_consumer.py"
    probe.write_text(
        """\
from collections import ChainMap, UserDict

from operatebench.identity import (
    ComponentIdentityV1,
    CompositeIdentityV1,
    identity_to_json,
    recompute_bundle,
)


def reject_nonexact_containers(
    user_dict: UserDict[str, ComponentIdentityV1],
    chain_map: ChainMap[str, ComponentIdentityV1],
    pairs: tuple[tuple[str, ComponentIdentityV1], ...],
    composite: CompositeIdentityV1,
) -> None:
    recompute_bundle("runtime_bundle", user_dict)
    recompute_bundle("runtime_bundle", chain_map)
    recompute_bundle("runtime_bundle", pairs)
    identity_to_json(composite, resolved_member_identities=user_dict)
    identity_to_json(composite, resolved_member_identities=chain_map)
    identity_to_json(composite, resolved_member_identities=pairs)
""",
        encoding="utf-8",
    )
    mypy = shutil.which("mypy")
    assert mypy is not None
    result = subprocess.run(
        [mypy, "--strict", "--no-incremental", str(probe)],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert output.count("[arg-type]") == 6, output
    assert "UserDict[str, ComponentIdentityV1]" in output
    assert "ChainMap[str, ComponentIdentityV1]" in output
    assert "tuple[tuple[str, ComponentIdentityV1], ...]" in output


def test_strict_public_containers_still_reject_broad_abcs_and_tuples() -> None:
    identity = _identity_module()
    empty_mappings = (UserDict(), ChainMap())

    for projection in empty_mappings:
        with pytest.raises(
            identity.IdentityValidationError, match="ordinary JSON object"
        ):
            identity.recompute_component("card_schema_content", projection, [])
    with pytest.raises(
        identity.IdentityValidationError, match="requirements must be an array"
    ):
        identity.recompute_component("card_schema_content", {}, ())
    for resolver in (*empty_mappings, ()):
        with pytest.raises(
            identity.IdentityValidationError, match="ordinary or frozen mapping"
        ):
            identity.recompute_bundle("runtime_bundle", resolver)
    for manifest in (*empty_mappings, ()):
        with pytest.raises(identity.IdentityValidationError, match="ordinary object"):
            identity.recompute_manifest_digest(manifest)


def test_identity_module_has_no_card_or_version_dependency() -> None:
    identity = _identity_module()
    source = Path(identity.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert {name for name in imports if name.startswith("operatebench.")} == {
        "operatebench.jsonsafe"
    }
    identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert identifiers.isdisjoint({"OPERATEBENCH_VERSION", "__version__"})


def test_resources_are_byte_identical_and_registry_is_exact() -> None:
    identity = _identity_module()
    names = (
        "identity-manifest-v1.schema.json",
        "identity-component-registry-v1.json",
        "identity-compatibility-v1.registry.json",
        "identity-manifest-digest-v1.golden.json",
    )
    for name in names:
        packaged = files("operatebench.resources.identity").joinpath(name)
        normative = files("operatebench").joinpath(f"../../docs/schemas/{name}")
        assert packaged.read_bytes() == normative.read_bytes()
    registry = identity.load_component_registry()
    assert tuple(row["name"] for row in registry["components"]) == COMPONENTS
    assert {
        row["name"]: tuple(row["requires"]) for row in registry["components"]
    } == EDGES
    assert {
        row["name"]: tuple(row["members"]) for row in registry["composites"]
    } == BUNDLES
    domains = [
        row["domain"] for group in ("components", "composites") for row in registry[group]
    ]
    domains.append(identity.MANIFEST_DIGEST_DOMAIN)
    assert len(domains) == len(set(domains))
    assert all(
        row["schema_version"] == 1
        for group in ("components", "composites")
        for row in registry[group]
    )


def test_compatibility_contract_is_exact_and_loads_are_detached() -> None:
    identity = _identity_module()
    expected = {
        "schema": "operatebench.compatibility_contract.v1",
        "schema_version": 1,
        "requirement_kinds": {
            "exact_digest": 1,
            "closed_schema_version_range": 1,
            "named_predicate": 1,
        },
        "decision_predicates": {
            "can_runtime_replay": 1,
            "can_evaluator_regrade": 1,
            "can_analysis_recompute": 1,
            "can_compare": 1,
            "can_pool": 1,
        },
        "named_requirement_predicates": [],
    }
    first = identity.load_compatibility_registry()
    assert first["contract"] == expected
    first["contract"]["schema_version"] = 9
    assert identity.load_compatibility_registry()["contract"] == expected


def test_schema_is_draft_202012_strict_and_rejects_wrong_shapes() -> None:
    identity = _identity_module()
    jsonschema = pytest.importorskip("jsonschema")
    schema = identity.load_identity_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    for definition in schema["$defs"].values():
        if definition.get("type") == "object":
            assert definition.get("additionalProperties") is False
    identities = _all_identities()
    runtime = identity.recompute_bundle(
        "runtime_bundle", _bundle_members("runtime_bundle", identities)
    )
    experiment = identity.recompute_bundle(
        "experiment_bundle", _bundle_closure("experiment_bundle", identities)
    )
    manifest = {
        "schema": identity.IDENTITY_MANIFEST_SCHEMA,
        "schema_version": 1,
        "components": {
            name: identity.identity_to_json(value) for name, value in identities.items()
        },
        "composites": {
            "runtime_bundle": identity.identity_to_json(
                runtime,
                resolved_member_identities=_bundle_closure("runtime_bundle", identities),
            ),
            "experiment_bundle": identity.identity_to_json(
                experiment,
                resolved_member_identities=_bundle_closure(
                    "experiment_bundle", identities
                ),
            ),
        },
        "compatibility_contract": identity.load_compatibility_registry()["contract"],
        "manifest_digest_sha256": "0" * 64,
    }
    manifest["manifest_digest_sha256"] = identity.recompute_manifest_digest(manifest)
    assert not list(validator.iter_errors(manifest))
    mutations = []
    for key, value in (("unknown", 1), ("schema_version", None), ("components", [])):
        changed = deepcopy(manifest)
        changed[key] = value
        mutations.append(changed)
    missing = deepcopy(manifest)
    del missing["composites"]
    mutations.append(missing)
    boolean_version = deepcopy(manifest)
    boolean_version["schema_version"] = True
    mutations.append(boolean_version)
    for changed in mutations:
        assert list(validator.iter_errors(changed))


def test_component_recompute_is_canonical_immutable_and_detached() -> None:
    identity = _identity_module()
    projection = {"/z": [1, {"ok": True}], "/a~1b/~0x": "value"}
    result = identity.recompute_component("card_schema_content", projection, [])
    projection["/z"][1]["ok"] = False
    assert result.owned_fields == ("/a~1b/~0x", "/z")
    assert result.excludes == ()
    assert result.domain == "operatebench.digest.component.card_schema_content.v1"
    assert isinstance(result.owned_projection, MappingProxyType)
    serialized = identity.identity_to_json(result)
    assert serialized["owned_fields"] == ["/a~1b/~0x", "/z"]
    assert serialized["owned_projection"] == {
        "/a~1b/~0x": "value",
        "/z": [1, {"ok": True}],
    }
    wrapper = {
        "domain": result.domain,
        "schema": identity.COMPONENT_IDENTITY_SCHEMA,
        "content": {"/a~1b/~0x": "value", "/z": [1, {"ok": True}]},
    }
    assert (
        result.content_digest_sha256
        == hashlib.sha256(canonical_json_bytes(wrapper, "test")).hexdigest()
    )


def test_requirement_variants_are_normalized_and_metadata_moves_bundle() -> None:
    identity = _identity_module()
    root = identity.recompute_component("card_schema_content", {"/a": 1}, [])
    variants = (
        {"kind": "exact_digest", "content_digest_sha256": root.content_digest_sha256},
        {
            "kind": "closed_schema_version_range",
            "minimum_schema_version": 1,
            "maximum_schema_version": 2,
        },
        {
            "kind": "named_predicate",
            "predicate": "operatebench.requirement.test",
            "predicate_version": 1,
        },
    )
    outputs = []
    for compatible in variants:
        outputs.append(
            identity.recompute_component(
                "operation_core_content",
                {"/b": 2},
                [{"component": "card_schema_content", "compatible": compatible}],
            )
        )
    assert len({item.content_digest_sha256 for item in outputs}) == 1
    bundles = []
    all_ids = _all_identities()
    for output in outputs:
        changed = dict(all_ids)
        changed["operation_core_content"] = output
        bundles.append(
            identity.recompute_bundle(
                "runtime_bundle", _bundle_members("runtime_bundle", changed)
            ).bundle_digest_sha256
        )
    assert len(set(bundles)) == 3


def test_named_predicate_runtime_and_schema_string_caps_are_sound() -> None:
    identity = _identity_module()
    jsonschema = pytest.importorskip("jsonschema")
    requirement_schema = identity.load_identity_schema()["$defs"][
        "namedPredicateRequirement"
    ]
    validator = jsonschema.Draft202012Validator(requirement_schema)

    def requirement(predicate: str) -> dict[str, object]:
        return {
            "kind": "named_predicate",
            "predicate": predicate,
            "predicate_version": 1,
        }

    normal = requirement("operatebench.requirement.test")
    assert not list(validator.iter_errors(normal))

    schema_boundary = requirement("a." + "b" * (262_144 - 2))
    assert not list(validator.iter_errors(schema_boundary))
    assert list(validator.iter_errors(requirement("a." + "b" * (262_145 - 2))))

    runtime_boundary = requirement(
        "a." + "b" * (identity.MAX_COMPONENT_IDENTITY_BYTES - 1_000)
    )
    component = identity.recompute_component(
        "operation_core_content",
        {"/x": 1},
        [{"component": "card_schema_content", "compatible": runtime_boundary}],
    )
    assert identity.identity_to_json(component)["requires"][0]["compatible"] == (
        runtime_boundary
    )

    oversized = requirement("a." + "b" * (identity.MAX_IDENTITY_STRING_BYTES - 1))
    with pytest.raises(
        identity.IdentityValidationError,
        match=r"^requirement predicate exceeds Identity v1 resource limits$",
    ):
        identity.recompute_component(
            "operation_core_content",
            {"/x": 1},
            [{"component": "card_schema_content", "compatible": oversized}],
        )


@pytest.mark.parametrize(
    "bad",
    [
        {
            "kind": "closed_schema_version_range",
            "minimum_schema_version": True,
            "maximum_schema_version": 1,
        },
        {
            "kind": "closed_schema_version_range",
            "minimum_schema_version": 2,
            "maximum_schema_version": 1,
        },
        {
            "kind": "named_predicate",
            "predicate": "not-namespaced",
            "predicate_version": 1,
        },
        {"kind": "exact_digest", "content_digest_sha256": "A" * 64},
        {
            "kind": "exact_digest",
            "content_digest_sha256": "0" * 64,
            "predicate_version": 1,
        },
    ],
)
def test_malformed_or_ambiguous_requirement_is_refused(bad: dict[str, object]) -> None:
    identity = _identity_module()
    with pytest.raises(identity.IdentityValidationError):
        identity.recompute_component(
            "operation_core_content",
            {"/x": 1},
            [{"component": "card_schema_content", "compatible": bad}],
        )


def test_names_edges_members_and_paths_are_closed() -> None:
    identity = _identity_module()
    valid_root = identity.recompute_component("card_schema_content", {"/x": 1}, [])
    digest_requirement = [
        {
            "component": "card_schema_content",
            "compatible": {
                "kind": "exact_digest",
                "content_digest_sha256": valid_root.content_digest_sha256,
            },
        }
    ]
    bad_calls = (
        lambda: identity.recompute_component("unknown", {"/x": 1}, []),
        lambda: identity.recompute_component(
            "card_schema_content", {"/x": 1}, digest_requirement
        ),
        lambda: identity.recompute_component("operation_core_content", {"/x": 1}, []),
        lambda: identity.recompute_component(
            "operation_core_content", {"/x": 1}, digest_requirement * 2
        ),
        lambda: identity.recompute_component("card_schema_content", {"": 1}, []),
        lambda: identity.recompute_component("card_schema_content", {"/wild/*": 1}, []),
        lambda: identity.recompute_component(
            "card_schema_content", {"/" + "x" * 1025: 1}, []
        ),
        lambda: identity.recompute_component(
            "card_schema_content", {"/bad~2escape": 1}, []
        ),
        lambda: identity.recompute_bundle("unknown", {}),
        lambda: identity.recompute_bundle("runtime_bundle", {}),
    )
    for call in bad_calls:
        with pytest.raises(identity.IdentityValidationError):
            call()


def test_public_component_and_bundle_names_use_the_namespaced_string_cap() -> None:
    identity = _identity_module()
    oversized = "a_" + "b" * (identity.MAX_IDENTITY_STRING_BYTES - 1)
    for call in (
        lambda: identity.recompute_component(oversized, {"/x": 1}, []),
        lambda: identity.recompute_bundle(oversized, {}),
    ):
        with pytest.raises(
            identity.IdentityValidationError,
            match=r"exceeds Identity v1 resource limits$",
        ):
            call()


def test_fractional_float_cycle_surrogate_and_depth_refusals_have_domain_error() -> None:
    identity = _identity_module()
    cycle = {}
    cycle["self"] = cycle
    deep = value = {}
    for _ in range(MAX_JSON_DEPTH + 1):
        value["x"] = {}
        value = value["x"]
    bad_values = (1.5, float("nan"), float("inf"), "\ud800", cycle, deep, {1: "x"})
    for value in bad_values:
        with pytest.raises(identity.IdentityValidationError) as caught:
            identity.recompute_component("card_schema_content", {"/x": value}, [])
        assert caught.value.__cause__ is not None
    with pytest.raises(identity.IdentityValidationError):
        identity.recompute_component(True, {"/x": 1}, [])
    with pytest.raises(identity.IdentityValidationError):
        identity.recompute_component("card_schema_content", [("/x", 1)], [])


def test_bundle_and_manifest_wrappers_match_golden_vectors() -> None:
    identity = _identity_module()
    vectors = identity.load_identity_golden_vectors()
    component = identity.recompute_component(
        vectors["component"]["name"], vectors["component"]["owned_projection"], []
    )
    assert (
        canonical_json_bytes(vectors["component"]["wrapper"], "golden").hex()
        == vectors["component"]["canonical_utf8_hex"]
    )
    assert component.content_digest_sha256 == vectors["component"]["digest_sha256"]
    assert (
        hashlib.sha256(
            bytes.fromhex(vectors["component"]["canonical_utf8_hex"])
        ).hexdigest()
        == vectors["component"]["digest_sha256"]
    )

    identities = _all_identities()
    runtime_members = _bundle_members("runtime_bundle", identities)
    runtime = identity.recompute_bundle("runtime_bundle", runtime_members)
    runtime_wrapper = {
        "domain": runtime.domain,
        "schema": identity.COMPOSITE_IDENTITY_SCHEMA,
        "content": {
            "members": [
                {
                    "name": name,
                    "identity": identity.identity_to_json(runtime_members[name]),
                }
                for name in BUNDLES["runtime_bundle"]
            ]
        },
    }
    assert vectors["composite"]["wrapper"] == runtime_wrapper
    assert (
        canonical_json_bytes(runtime_wrapper, "golden").hex()
        == vectors["composite"]["canonical_utf8_hex"]
    )
    assert runtime.bundle_digest_sha256 == vectors["composite"]["digest_sha256"]
    experiment = identity.recompute_bundle(
        "experiment_bundle", _bundle_closure("experiment_bundle", identities)
    )
    manifest = {
        "schema": identity.IDENTITY_MANIFEST_SCHEMA,
        "schema_version": 1,
        "components": {
            name: identity.identity_to_json(item)
            for name, item in reversed(tuple(identities.items()))
        },
        "composites": {
            "experiment_bundle": identity.identity_to_json(
                experiment,
                resolved_member_identities=_bundle_closure(
                    "experiment_bundle", identities
                ),
            ),
            "runtime_bundle": identity.identity_to_json(
                runtime,
                resolved_member_identities=_bundle_closure("runtime_bundle", identities),
            ),
        },
        "compatibility_contract": identity.load_compatibility_registry()["contract"],
        "manifest_digest_sha256": "f" * 64,
    }
    digest = identity.recompute_manifest_digest(manifest)
    manifest_wrapper = {
        "domain": identity.MANIFEST_DIGEST_DOMAIN,
        "schema": identity.IDENTITY_MANIFEST_SCHEMA,
        "content": {
            key: value
            for key, value in manifest.items()
            if key != "manifest_digest_sha256"
        },
    }
    assert vectors["manifest"]["wrapper"] == manifest_wrapper
    assert (
        canonical_json_bytes(manifest_wrapper, "golden").hex()
        == vectors["manifest"]["canonical_utf8_hex"]
    )
    assert digest == vectors["manifest"]["digest_sha256"]
    reordered = dict(reversed(tuple(manifest.items())))
    assert identity.recompute_manifest_digest(reordered) == digest
    changed = deepcopy(manifest)
    changed["components"]["operation_core_content"]["requires"][0]["compatible"] = {
        "kind": "closed_schema_version_range",
        "minimum_schema_version": 1,
        "maximum_schema_version": 1,
    }
    assert identity.recompute_manifest_digest(changed) != digest


def test_bundle_refuses_incomplete_or_forged_member_identities() -> None:
    identity = _identity_module()
    identities = _all_identities()
    operation = identities["operation_core_content"]

    bad_components = (
        replace(operation, schema="operatebench.component_identity.v2"),
        replace(operation, schema_version=2),
        replace(operation, content_digest_sha256="f" * 64),
        replace(operation, owned_fields=("/different",)),
        replace(operation, excludes=("/metadata",)),
    )
    for bad in bad_components:
        changed = dict(identities)
        changed["operation_core_content"] = bad
        with pytest.raises(identity.IdentityValidationError):
            identity.recompute_bundle(
                "runtime_bundle", _bundle_members("runtime_bundle", changed)
            )

    runtime = identity.recompute_bundle(
        "runtime_bundle", _bundle_members("runtime_bundle", identities)
    )
    for bad in (
        replace(runtime, schema="operatebench.composite_identity.v2"),
        replace(runtime, schema_version=2),
        replace(runtime, members=tuple(reversed(runtime.members))),
        replace(runtime, bundle_digest_sha256="F" * 64),
    ):
        with pytest.raises(identity.IdentityValidationError):
            identity.recompute_bundle(
                "experiment_bundle",
                {
                    "runtime_bundle": bad,
                    "evaluator_bundle_content": identities["evaluator_bundle_content"],
                    "analysis_contract_content": identities["analysis_contract_content"],
                },
            )


def test_loader_duplicate_keys_and_invalid_constants_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity_module()
    identity._clear_resource_caches()
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1.5}', b"\xff"):
        monkeypatch.setattr(identity, "_read_resource_bytes", lambda _name, raw=raw: raw)
        identity._clear_resource_caches()
        with pytest.raises(identity.IdentityValidationError) as caught:
            identity.load_component_registry()
        assert caught.value.__cause__ is not None or "duplicate" in str(caught.value)


def test_registry_versions_and_empty_named_predicate_set_are_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity_module()
    cases = []

    component_registry = identity.load_component_registry()
    changed = deepcopy(component_registry)
    changed["schema_version"] = 2
    cases.append(
        ("identity-component-registry-v1.json", changed, identity.load_component_registry)
    )
    changed = deepcopy(component_registry)
    changed["components"][0]["schema_version"] = 2
    cases.append(
        ("identity-component-registry-v1.json", changed, identity.load_component_registry)
    )
    changed = deepcopy(component_registry)
    changed["components"][0]["schema_version"] = True
    cases.append(
        ("identity-component-registry-v1.json", changed, identity.load_component_registry)
    )
    changed = deepcopy(component_registry)
    changed["composites"][0]["schema_version"] = 2
    cases.append(
        ("identity-component-registry-v1.json", changed, identity.load_component_registry)
    )

    compatibility_registry = identity.load_compatibility_registry()
    changed = deepcopy(compatibility_registry)
    changed["schema_version"] = 2
    cases.append(
        (
            "identity-compatibility-v1.registry.json",
            changed,
            identity.load_compatibility_registry,
        )
    )
    changed = deepcopy(compatibility_registry)
    changed["contract"]["named_requirement_predicates"] = [
        "operatebench.requirement.unregistered"
    ]
    cases.append(
        (
            "identity-compatibility-v1.registry.json",
            changed,
            identity.load_compatibility_registry,
        )
    )

    for resource_name, changed, loader in cases:
        raw = json.dumps(changed).encode()
        monkeypatch.setattr(
            identity,
            "_read_resource_bytes",
            lambda name, expected=resource_name, raw=raw: (
                raw if name == expected else pytest.fail(f"unexpected resource {name}")
            ),
        )
        identity._clear_resource_caches()
        with pytest.raises(identity.IdentityValidationError):
            loader()


def test_identity_to_json_is_fresh_and_rejects_unsupported_objects() -> None:
    identity = _identity_module()
    result = identity.recompute_component("card_schema_content", {"/x": {"a": [1]}}, [])
    first = identity.identity_to_json(result)
    second = identity.identity_to_json(result)
    first["owned_fields"].append("/y")
    assert second["owned_fields"] == ["/x"]
    assert result.owned_projection["/x"]["a"] == (1,)
    with pytest.raises(identity.IdentityValidationError):
        identity.identity_to_json(object())


def test_experiment_bundle_requires_leaf_closure_and_rederives_runtime() -> None:
    identity = _identity_module()
    identities = _all_identities()
    runtime = identity.recompute_bundle(
        "runtime_bundle", _bundle_closure("runtime_bundle", identities)
    )
    for stale in ("0" * 64, "f" * 64):
        with pytest.raises(identity.IdentityValidationError):
            identity.recompute_bundle(
                "experiment_bundle",
                {
                    "runtime_bundle": replace(runtime, bundle_digest_sha256=stale),
                    "evaluator_bundle_content": identities["evaluator_bundle_content"],
                    "analysis_contract_content": identities["analysis_contract_content"],
                },
            )
    experiment = identity.recompute_bundle(
        "experiment_bundle", _bundle_closure("experiment_bundle", identities)
    )
    expected_runtime = identity.identity_to_json(
        runtime,
        resolved_member_identities=_bundle_closure("runtime_bundle", identities),
    )
    assert experiment.members == BUNDLES["experiment_bundle"]
    assert expected_runtime["bundle_digest_sha256"] == runtime.bundle_digest_sha256


def test_forged_component_cannot_be_projected_or_bundled() -> None:
    identity = _identity_module()
    identities = _all_identities()
    genuine = identities["operation_core_content"]
    forged = replace(genuine, content_digest_sha256="0" * 64)
    with pytest.raises(identity.IdentityValidationError):
        identity.identity_to_json(forged)
    changed = dict(identities)
    changed["operation_core_content"] = forged
    with pytest.raises(identity.IdentityValidationError):
        identity.recompute_bundle(
            "runtime_bundle", _bundle_closure("runtime_bundle", changed)
        )
    assert not hasattr(identity, "_component_projection")


def test_schema_strict_json_values_match_runtime_for_float_and_surrogate() -> None:
    identity = _identity_module()
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(identity.load_identity_schema())
    identities = _all_identities()
    runtime = identity.recompute_bundle(
        "runtime_bundle", _bundle_closure("runtime_bundle", identities)
    )
    experiment = identity.recompute_bundle(
        "experiment_bundle", _bundle_closure("experiment_bundle", identities)
    )
    manifest = {
        "schema": identity.IDENTITY_MANIFEST_SCHEMA,
        "schema_version": 1,
        "components": {
            name: identity.identity_to_json(item) for name, item in identities.items()
        },
        "composites": {
            "runtime_bundle": identity.identity_to_json(
                runtime,
                resolved_member_identities=_bundle_closure("runtime_bundle", identities),
            ),
            "experiment_bundle": identity.identity_to_json(
                experiment,
                resolved_member_identities=_bundle_closure(
                    "experiment_bundle", identities
                ),
            ),
        },
        "compatibility_contract": identity.load_compatibility_registry()["contract"],
        "manifest_digest_sha256": "0" * 64,
    }
    for bad in (1.5, json.loads('"\\ud800"')):
        changed = deepcopy(manifest)
        changed["components"]["card_schema_content"]["owned_projection"][
            "/fixture/card_schema_content"
        ] = {"nested": [bad]}
        assert list(validator.iter_errors(changed))
        with pytest.raises(identity.IdentityValidationError):
            identity.recompute_component(
                "card_schema_content", {"/x": {"nested": [bad]}}, []
            )


def test_pointer_empty_tokens_are_explicitly_refused() -> None:
    identity = _identity_module()
    for pointer in ("/", "/a/"):
        with pytest.raises(identity.IdentityValidationError, match="non-empty-token"):
            identity.recompute_component("card_schema_content", {pointer: 1}, [])


def test_hostile_objects_fail_static_without_invoking_hooks() -> None:
    identity = _identity_module()

    class EvilRepr:
        def __repr__(self):
            raise RuntimeError("HOSTILE_REPR_RAN")

    class EvilEq:
        def __eq__(self, _other):
            raise RuntimeError("HOSTILE_EQ_RAN")

    class EvilMapping(dict):
        def items(self):
            raise ValueError("HOSTILE_ITERATION")

    for call in (
        lambda: identity.recompute_component("card_schema_content", {EvilRepr(): 1}, []),
        lambda: identity.recompute_component(
            "operation_core_content",
            {"/x": 1},
            [{"component": "card_schema_content", "compatible": {"kind": EvilRepr()}}],
        ),
        lambda: identity.recompute_component(
            "card_schema_content", {"/x": EvilMapping()}, []
        ),
    ):
        with pytest.raises(identity.IdentityValidationError) as caught:
            call()
        assert "HOSTILE" not in str(caught.value)

    identities = _all_identities()
    forged = replace(identities["operation_core_content"], name=EvilEq())
    changed = dict(identities)
    changed["operation_core_content"] = forged
    with pytest.raises(identity.IdentityValidationError):
        identity.recompute_bundle(
            "runtime_bundle", _bundle_closure("runtime_bundle", changed)
        )


def test_packaged_resources_are_hash_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = _identity_module()
    loaders = {
        "identity-manifest-v1.schema.json": identity.load_identity_schema,
        "identity-component-registry-v1.json": identity.load_component_registry,
        "identity-compatibility-v1.registry.json": identity.load_compatibility_registry,
        "identity-manifest-digest-v1.golden.json": identity.load_identity_golden_vectors,
    }
    for resource_name, loader in loaders.items():
        for raw in (
            b'{"attacker":1}',
            files("operatebench.resources.identity").joinpath(resource_name).read_bytes()
            + b" ",
        ):
            monkeypatch.setattr(
                identity, "_read_resource_bytes", lambda _name, raw=raw: raw
            )
            identity._clear_resource_caches()
            with pytest.raises(identity.IdentityValidationError) as caught:
                loader()
            assert caught.value.__cause__ is not None


def test_resource_package_import_errors_are_translated_for_every_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity_module()
    loaders = (
        identity.load_identity_schema,
        identity.load_component_registry,
        identity.load_compatibility_registry,
        identity.load_identity_golden_vectors,
    )
    sentinel = ModuleNotFoundError("SENTINEL_RESOURCE_PACKAGE_MISSING")

    def unavailable(_package: str):
        raise sentinel

    monkeypatch.setattr(identity, "files", unavailable)
    for loader in loaders:
        identity._clear_resource_caches()
        with pytest.raises(
            identity.IdentityValidationError,
            match=r"^cannot resolve packaged identity resource package$",
        ) as caught:
            loader()
        assert caught.value.__cause__ is sentinel


@pytest.mark.parametrize(
    "sentinel",
    [
        TypeError("SENTINEL_PROGRAMMER_ERROR"),
        SystemExit("SENTINEL_EXIT"),
        BaseException("SENTINEL_BASE"),
    ],
)
def test_resource_package_non_import_failures_propagate_raw(
    monkeypatch: pytest.MonkeyPatch, sentinel: BaseException
) -> None:
    identity = _identity_module()

    def fail_raw(_package: str):
        raise sentinel

    monkeypatch.setattr(identity, "files", fail_raw)
    identity._clear_resource_caches()
    with pytest.raises(type(sentinel)) as caught:
        identity.load_identity_schema()
    assert caught.value is sentinel


def test_canonical_encoder_typeerror_propagates_as_programmer_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity_module()
    identities = _all_identities()
    runtime = identity.recompute_bundle(
        "runtime_bundle", _bundle_closure("runtime_bundle", identities)
    )
    manifest = {
        "schema": identity.IDENTITY_MANIFEST_SCHEMA,
        "schema_version": 1,
        "components": {},
        "composites": {},
        "compatibility_contract": {},
        "manifest_digest_sha256": "0" * 64,
    }

    def programmer_bug(*_args, **_kwargs):
        raise TypeError("PROGRAMMER_BUG")

    monkeypatch.setattr(identity, "canonical_json_bytes", programmer_bug)
    calls = (
        lambda: identity.recompute_component("card_schema_content", {"/x": 1}, []),
        lambda: identity.recompute_bundle(
            "runtime_bundle", _bundle_closure("runtime_bundle", identities)
        ),
        lambda: identity.recompute_manifest_digest(manifest),
    )
    for call in calls:
        with pytest.raises(TypeError, match="PROGRAMMER_BUG"):
            call()
    assert runtime.bundle_digest_sha256


def test_semantic_resource_caps_fail_before_canonical_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity_module()

    def sentinel(*_args, **_kwargs):
        pytest.fail("canonical encoder must not run")

    monkeypatch.setattr(identity, "canonical_json_bytes", sentinel)
    too_wide = {f"/f{i}": i for i in range(identity.MAX_OWNED_PROJECTION_FIELDS + 1)}
    with pytest.raises(identity.IdentityValidationError, match="resource limits"):
        identity.recompute_component("card_schema_content", too_wide, [])


def test_semantic_resource_cap_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = _identity_module()

    monkeypatch.setattr(identity, "MAX_OWNED_PROJECTION_FIELDS", 2)
    identity.recompute_component("card_schema_content", {"/a": 1, "/b": 2}, [])
    with pytest.raises(identity.IdentityValidationError, match="resource limits"):
        identity.recompute_component(
            "card_schema_content", {"/a": 1, "/b": 2, "/c": 3}, []
        )

    monkeypatch.setattr(identity, "MAX_OWNED_PROJECTION_FIELDS", 4_096)
    monkeypatch.setattr(identity, "MAX_STRICT_JSON_CONTAINER_ITEMS", 2)
    identity._strict_json([1, 2], "test value")
    with pytest.raises(identity.IdentityValidationError, match="resource limits"):
        identity._strict_json([1, 2, 3], "test value")

    monkeypatch.setattr(identity, "MAX_STRICT_JSON_CONTAINER_ITEMS", 4_096)
    monkeypatch.setattr(identity, "MAX_IDENTITY_STRING_BYTES", 4)
    identity._strict_json("test", "test value")
    with pytest.raises(identity.IdentityValidationError, match="resource limits"):
        identity._strict_json("tests", "test value")

    monkeypatch.setattr(identity, "MAX_IDENTITY_STRING_BYTES", 1024 * 1024)
    monkeypatch.setattr(identity, "MAX_STRICT_JSON_NODES", 3)
    identity._strict_json([1, 2], "test value")
    with pytest.raises(identity.IdentityValidationError, match="resource limits"):
        identity._strict_json([1, 2, 3], "test value")

    monkeypatch.setattr(identity, "MAX_STRICT_JSON_NODES", 65_536)
    monkeypatch.setattr(identity, "MAX_CANONICAL_UTF8_PAYLOAD_BYTES", 100)
    identity._strict_json("x" * 98, "test value")
    with pytest.raises(identity.IdentityValidationError, match="resource limits"):
        identity._strict_json("x" * 99, "test value")


def test_hostile_mapping_exception_is_translated_with_exact_cause() -> None:
    identity = _identity_module()

    class HostileMapping(dict):
        def items(self):
            raise ValueError("hostile mapping iteration")

    genuine = identity.recompute_component("card_schema_content", {"/x": 1}, [])
    forged = replace(genuine, owned_projection=MappingProxyType(HostileMapping()))
    with pytest.raises(identity.IdentityValidationError) as caught:
        identity.identity_to_json(forged)
    assert type(caught.value.__cause__) is ValueError


def test_mapping_exception_groups_propagate_unchanged() -> None:
    identity = _identity_module()

    groups = (
        ExceptionGroup("hostile", [ValueError("ordinary")]),
        BaseExceptionGroup("hostile", [KeyboardInterrupt()]),
    )

    class HostileMapping(dict):
        def __init__(self, raised):
            super().__init__()
            self.raised = raised

        def items(self):
            raise self.raised

    for group in groups:
        genuine = identity.recompute_component("card_schema_content", {"/x": 1}, [])
        forged = replace(
            genuine, owned_projection=MappingProxyType(HostileMapping(group))
        )
        with pytest.raises(BaseExceptionGroup) as caught:
            identity.identity_to_json(forged)
        assert caught.value is group


def test_mapping_proxy_iteration_is_incrementally_capped() -> None:
    identity = _identity_module()
    yielded = 0

    class HostileBacking(dict):
        def __len__(self):
            pytest.fail("hostile mapping length must not run")

        def items(self):
            def pairs():
                nonlocal yielded
                for index in range(identity.MAX_STRICT_JSON_NODES + 1_000):
                    yielded += 1
                    yield (f"k{index}", None)

            return pairs()

    with pytest.raises(identity.IdentityValidationError, match="resource limits"):
        identity._strict_json(
            MappingProxyType(HostileBacking()), "hostile proxy", allow_frozen=True
        )
    assert yielded == identity.MAX_STRICT_JSON_CONTAINER_ITEMS + 1


def test_infinite_mapping_proxy_iterator_terminates_at_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity_module()
    yielded = 0
    monkeypatch.setattr(identity, "MAX_STRICT_JSON_CONTAINER_ITEMS", 3)

    class InfiniteBacking(dict):
        def items(self):
            def pairs():
                nonlocal yielded
                index = 0
                while True:
                    yielded += 1
                    if yielded > 100:
                        raise RuntimeError("mapping iterator was not capped")
                    yield (f"k{index}", None)
                    index += 1

            return pairs()

    with pytest.raises(identity.IdentityValidationError, match="resource limits"):
        identity._strict_json(
            MappingProxyType(InfiniteBacking()), "infinite proxy", allow_frozen=True
        )
    assert yielded == 4


def test_requirement_width_is_refused_before_rows_or_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity_module()
    accessed = 0

    class SentinelRow(dict):
        def __getitem__(self, key):
            nonlocal accessed
            accessed += 1
            return super().__getitem__(key)

    row = SentinelRow(
        {
            "component": "card_schema_content",
            "compatible": {
                "kind": "exact_digest",
                "content_digest_sha256": "0" * 64,
            },
        }
    )
    oversized = [row] * 100_000
    monkeypatch.setattr(
        identity,
        "canonical_json_bytes",
        lambda *_args, **_kwargs: pytest.fail("canonical encoder must not run"),
    )
    with pytest.raises(identity.IdentityValidationError, match="registry cardinality"):
        identity.recompute_component("operation_core_content", {"/x": 1}, oversized)
    assert accessed == 0


def test_fixed_cardinality_identity_tuples_reject_before_element_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity_module()
    identities = _all_identities()
    genuine = identities["operation_core_content"]
    runtime_leaves = _bundle_closure("runtime_bundle", identities)
    runtime = identity.recompute_bundle("runtime_bundle", runtime_leaves)
    oversized = ("/never/visit",) * 100_000
    projection_visits = 0

    class CountingProjection(dict):
        def items(self):
            nonlocal projection_visits
            projection_visits += 1
            return super().items()

    projection = MappingProxyType(CountingProjection(dict(genuine.owned_projection)))
    started = time.perf_counter()
    for forged in (
        replace(genuine, owned_fields=oversized, owned_projection=projection),
        replace(genuine, excludes=oversized, owned_projection=projection),
    ):
        with pytest.raises(identity.IdentityValidationError):
            identity.identity_to_json(forged)
    assert time.perf_counter() - started < 1.0
    assert projection_visits == 0

    with pytest.raises(identity.IdentityValidationError):
        identity.identity_to_json(
            replace(runtime, members=("operation_core_content",) * 100_000),
            resolved_member_identities=runtime_leaves,
        )


def test_forged_requirement_tuple_width_is_refused_before_rows_or_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity_module()
    genuine = identity.recompute_component(
        "operation_core_content",
        {"/x": 1},
        [
            {
                "component": "card_schema_content",
                "compatible": {
                    "kind": "exact_digest",
                    "content_digest_sha256": "0" * 64,
                },
            }
        ],
    )
    visited = 0

    class CountingBacking(dict):
        def items(self):
            nonlocal visited
            visited += 1
            return super().items()

    requirement = identity.RequirementV1(
        "card_schema_content",
        MappingProxyType(
            CountingBacking({"kind": "exact_digest", "content_digest_sha256": "0" * 64})
        ),
    )
    forged = replace(genuine, requires=(requirement,) * 100_000)
    monkeypatch.setattr(
        identity,
        "canonical_json_bytes",
        lambda *_args, **_kwargs: pytest.fail("canonical encoder must not run"),
    )
    with pytest.raises(identity.IdentityValidationError, match="registry cardinality"):
        identity.identity_to_json(forged)
    assert visited == 0


def test_resolver_mapping_proxy_snapshot_stops_at_fixed_closure_plus_one() -> None:
    identity = _identity_module()
    yielded = 0

    class InfiniteResolver(dict):
        def __len__(self):
            pytest.fail("resolver proxy length must not be trusted")

        def items(self):
            def pairs():
                nonlocal yielded
                index = 0
                while True:
                    yielded += 1
                    if yielded > len(BUNDLES["runtime_bundle"]) + 1:
                        pytest.fail("resolver snapshot exceeded its fixed closure cap")
                    yield (f"hostile_{index}", object())
                    index += 1

            return pairs()

    with pytest.raises(identity.IdentityValidationError, match="fixed leaf closure"):
        identity.recompute_bundle("runtime_bundle", MappingProxyType(InfiniteResolver()))
    assert yielded == len(BUNDLES["runtime_bundle"]) + 1


def test_oversized_exact_dict_resolver_rejects_before_identity_value_hooks() -> None:
    identity = _identity_module()
    identities = _all_identities()
    genuine = identities["operation_core_content"]
    projection_visits = 0

    class CountingProjection(dict):
        def items(self):
            nonlocal projection_visits
            projection_visits += 1
            return super().items()

    forged = replace(
        genuine,
        owned_projection=MappingProxyType(
            CountingProjection(dict(genuine.owned_projection))
        ),
    )
    oversized = {f"hostile_{index}": forged for index in range(100_000)}
    started = time.perf_counter()
    with pytest.raises(identity.IdentityValidationError, match="fixed leaf closure"):
        identity.recompute_bundle("runtime_bundle", oversized)
    assert time.perf_counter() - started < 1.0
    assert projection_visits == 0


def test_resolver_mapping_proxy_hooks_and_duplicates_fail_closed() -> None:
    identity = _identity_module()

    class RaisingResolver(dict):
        def __init__(self, raised):
            super().__init__()
            self.raised = raised

        def items(self):
            raise self.raised

    ordinary = ValueError("HOSTILE_RESOLVER_ITEMS")
    with pytest.raises(identity.IdentityValidationError) as caught:
        identity.recompute_bundle(
            "runtime_bundle", MappingProxyType(RaisingResolver(ordinary))
        )
    assert caught.value.__cause__ is ordinary

    for raised in (
        ExceptionGroup("hostile", [ValueError("ordinary")]),
        KeyboardInterrupt(),
    ):
        with pytest.raises(BaseException) as caught:
            identity.recompute_bundle(
                "runtime_bundle", MappingProxyType(RaisingResolver(raised))
            )
        assert caught.value is raised

    class DuplicateResolver(dict):
        def items(self):
            return iter((("duplicate", object()), ("duplicate", object())))

    with pytest.raises(identity.IdentityValidationError, match="unique"):
        identity.recompute_bundle("runtime_bundle", MappingProxyType(DuplicateResolver()))


def test_resolver_snapshot_prevents_component_hook_toctou() -> None:
    identity = _identity_module()
    identities = _all_identities()
    safe_leaves = _bundle_closure("runtime_bundle", identities)
    runtime = identity.recompute_bundle("runtime_bundle", safe_leaves)
    assert (
        identity.recompute_bundle("runtime_bundle", MappingProxyType(safe_leaves))
        == runtime
    )
    expected_json = identity.identity_to_json(
        runtime, resolved_member_identities=safe_leaves
    )

    def hostile_leaves():
        resolver = _bundle_closure("runtime_bundle", identities)
        genuine = identities["operation_core_content"]

        class MutatingProjection(dict):
            def items(self):
                resolver.pop("semantic_scenario_content", None)
                return super().items()

        resolver["operation_core_content"] = replace(
            genuine,
            owned_projection=MappingProxyType(
                MutatingProjection(dict(genuine.owned_projection))
            ),
        )
        return resolver

    direct_resolver = hostile_leaves()
    assert identity.recompute_bundle("runtime_bundle", direct_resolver) == runtime
    assert "semantic_scenario_content" not in direct_resolver

    projection_resolver = hostile_leaves()
    assert (
        identity.identity_to_json(runtime, resolved_member_identities=projection_resolver)
        == expected_json
    )
    assert "semantic_scenario_content" not in projection_resolver


def test_composite_projection_requires_exact_leaf_recomputation() -> None:
    identity = _identity_module()
    identities = _all_identities()
    runtime_leaves = _bundle_closure("runtime_bundle", identities)
    experiment_leaves = _bundle_closure("experiment_bundle", identities)
    runtime = identity.recompute_bundle("runtime_bundle", runtime_leaves)
    experiment = identity.recompute_bundle("experiment_bundle", experiment_leaves)

    with pytest.raises(identity.IdentityValidationError, match="resolved leaves"):
        identity.identity_to_json(runtime)
    assert (
        identity.identity_to_json(runtime, resolved_member_identities=runtime_leaves)[
            "bundle_digest_sha256"
        ]
        == runtime.bundle_digest_sha256
    )
    assert (
        identity.identity_to_json(
            experiment, resolved_member_identities=experiment_leaves
        )["bundle_digest_sha256"]
        == experiment.bundle_digest_sha256
    )

    for forged in (
        replace(runtime, bundle_digest_sha256="0" * 64),
        copy(runtime),
    ):
        if forged.bundle_digest_sha256 == runtime.bundle_digest_sha256:
            object.__setattr__(forged, "bundle_digest_sha256", "f" * 64)
        with pytest.raises(identity.IdentityValidationError):
            identity.identity_to_json(forged, resolved_member_identities=runtime_leaves)
    with pytest.raises(identity.IdentityValidationError):
        identity.identity_to_json(runtime, resolved_member_identities={})
    object.__setattr__(runtime, "bundle_digest_sha256", "0" * 64)
    with pytest.raises(identity.IdentityValidationError):
        identity.identity_to_json(runtime, resolved_member_identities=runtime_leaves)
    assert not hasattr(identity, "_TRUSTED_COMPOSITE_OBJECTS")
    assert not hasattr(identity, "_remember_composite")


def test_semantic_integer_normalization_matches_schema_and_canonical_bytes() -> None:
    identity = _identity_module()
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(identity.load_identity_schema())
    safe = 2**53 - 1

    integer = identity.recompute_component(
        "card_schema_content", {"/x": {"nested": [1]}}, []
    )
    integral_float = identity.recompute_component(
        "card_schema_content", {"/x": {"nested": [1.0]}}, []
    )
    assert integral_float == integer
    assert identity.identity_to_json(integral_float)["owned_projection"]["/x"] == {
        "nested": [1]
    }

    for accepted in (-safe, safe, float(-safe), float(safe)):
        result = identity.recompute_component(
            "card_schema_content", {"/x": {"nested": [accepted]}}, []
        )
        projected = identity.identity_to_json(result)
        assert type(projected["owned_projection"]["/x"]["nested"][0]) is int

    for rejected in (-safe - 1, safe + 1, 1.5, math.nan, math.inf, -math.inf):
        with pytest.raises(identity.IdentityValidationError):
            identity.recompute_component(
                "card_schema_content", {"/x": {"nested": [rejected]}}, []
            )

    strict = identity.load_identity_schema()["$defs"]["strictJsonValue"]
    integer_branch = next(
        branch for branch in strict["oneOf"] if branch.get("type") == "integer"
    )
    assert integer_branch["minimum"] == -safe
    assert integer_branch["maximum"] == safe

    def assert_integer_bounds(value: object) -> None:
        if type(value) is dict:
            if value.get("type") == "integer":
                assert value["minimum"] >= -safe
                assert value["maximum"] <= safe
            for child in value.values():
                assert_integer_bounds(child)
        elif type(value) is list:
            for child in value:
                assert_integer_bounds(child)

    assert_integer_bounds(identity.load_identity_schema())
    for value in (1, 1.0, -safe, safe):
        assert not list(validator.evolve(schema=strict).iter_errors(value))
    for value in (1.5, -safe - 1, safe + 1):
        assert list(validator.evolve(schema=strict).iter_errors(value))


def test_utf8_runtime_cap_and_conservative_schema_bound() -> None:
    identity = _identity_module()
    jsonschema = pytest.importorskip("jsonschema")
    strict = identity.load_identity_schema()["$defs"]["strictJsonValue"]
    validator = jsonschema.Draft202012Validator(strict)
    string_branch = next(
        branch for branch in strict["oneOf"] if branch.get("type") == "string"
    )
    object_branch = next(
        branch for branch in strict["oneOf"] if branch.get("type") == "object"
    )
    assert string_branch["maxLength"] == 262_144
    assert object_branch["propertyNames"]["maxLength"] == 262_144

    four_byte = "\U00010000"
    assert not list(validator.iter_errors(four_byte * 262_144))
    assert list(validator.iter_errors(four_byte * 262_145))
    assert list(validator.iter_errors("é" * 524_289))
    identity.recompute_component("card_schema_content", {"/x": "a" * 262_145}, [])
    for over_limit in (
        four_byte * 262_144,
        "é" * 524_288,
        four_byte * 262_145,
        "é" * 524_289,
    ):
        with pytest.raises(identity.IdentityValidationError, match="resource limits"):
            identity.recompute_component("card_schema_content", {"/x": over_limit}, [])


def test_public_json_rejects_tuple_and_list_subclass_without_laundering() -> None:
    identity = _identity_module()

    class ListSubclass(list):
        pass

    for value in ((1, 2), ListSubclass([1, 2])):
        with pytest.raises(
            identity.IdentityValidationError, match="canonical Identity v1 JSON"
        ):
            identity.recompute_component("card_schema_content", {"/x": [value]}, [])
    with pytest.raises(identity.IdentityValidationError, match="requirements must be"):
        identity.recompute_component("card_schema_content", {"/x": 1}, ())
    result = identity.recompute_component("card_schema_content", {"/x": [[1, 2]]}, [])
    assert identity.identity_to_json(result)["owned_projection"]["/x"] == [[1, 2]]


def test_component_validation_never_uses_attacker_equality() -> None:
    identity = _identity_module()

    class EvilEqDict(dict):
        def __eq__(self, _other):
            raise RuntimeError("EVIL_EQ_RAN")

        def __ne__(self, _other):
            raise RuntimeError("EVIL_EQ_RAN")

    class EvilEq:
        def __eq__(self, _other):
            raise RuntimeError("EVIL_EQ_RAN")

    identities = _all_identities()
    genuine = identities["operation_core_content"]
    projection_proxy = MappingProxyType(EvilEqDict(dict(genuine.owned_projection)))
    proxied = replace(genuine, owned_projection=projection_proxy)
    assert identity.identity_to_json(proxied) == identity.identity_to_json(genuine)

    requirement = genuine.requires[0]
    compatible_proxy = MappingProxyType(EvilEqDict(dict(requirement.compatible)))
    proxied_requirement = replace(requirement, compatible=compatible_proxy)
    proxied = replace(genuine, requires=(proxied_requirement,))
    assert identity.identity_to_json(proxied) == identity.identity_to_json(genuine)

    for forged in (
        replace(genuine, name=EvilEq()),
        replace(genuine, content_digest_sha256=EvilEq()),
        replace(genuine.requires[0], component=EvilEq()),
    ):
        with pytest.raises(identity.IdentityValidationError) as caught:
            if type(forged) is identity.RequirementV1:
                identity.identity_to_json(replace(genuine, requires=(forged,)))
            else:
                identity.identity_to_json(forged)
        assert "EVIL_EQ_RAN" not in str(caught.value)


def test_oversized_names_and_digests_stop_before_regex_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity_module()

    class RegexMustNotRun:
        def fullmatch(self, _value):
            pytest.fail("regex must not run before the fixed-width guard")

    original_name_re = identity._NAME_RE
    monkeypatch.setattr(identity, "_NAME_RE", RegexMustNotRun())
    oversized_name = "a." + "b" * 10_000_000
    with pytest.raises(identity.IdentityValidationError, match="resource limits"):
        identity.recompute_component(oversized_name, {"/x": 1}, [])

    monkeypatch.setattr(identity, "_NAME_RE", original_name_re)
    monkeypatch.setattr(identity, "_DIGEST_RE", RegexMustNotRun())
    with pytest.raises(identity.IdentityValidationError, match="64 lowercase"):
        identity.recompute_component(
            "operation_core_content",
            {"/x": 1},
            [
                {
                    "component": "card_schema_content",
                    "compatible": {
                        "kind": "exact_digest",
                        "content_digest_sha256": "0" * 10_000_000,
                    },
                }
            ],
        )


def test_forged_identity_names_reject_before_registry_hash_lookup() -> None:
    identity = _identity_module()

    class EvilHash(str):
        def __hash__(self):
            raise RuntimeError("EVIL_HASH_RAN")

    identities = _all_identities()
    component = identities["operation_core_content"]
    runtime_leaves = _bundle_closure("runtime_bundle", identities)
    runtime = identity.recompute_bundle("runtime_bundle", runtime_leaves)
    for forged, kwargs in (
        (replace(component, name=EvilHash(component.name)), {}),
        (
            replace(runtime, name=EvilHash(runtime.name)),
            {"resolved_member_identities": runtime_leaves},
        ),
    ):
        with pytest.raises(identity.IdentityValidationError) as caught:
            identity.identity_to_json(forged, **kwargs)
        assert "EVIL_HASH_RAN" not in str(caught.value)


def test_packaged_resource_read_is_incrementally_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity_module()
    monkeypatch.setattr(identity, "MAX_IDENTITY_RESOURCE_BYTES", 100_000)
    read_sizes = []
    closed = False

    class EndlessStream:
        def read(self, size: int) -> bytes:
            read_sizes.append(size)
            return b"x" * size

        def close(self) -> None:
            nonlocal closed
            closed = True

    class Resource:
        def joinpath(self, _name: str):
            return self

        def open(self, mode: str):
            assert mode == "rb"
            return EndlessStream()

    monkeypatch.setattr(identity, "files", lambda _package: Resource())
    with pytest.raises(
        identity.IdentityValidationError,
        match=r"^packaged identity resource 'oversized.json' exceeds resource limits$",
    ):
        identity._read_resource_bytes("oversized.json")
    assert sum(read_sizes) == identity.MAX_IDENTITY_RESOURCE_BYTES + 1
    assert max(read_sizes) <= 64 * 1024
    assert closed


@pytest.mark.parametrize(
    "loader_name",
    [
        "load_identity_schema",
        "load_identity_manifest_projection_census_schema",
        "load_component_registry",
        "load_compatibility_registry",
        "load_identity_golden_vectors",
    ],
)
def test_every_resource_loader_detaches_fresh_cumulative_overflow_payloads(
    monkeypatch: pytest.MonkeyPatch, loader_name: str
) -> None:
    identity = _identity_module()
    forbidden_ids: set[int] = set()
    read_sizes: list[int] = []
    close_calls = 0

    class EndlessStream:
        def read(self, size: int) -> bytes:
            read_sizes.append(size)
            chunk = bytes([len(read_sizes) % 251 + 1]) * size
            forbidden_ids.add(id(chunk))
            return chunk

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1

    class Resource:
        def joinpath(self, _name: str):
            return self

        def open(self, mode: str):
            assert mode == "rb"
            stream = EndlessStream()
            forbidden_ids.add(id(stream))
            return stream

    monkeypatch.setattr(identity, "files", lambda _package: Resource())
    identity._clear_resource_caches()
    loader = getattr(identity, loader_name)
    with pytest.raises(
        identity.IdentityValidationError, match=r"exceeds resource limits$"
    ) as caught:
        loader()

    assert sum(read_sizes) == identity.MAX_IDENTITY_RESOURCE_BYTES + 1
    assert max(read_sizes) <= 64 * 1024
    assert len(read_sizes) == 65
    assert close_calls == 1
    _assert_resource_exception_graph_is_clean(caught.value, forbidden_ids=forbidden_ids)


def test_packaged_resource_overreturn_is_rejected_before_accumulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity_module()
    calls = []
    close_calls = 0

    class Stream:
        hostile = b"x" * (8 * 1024 * 1024)

        def read(self, size: int) -> bytes:
            calls.append(size)
            return self.hostile

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1

    class Resource:
        def joinpath(self, _name: str):
            return self

        def open(self, _mode: str):
            return Stream()

    monkeypatch.setattr(identity, "files", lambda _package: Resource())
    with pytest.raises(
        identity.IdentityValidationError,
        match=r"^packaged identity resource stream exceeded requested read size$",
    ) as caught:
        identity._read_resource_bytes("hostile.json")
    assert type(caught.value.__cause__) is ValueError
    _assert_tracebacks_do_not_retain(
        caught.value,
        Stream.hostile,
        max_local_size=identity._RESOURCE_READ_CHUNK_BYTES,
    )
    assert calls == [identity._RESOURCE_READ_CHUNK_BYTES]
    assert close_calls == 1


def test_packaged_resource_nonbytes_is_domain_error_with_trusted_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity_module()
    close_calls = 0

    class Stream:
        hostile = "x" * (8 * 1024 * 1024)

        def read(self, _size: int):
            return self.hostile

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1

    class Resource:
        def joinpath(self, _name: str):
            return self

        def open(self, _mode: str):
            return Stream()

    monkeypatch.setattr(identity, "files", lambda _package: Resource())
    with pytest.raises(
        identity.IdentityValidationError,
        match=r"^packaged identity resource stream returned non-bytes$",
    ) as caught:
        identity._read_resource_bytes("hostile.json")
    assert type(caught.value.__cause__) is TypeError
    assert str(caught.value.__cause__) == "packaged resource read must return bytes"
    _assert_tracebacks_do_not_retain(
        caught.value,
        Stream.hostile,
        max_local_size=identity._RESOURCE_READ_CHUNK_BYTES,
    )
    assert close_calls == 1


def test_packaged_resource_read_call_count_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity_module()
    calls = 0
    close_calls = 0

    class Stream:
        def read(self, _size: int) -> bytes:
            nonlocal calls
            calls += 1
            return b"x"

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1

    class Resource:
        def joinpath(self, _name: str):
            return self

        def open(self, _mode: str):
            return Stream()

    monkeypatch.setattr(identity, "files", lambda _package: Resource())
    with pytest.raises(
        identity.IdentityValidationError,
        match=r"^packaged identity resource 'endless.json' exceeds resource limits$",
    ):
        identity._read_resource_bytes("endless.json")
    assert calls == identity.MAX_IDENTITY_RESOURCE_READ_CALLS
    assert close_calls == 1


def test_ordinary_large_packaged_resource_read_is_exact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = _identity_module()
    raw = bytes(range(256)) * 4_096
    (tmp_path / "large.json").write_bytes(raw)
    monkeypatch.setattr(identity, "files", lambda _package: tmp_path)
    assert identity._read_resource_bytes("large.json") == raw


def test_packaged_resource_read_translates_open_and_read_oserrors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity_module()

    class Resource:
        def __init__(self, *, fail_open: bool):
            self.fail_open = fail_open

        def joinpath(self, _name: str):
            return self

        def open(self, _mode: str):
            if self.fail_open:
                raise OSError("OPEN_SENTINEL")

            class Stream:
                def read(self, _size: int) -> bytes:
                    raise OSError("READ_SENTINEL")

                def close(self) -> None:
                    pass

            return Stream()

    for fail_open in (True, False):
        monkeypatch.setattr(
            identity, "files", lambda _package, x=fail_open: Resource(fail_open=x)
        )
        with pytest.raises(
            identity.IdentityValidationError,
            match=r"^cannot read packaged identity resource 'broken.json'$",
        ) as caught:
            identity._read_resource_bytes("broken.json")
        assert type(caught.value.__cause__) is OSError


def test_packaged_resource_close_failure_does_not_mask_read_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity_module()
    read_failure = OSError("READ_SENTINEL")

    class Stream:
        def read(self, _size: int) -> bytes:
            raise read_failure

        def close(self) -> None:
            raise ValueError("CLOSE_MASK")

    class Resource:
        def joinpath(self, _name: str):
            return self

        def open(self, _mode: str):
            return Stream()

    monkeypatch.setattr(identity, "files", lambda _package: Resource())
    with pytest.raises(
        identity.IdentityValidationError,
        match=r"^cannot read packaged identity resource 'broken.json'$",
    ) as caught:
        identity._read_resource_bytes("broken.json")
    assert caught.value.__cause__ is read_failure


@pytest.mark.parametrize(
    "read_failure",
    [
        TypeError("READ_TYPE"),
        SystemExit("READ_EXIT"),
        ExceptionGroup("READ_GROUP", [ValueError("READ_MEMBER")]),
    ],
)
def test_packaged_resource_close_failure_does_not_mask_raw_base_failures(
    monkeypatch: pytest.MonkeyPatch, read_failure: BaseException
) -> None:
    identity = _identity_module()

    class Stream:
        def read(self, _size: int) -> bytes:
            raise read_failure

        def close(self) -> None:
            raise OSError("CLOSE_MASK")

    class Resource:
        def joinpath(self, _name: str):
            return self

        def open(self, _mode: str):
            return Stream()

    monkeypatch.setattr(identity, "files", lambda _package: Resource())
    with pytest.raises(type(read_failure)) as caught:
        identity._read_resource_bytes("broken.json")
    assert caught.value is read_failure


@pytest.mark.parametrize(
    "close_failure", [OSError("CLOSE_OS"), ValueError("CLOSE_VALUE")]
)
def test_packaged_resource_successful_read_translates_ordinary_close_failure(
    monkeypatch: pytest.MonkeyPatch, close_failure: Exception
) -> None:
    identity = _identity_module()

    class Stream:
        def read(self, _size: int) -> bytes:
            return b""

        def close(self) -> None:
            raise close_failure

    class Resource:
        def joinpath(self, _name: str):
            return self

        def open(self, _mode: str):
            return Stream()

    monkeypatch.setattr(identity, "files", lambda _package: Resource())
    with pytest.raises(
        identity.IdentityValidationError,
        match=r"^cannot read packaged identity resource 'broken.json'$",
    ) as caught:
        identity._read_resource_bytes("broken.json")
    assert caught.value.__cause__ is close_failure


@pytest.mark.parametrize(
    "close_failure",
    [
        SystemExit("CLOSE_EXIT"),
        ExceptionGroup("CLOSE_GROUP", [ValueError("CLOSE_MEMBER")]),
    ],
)
def test_packaged_resource_successful_read_propagates_control_close_failure(
    monkeypatch: pytest.MonkeyPatch, close_failure: BaseException
) -> None:
    identity = _identity_module()

    class Stream:
        def read(self, _size: int) -> bytes:
            return b""

        def close(self) -> None:
            raise close_failure

    class Resource:
        def joinpath(self, _name: str):
            return self

        def open(self, _mode: str):
            return Stream()

    monkeypatch.setattr(identity, "files", lambda _package: Resource())
    with pytest.raises(type(close_failure)) as caught:
        identity._read_resource_bytes("broken.json")
    assert caught.value is close_failure


@pytest.mark.parametrize("payload_size", [600_000, 336 * 1024 + 1])
def test_complete_component_envelope_rejects_oversized_projection(
    payload_size: int,
) -> None:
    identity = _identity_module()
    with pytest.raises(
        identity.IdentityValidationError,
        match=(
            r"^component card_schema_content identity exceeds "
            r"Identity v1 resource limits$"
        ),
    ):
        identity.recompute_component(
            "card_schema_content", {"/payload": "x" * payload_size}, []
        )


def test_eleven_near_budget_components_close_complete_manifest_budget() -> None:
    identity = _identity_module()
    identities = {}
    for name in COMPONENTS:
        requirements = _requirements(name, identities)

        def construct(
            node_count: int,
            payload_size: int,
            component_name=name,
            component_requirements=requirements,
        ):
            first = min(node_count, 4_096)
            first_payload = payload_size // 2
            projection = {
                f"/nodes/{component_name}/a": [None] * first,
                f"/nodes/{component_name}/b": [None] * (node_count - first),
                f"/payload/{component_name}/a": "x" * first_payload,
                f"/payload/{component_name}/b": "x" * (payload_size - first_payload),
            }
            return identity.recompute_component(
                component_name, projection, component_requirements
            )

        low, high = 0, identity.MAX_COMPONENT_IDENTITY_NODES
        while low < high:
            candidate = (low + high + 1) // 2
            try:
                construct(candidate, 0)
            except identity.IdentityValidationError:
                high = candidate - 1
            else:
                low = candidate
        node_count = low
        with pytest.raises(identity.IdentityValidationError, match="resource limits"):
            construct(node_count + 1, 0)

        low, high = 0, identity.MAX_COMPONENT_IDENTITY_BYTES
        while low < high:
            candidate = (low + high + 1) // 2
            try:
                construct(node_count, candidate)
            except identity.IdentityValidationError:
                high = candidate - 1
            else:
                low = candidate
        identities[name] = construct(node_count, low)
        with pytest.raises(identity.IdentityValidationError, match="resource limits"):
            construct(node_count, low + 1)
        serialized = identity.identity_to_json(identities[name])
        assert len(canonical_json_bytes(serialized, name)) == (
            identity.MAX_COMPONENT_IDENTITY_BYTES
        )

    runtime = identity.recompute_bundle(
        "runtime_bundle", _bundle_members("runtime_bundle", identities)
    )
    experiment = identity.recompute_bundle(
        "experiment_bundle", _bundle_closure("experiment_bundle", identities)
    )
    manifest = {
        "schema": identity.IDENTITY_MANIFEST_SCHEMA,
        "schema_version": 1,
        "components": {
            name: identity.identity_to_json(component)
            for name, component in identities.items()
        },
        "composites": {
            "runtime_bundle": identity.identity_to_json(
                runtime,
                resolved_member_identities=_bundle_closure("runtime_bundle", identities),
            ),
            "experiment_bundle": identity.identity_to_json(
                experiment,
                resolved_member_identities=_bundle_closure(
                    "experiment_bundle", identities
                ),
            ),
        },
        "compatibility_contract": identity.load_compatibility_registry()["contract"],
        "manifest_digest_sha256": "0" * 64,
    }

    def count_nodes(value) -> int:
        if type(value) is dict:
            return 1 + sum(1 + count_nodes(child) for child in value.values())
        if type(value) is list:
            return 1 + sum(count_nodes(child) for child in value)
        return 1

    assert len(canonical_json_bytes(manifest, "near-cap manifest")) <= (
        identity.MAX_CANONICAL_UTF8_PAYLOAD_BYTES
    )
    assert count_nodes(manifest) <= identity.MAX_STRICT_JSON_NODES
    manifest["manifest_digest_sha256"] = identity.recompute_manifest_digest(manifest)
    assert len(manifest["manifest_digest_sha256"]) == 64
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(identity.load_identity_schema())
    assert not list(validator.iter_errors(manifest))


def test_complete_component_node_envelope_is_bounded() -> None:
    identity = _identity_module()
    accepted = [[None] * 2_790, [None] * 2_790]
    assert identity.recompute_component("card_schema_content", {"/nodes": accepted}, [])
    rejected = [[None] * 2_795, [None] * 2_795]
    with pytest.raises(identity.IdentityValidationError, match="resource limits"):
        identity.recompute_component("card_schema_content", {"/nodes": rejected}, [])


def test_identity_schema_patterns_use_absolute_end_and_reject_newline_suffixes() -> None:
    identity = _identity_module()
    jsonschema = pytest.importorskip("jsonschema")
    schema = identity.load_identity_schema()
    patterns = []

    def collect(value: object) -> None:
        if type(value) is dict:
            for key, child in value.items():
                if key == "pattern":
                    patterns.append(child)
                collect(child)
        elif type(value) is list:
            for child in value:
                collect(child)

    collect(schema)
    assert patterns
    assert all(not pattern.endswith("$") for pattern in patterns)
    assert all(pattern.endswith(r"(?![\s\S])") for pattern in patterns)

    identities = _all_identities()
    runtime_leaves = _bundle_closure("runtime_bundle", identities)
    experiment_leaves = _bundle_closure("experiment_bundle", identities)
    runtime = identity.recompute_bundle("runtime_bundle", runtime_leaves)
    experiment = identity.recompute_bundle("experiment_bundle", experiment_leaves)
    manifest = {
        "schema": identity.IDENTITY_MANIFEST_SCHEMA,
        "schema_version": 1,
        "components": {
            name: identity.identity_to_json(item) for name, item in identities.items()
        },
        "composites": {
            "runtime_bundle": identity.identity_to_json(
                runtime, resolved_member_identities=runtime_leaves
            ),
            "experiment_bundle": identity.identity_to_json(
                experiment, resolved_member_identities=experiment_leaves
            ),
        },
        "compatibility_contract": identity.load_compatibility_registry()["contract"],
        "manifest_digest_sha256": "0" * 64,
    }
    manifest["manifest_digest_sha256"] = identity.recompute_manifest_digest(manifest)
    validator = jsonschema.Draft202012Validator(schema)
    assert not list(validator.iter_errors(manifest))
    for suffix in ("\n", "\r\n"):
        for path in ("digest", "predicate"):
            changed = deepcopy(manifest)
            compatible = changed["components"]["operation_core_content"]["requires"][0][
                "compatible"
            ]
            if path == "digest":
                compatible["content_digest_sha256"] += suffix
            else:
                compatible.clear()
                compatible.update(
                    {
                        "kind": "named_predicate",
                        "predicate": "operatebench.requirement.test" + suffix,
                        "predicate_version": 1,
                    }
                )
            changed["manifest_digest_sha256"] = identity.recompute_manifest_digest(
                changed
            )
            assert list(validator.iter_errors(changed))

    node = shutil.which("node")
    if node is not None:
        script = "for (const p of JSON.parse(process.argv[1])) new RegExp(p);"
        subprocess.run([node, "-e", script, json.dumps(patterns)], check=True)
