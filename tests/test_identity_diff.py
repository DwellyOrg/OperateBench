# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
from types import MappingProxyType

import pytest

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
RUNTIME_MEMBERS = (*COMPONENTS[2:9], COMPONENTS[1])


def _manifest(
    projections: dict[str, dict[str, object]] | None = None,
    *,
    ranges: bool = False,
    range_owners: frozenset[str] = frozenset(),
):
    import operatebench.identity as identity

    projections = projections or {
        name: {f"/fixture/{name}": index} for index, name in enumerate(COMPONENTS)
    }
    records = {}
    for name in COMPONENTS:
        requirements = [
            {
                "component": dependency,
                "compatible": (
                    {
                        "kind": "closed_schema_version_range",
                        "minimum_schema_version": 1,
                        "maximum_schema_version": 1,
                    }
                    if ranges or name in range_owners
                    else {
                        "kind": "exact_digest",
                        "content_digest_sha256": records[
                            dependency
                        ].content_digest_sha256,
                    }
                ),
            }
            for dependency in EDGES[name]
        ]
        records[name] = identity.recompute_component(
            name, deepcopy(projections[name]), requirements
        )
    runtime_leaves = {name: records[name] for name in RUNTIME_MEMBERS}
    experiment_leaves = {name: records[name] for name in COMPONENTS[1:]}
    runtime = identity.recompute_bundle("runtime_bundle", runtime_leaves)
    experiment = identity.recompute_bundle("experiment_bundle", experiment_leaves)
    wire = {
        "schema": identity.IDENTITY_MANIFEST_SCHEMA,
        "schema_version": 1,
        "components": {
            name: identity.identity_to_json(record) for name, record in records.items()
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
    wire["manifest_digest_sha256"] = identity.recompute_manifest_digest(wire)
    census = {
        "schema": identity.IDENTITY_MANIFEST_PROJECTION_CENSUS_SCHEMA,
        "schema_version": 1,
        "manifest_schema": identity.IDENTITY_MANIFEST_SCHEMA,
        "owners": list(COMPONENTS),
        "entries": [
            {"field_path": path, "owner": owner}
            for owner in COMPONENTS
            for path in sorted(projections[owner])
        ],
    }
    return identity.validate_identity_manifest(wire, census, projections)


def test_owned_mutation_record_and_public_surface_are_exact() -> None:
    import operatebench.identity as identity

    assert identity.OWNED_MUTATION_CONTRACT_VERSION == 1
    assert [field.name for field in fields(identity.OwnedMutation)] == [
        "mutation_contract_version",
        "owner",
        "changed_field_paths",
        "changed_requirement_components",
        "old_content_digest_sha256",
        "new_content_digest_sha256",
        "changed_composites",
    ]
    mutation = identity.OwnedMutation(
        1, "card_schema_content", (), (), "0" * 64, "1" * 64, ()
    )
    assert mutation.mutation_contract_version == 1
    assert not hasattr(mutation, "__dict__")
    with pytest.raises(FrozenInstanceError):
        mutation.owner = "operation_core_content"
    assert "OWNED_MUTATION_CONTRACT_VERSION" in identity.__all__
    assert "OwnedMutation" in identity.__all__
    assert "identity_diff" in identity.__all__


def test_identical_manifests_have_exact_empty_tuple_diff() -> None:
    import operatebench.identity as identity

    old = _manifest()
    assert identity.identity_diff(old, old) == ()
    assert type(identity.identity_diff(old, _manifest())) is tuple


@pytest.mark.parametrize("owner", COMPONENTS)
def test_one_projection_change_is_owned_and_registry_composites_are_exact(
    owner: str,
) -> None:
    import operatebench.identity as identity

    old_projections = {
        name: {f"/fixture/{name}": index} for index, name in enumerate(COMPONENTS)
    }
    new_projections = deepcopy(old_projections)
    path = f"/fixture/{owner}"
    new_projections[owner][path] = 99
    old = _manifest(old_projections, ranges=True)
    new = _manifest(new_projections, ranges=True)

    assert identity.identity_diff(old, new) == (
        identity.OwnedMutation(
            1,
            owner,
            (path,),
            (),
            old.components[owner].content_digest_sha256,
            new.components[owner].content_digest_sha256,
            (
                ()
                if owner == "card_schema_content"
                else ("experiment_bundle",)
                if owner in ("evaluator_bundle_content", "analysis_contract_content")
                else ("runtime_bundle", "experiment_bundle")
            ),
        ),
    )


@pytest.mark.parametrize("owner", COMPONENTS)
@pytest.mark.parametrize("operation", ("add", "remove"))
def test_paths_added_or_removed_under_the_same_owner_are_legal(
    owner: str, operation: str
) -> None:
    import operatebench.identity as identity

    old_projections: dict[str, dict[str, object]] = {
        name: {f"/fixture/{name}": index} for index, name in enumerate(COMPONENTS)
    }
    new_projections = deepcopy(old_projections)
    path = f"/additional/{owner}"
    if operation == "add":
        new_projections[owner][path] = None
    else:
        old_projections[owner][path] = None
    rows = identity.identity_diff(
        _manifest(old_projections, ranges=True),
        _manifest(new_projections, ranges=True),
    )
    assert len(rows) == 1
    assert rows[0].owner == owner
    assert rows[0].changed_field_paths == (path,)


def test_strict_json_changes_are_type_sensitive_ordered_and_deduplicated() -> None:
    import operatebench.identity as identity

    owner = "card_schema_content"
    old_projections: dict[str, dict[str, object]] = {name: {} for name in COMPONENTS}
    new_projections = deepcopy(old_projections)
    old_projections[owner] = {"/z": True, "/a": {"b": 1, "a": [1, 2]}, "/same": 1}
    new_projections[owner] = {"/z": 1, "/a": {"a": [2, 1], "b": 1}, "/same": 1.0}
    rows = identity.identity_diff(
        _manifest(old_projections, ranges=True),
        _manifest(new_projections, ranges=True),
    )
    assert rows[0].changed_field_paths == ("/a", "/z")


@pytest.mark.parametrize("owner", tuple(name for name in COMPONENTS if EDGES[name]))
def test_requirement_only_change_names_dependency_and_keeps_content_digest(
    owner: str,
) -> None:
    import operatebench.identity as identity

    old = _manifest()
    new = _manifest(range_owners=frozenset({owner}))
    row = next(row for row in identity.identity_diff(old, new) if row.owner == owner)
    assert row.changed_field_paths == ()
    assert row.changed_requirement_components == EDGES[owner]
    assert row.old_content_digest_sha256 == row.new_content_digest_sha256


@pytest.mark.parametrize(
    ("changed_owner", "dependents"),
    (
        (
            "card_schema_content",
            (
                "operation_core_content",
                "semantic_scenario_content",
                "variant_content",
                "runtime_contract_content",
                "evaluator_bundle_content",
            ),
        ),
        ("build_provenance_content", ("provider_run_plan_content",)),
        (
            "operation_core_content",
            (
                "semantic_scenario_content",
                "runtime_contract_content",
                "evaluator_bundle_content",
            ),
        ),
        ("semantic_scenario_content", ("variant_content", "arm_protocol_content")),
        ("variant_content", ("arm_protocol_content",)),
        (
            "runtime_contract_content",
            (
                "scaffold_content",
                "arm_protocol_content",
                "provider_run_plan_content",
                "evaluator_bundle_content",
            ),
        ),
        ("scaffold_content", ("provider_run_plan_content",)),
        ("arm_protocol_content", ()),
        ("provider_run_plan_content", ()),
        ("evaluator_bundle_content", ("analysis_contract_content",)),
        ("analysis_contract_content", ()),
    ),
)
def test_exact_digest_changes_propagate_only_to_direct_dependents(
    changed_owner: str, dependents: tuple[str, ...]
) -> None:
    import operatebench.identity as identity

    old_projections: dict[str, dict[str, object]] = {
        name: {f"/fixture/{name}": index} for index, name in enumerate(COMPONENTS)
    }
    new_projections = deepcopy(old_projections)
    new_projections[changed_owner][f"/fixture/{changed_owner}"] = 99
    rows = identity.identity_diff(_manifest(old_projections), _manifest(new_projections))
    assert tuple(row.owner for row in rows) == tuple(
        owner for owner in COMPONENTS if owner == changed_owner or owner in dependents
    )
    for row in rows:
        if row.owner in dependents:
            assert row.changed_field_paths == ()
            assert row.changed_requirement_components == (changed_owner,)


def test_owner_reassignment_and_forged_records_fail_closed() -> None:
    import operatebench.identity as identity

    old_projections: dict[str, dict[str, object]] = {name: {} for name in COMPONENTS}
    new_projections = deepcopy(old_projections)
    old_projections["card_schema_content"]["/moved"] = 1
    new_projections["operation_core_content"]["/moved"] = 1
    old = _manifest(old_projections, ranges=True)
    new = _manifest(new_projections, ranges=True)
    with pytest.raises(identity.IdentityValidationError, match="reassignment"):
        identity.identity_diff(old, new)

    genuine = _manifest()
    component = genuine.components["card_schema_content"]
    forged_components = dict(genuine.components)
    forged_components["card_schema_content"] = replace(
        component, content_digest_sha256="0" * 64
    )
    with pytest.raises(identity.IdentityValidationError):
        identity.identity_diff(
            replace(genuine, components=MappingProxyType(forged_components)), genuine
        )
    forged_composites = dict(genuine.composites)
    forged_composites["runtime_bundle"] = replace(
        genuine.composites["runtime_bundle"], bundle_digest_sha256="0" * 64
    )
    with pytest.raises(identity.IdentityValidationError):
        identity.identity_diff(
            replace(genuine, composites=MappingProxyType(forged_composites)), genuine
        )
    with pytest.raises(identity.IdentityValidationError, match="manifest digest"):
        identity.identity_diff(replace(genuine, manifest_digest_sha256="0" * 64), genuine)
    changed_contract = dict(genuine.compatibility_contract)
    changed_contract["schema_version"] = 2
    with pytest.raises(identity.IdentityValidationError, match="compatibility"):
        identity.identity_diff(
            replace(genuine, compatibility_contract=MappingProxyType(changed_contract)),
            genuine,
        )


def test_exact_input_classes_and_no_requirement_evaluator_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import operatebench.identity as identity

    genuine = _manifest()

    class ManifestSubclass(identity.ValidatedIdentityManifest):
        pass

    with pytest.raises(identity.IdentityValidationError, match="exact validated"):
        identity.identity_diff({}, genuine)
    subclass = ManifestSubclass(
        genuine.schema,
        genuine.schema_version,
        genuine.components,
        genuine.composites,
        genuine.compatibility_contract,
        genuine.manifest_digest_sha256,
    )
    with pytest.raises(identity.IdentityValidationError, match="exact validated"):
        identity.identity_diff(subclass, genuine)
    monkeypatch.setattr(
        identity,
        "evaluate_component_requirements",
        lambda *_args, **_kwargs: pytest.fail("requirement evaluator must not run"),
    )
    assert identity.identity_diff(genuine, genuine) == ()


def test_duplicate_ownership_and_hostile_snapshot_fail_without_retention() -> None:
    import operatebench.identity as identity

    genuine = _manifest()
    operation = genuine.components["operation_core_content"]
    duplicate = identity.recompute_component(
        "operation_core_content",
        {"/fixture/card_schema_content": 1},
        [
            {
                "component": requirement.component,
                "compatible": dict(requirement.compatible),
            }
            for requirement in operation.requires
        ],
    )
    forged_components = dict(genuine.components)
    forged_components["operation_core_content"] = duplicate
    forged = replace(genuine, components=MappingProxyType(forged_components))
    with pytest.raises(identity.IdentityValidationError, match="unique ownership"):
        identity.identity_diff(forged, genuine)

    class HostileBacking(dict):
        def items(self):
            raise ValueError("ATTACKER_CONTROLLED_TEXT")

    hostile = replace(genuine, components=MappingProxyType(HostileBacking()))
    with pytest.raises(identity.IdentityValidationError) as caught:
        identity.identity_diff(hostile, genuine)
    assert "ATTACKER_CONTROLLED_TEXT" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    traceback = caught.value.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_globals.get("__name__") == "operatebench.identity":
            assert all(
                value is not hostile for value in traceback.tb_frame.f_locals.values()
            )
        traceback = traceback.tb_next


def test_exception_groups_propagate_without_retaining_manifest() -> None:
    import operatebench.identity as identity

    genuine = _manifest()
    sentinel = ExceptionGroup("sentinel", [ValueError("ordinary")])

    class GroupBacking(dict):
        def items(self):
            raise sentinel

    hostile = replace(genuine, components=MappingProxyType(GroupBacking()))
    with pytest.raises(ExceptionGroup) as caught:
        identity.identity_diff(hostile, genuine)
    assert caught.value is sentinel
    traceback = caught.value.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_globals.get("__name__") == "operatebench.identity":
            assert all(
                value is not hostile for value in traceback.tb_frame.f_locals.values()
            )
        traceback = traceback.tb_next
