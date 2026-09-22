# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tomllib
from collections import ChainMap, UserDict
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any

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
RUNTIME_MEMBERS = (
    "operation_core_content",
    "semantic_scenario_content",
    "variant_content",
    "runtime_contract_content",
    "scaffold_content",
    "arm_protocol_content",
    "provider_run_plan_content",
    "build_provenance_content",
)


def _identity():
    import operatebench.identity as identity

    return identity


def _fixture(*, empty: bool = False, integral_float: bool = False):
    identity = _identity()
    projections = {
        name: (
            {}
            if empty
            else {f"/fixture/{name}": float(index) if integral_float else index}
        )
        for index, name in enumerate(COMPONENTS)
    }
    identities = {}
    for name in COMPONENTS:
        requirements = [
            {
                "component": dependency,
                "compatible": {
                    "kind": "exact_digest",
                    "content_digest_sha256": identities[dependency].content_digest_sha256,
                },
            }
            for dependency in EDGES[name]
        ]
        identities[name] = identity.recompute_component(
            name, deepcopy(projections[name]), requirements
        )
    runtime = identity.recompute_bundle(
        "runtime_bundle", {name: identities[name] for name in RUNTIME_MEMBERS}
    )
    experiment_members = {
        name: identities[name]
        for name in (
            *RUNTIME_MEMBERS,
            "evaluator_bundle_content",
            "analysis_contract_content",
        )
    }
    experiment = identity.recompute_bundle("experiment_bundle", experiment_members)
    manifest = {
        "schema": identity.IDENTITY_MANIFEST_SCHEMA,
        "schema_version": 1,
        "components": {
            name: identity.identity_to_json(record) for name, record in identities.items()
        },
        "composites": {
            "runtime_bundle": identity.identity_to_json(
                runtime,
                resolved_member_identities={
                    name: identities[name] for name in RUNTIME_MEMBERS
                },
            ),
            "experiment_bundle": identity.identity_to_json(
                experiment, resolved_member_identities=experiment_members
            ),
        },
        "compatibility_contract": identity.load_compatibility_registry()["contract"],
        "manifest_digest_sha256": "0" * 64,
    }
    manifest["manifest_digest_sha256"] = identity.recompute_manifest_digest(manifest)
    census = {
        "schema": "operatebench.identity_manifest_projection_census.v1",
        "schema_version": 1,
        "manifest_schema": identity.IDENTITY_MANIFEST_SCHEMA,
        "owners": list(COMPONENTS),
        "entries": [
            {"field_path": path, "owner": owner}
            for owner in COMPONENTS
            for path in sorted(projections[owner])
        ],
    }
    return manifest, census, projections


def _refresh_downstream(manifest, projections):
    """Refresh all attacker-controlled digests without invoking the validator."""
    identity = _identity()
    records = {}
    for name in COMPONENTS:
        records[name] = identity.recompute_component(
            name, projections[name], manifest["components"][name]["requires"]
        )
        manifest["components"][name] = identity.identity_to_json(records[name])
    runtime = identity.recompute_bundle(
        "runtime_bundle", {name: records[name] for name in RUNTIME_MEMBERS}
    )
    experiment_members = {
        name: records[name]
        for name in (
            *RUNTIME_MEMBERS,
            "evaluator_bundle_content",
            "analysis_contract_content",
        )
    }
    experiment = identity.recompute_bundle("experiment_bundle", experiment_members)
    manifest["composites"]["runtime_bundle"] = identity.identity_to_json(
        runtime,
        resolved_member_identities={name: records[name] for name in RUNTIME_MEMBERS},
    )
    manifest["composites"]["experiment_bundle"] = identity.identity_to_json(
        experiment, resolved_member_identities=experiment_members
    )
    manifest["manifest_digest_sha256"] = identity.recompute_manifest_digest(manifest)


def test_validates_complete_projection_backed_manifest_and_detaches_output() -> None:
    identity = _identity()
    manifest, census, projections = _fixture(integral_float=True)

    path = f"/fixture/{COMPONENTS[1]}"
    assert type(manifest["components"][COMPONENTS[1]]["owned_projection"][path]) is int
    assert type(projections[COMPONENTS[1]][path]) is float

    result = identity.validate_identity_manifest(manifest, census, projections)

    assert type(result) is identity.ValidatedIdentityManifest
    assert tuple(result.components) == COMPONENTS
    assert tuple(result.composites) == ("runtime_bundle", "experiment_bundle")
    assert type(result.components) is MappingProxyType
    assert type(result.composites) is MappingProxyType
    assert type(result.compatibility_contract) is MappingProxyType
    assert (
        result.components[COMPONENTS[1]].owned_projection[f"/fixture/{COMPONENTS[1]}"]
        == 1
    )
    assert result.manifest_digest_sha256 == manifest["manifest_digest_sha256"]
    with pytest.raises(TypeError):
        result.components[COMPONENTS[0]].owned_projection["/new"] = 1
    with pytest.raises(TypeError):
        result.compatibility_contract["schema_version"] = 2
    with pytest.raises(FrozenInstanceError):
        result.schema_version = 2
    projections[COMPONENTS[0]][f"/fixture/{COMPONENTS[0]}"] = 99
    manifest["components"][COMPONENTS[0]]["owned_projection"].clear()
    census["entries"].clear()
    assert len(result.components[COMPONENTS[0]].owned_projection) == 1
    assert result == replace(result)


def test_empty_component_projections_are_present_and_valid() -> None:
    identity = _identity()
    manifest, census, projections = _fixture(empty=True)
    result = identity.validate_identity_manifest(manifest, census, projections)
    assert all(not record.owned_fields for record in result.components.values())
    assert census["entries"] == []


def test_rejects_b1_census_as_statically_insufficient_scope() -> None:
    identity = _identity()
    manifest, _, projections = _fixture()
    b1 = {
        "schema": "operatebench.identity_field_census.v1",
        "schema_version": 1,
        "artifact_schema": "operatebench.b1_card_contract.v1",
        "entries": [],
    }
    with pytest.raises(identity.IdentityValidationError, match="insufficient scope"):
        identity.validate_identity_manifest(manifest, b1, projections)


class _HostileText(str):
    calls = 0
    raised: BaseException = RuntimeError("hostile text protocol ran")

    def __hash__(self) -> int:
        return str.__hash__(self)

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise type(self).raised


class _HostileList(list):
    calls = 0
    raised: BaseException = RuntimeError("hostile list protocol ran")

    def __iter__(self):
        type(self).calls += 1
        raise type(self).raised

    def __len__(self) -> int:
        type(self).calls += 1
        raise type(self).raised


class _HostileDict(dict):
    calls = 0
    raised: BaseException = RuntimeError("hostile dict protocol ran")

    def items(self):
        type(self).calls += 1
        raise type(self).raised

    def __iter__(self):
        type(self).calls += 1
        raise type(self).raised

    def __len__(self) -> int:
        type(self).calls += 1
        raise type(self).raised


@pytest.mark.parametrize(
    "raised",
    [RuntimeError("hostile equality ran"), KeyboardInterrupt("hostile equality ran")],
)
@pytest.mark.parametrize(
    ("root_name", "authored_key"),
    [
        ("manifest", "components"),
        ("manifest", "composites"),
        ("census", "schema"),
        ("census", "artifact_schema"),
        *(("projections", component) for component in COMPONENTS),
    ],
)
def test_hostile_root_keys_are_rejected_without_equality_dispatch(
    root_name: str, authored_key: str, raised: BaseException
) -> None:
    identity = _identity()
    manifest, census, projections = _fixture()
    roots = {"manifest": manifest, "census": census, "projections": projections}
    root = roots[root_name]
    if root_name == "census" and authored_key == "artifact_schema":
        root.pop("manifest_schema")
        value = "operatebench.b1_card_contract.v1"
    else:
        value = root.pop(authored_key)
    hostile_key = _HostileText(authored_key)
    root[hostile_key] = value
    _HostileText.calls = 0
    _HostileText.raised = raised

    with pytest.raises(
        identity.IdentityValidationError, match="canonical Identity v1 JSON"
    ) as caught:
        identity.validate_identity_manifest(manifest, census, projections)
    assert _HostileText.calls == 0
    assert type(caught.value.__cause__) is TypeError


@pytest.mark.parametrize(
    "raised",
    [RuntimeError("hostile equality ran"), KeyboardInterrupt("hostile equality ran")],
)
@pytest.mark.parametrize("field", ["schema", "artifact_schema"])
def test_hostile_census_discriminator_values_are_rejected_without_equality_dispatch(
    field: str, raised: BaseException
) -> None:
    identity = _identity()
    manifest, census, projections = _fixture()
    if field == "artifact_schema":
        census.pop("manifest_schema")
        census[field] = _HostileText("operatebench.b1_card_contract.v1")
    else:
        census[field] = _HostileText("operatebench.identity_field_census.v1")
    _HostileText.calls = 0
    _HostileText.raised = raised

    with pytest.raises(
        identity.IdentityValidationError, match="canonical Identity v1 JSON"
    ):
        identity.validate_identity_manifest(manifest, census, projections)
    assert _HostileText.calls == 0


@pytest.mark.parametrize(
    ("container_type", "raised"),
    [
        (_HostileList, RuntimeError("hostile container protocol ran")),
        (_HostileList, KeyboardInterrupt("hostile container protocol ran")),
        (_HostileDict, RuntimeError("hostile container protocol ran")),
        (_HostileDict, KeyboardInterrupt("hostile container protocol ran")),
    ],
)
def test_hostile_nested_containers_are_rejected_before_their_protocols_run(
    container_type: type[_HostileList] | type[_HostileDict], raised: BaseException
) -> None:
    identity = _identity()
    manifest, census, projections = _fixture()
    nested = container_type()
    container_type.calls = 0
    container_type.raised = raised
    projection: dict[str, Any] = projections[COMPONENTS[0]]
    projection[f"/fixture/{COMPONENTS[0]}"] = nested

    with pytest.raises(
        identity.IdentityValidationError, match="canonical Identity v1 JSON"
    ):
        identity.validate_identity_manifest(manifest, census, projections)
    assert container_type.calls == 0


def test_root_cardinality_refusal_precedes_key_dispatch_and_traversal() -> None:
    identity = _identity()
    manifest, census, projections = _fixture()
    hostile_key = _HostileText("components")
    manifest.pop("components")
    manifest[hostile_key] = {}
    manifest["extra"] = [object()] * (identity.MAX_STRICT_JSON_CONTAINER_ITEMS + 1)
    _HostileText.calls = 0
    _HostileText.raised = KeyboardInterrupt("hostile equality ran")

    with pytest.raises(
        identity.IdentityValidationError, match="identity manifest fields"
    ):
        identity.validate_identity_manifest(manifest, census, projections)
    assert _HostileText.calls == 0


class _TracebackAttacker:
    calls = 0

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("attacker representation ran")


_TRUSTED_EXCEPTION_TEXT_LIMIT = 512


def _assert_manifest_exception_graph_is_detached(
    exc: BaseException,
    *,
    raw_roots: tuple[object, object, object],
    forbidden: tuple[object, ...],
    max_local_size: int,
) -> None:
    """Recursively inspect exception frames without dispatching attacker protocols."""

    forbidden_ids = {id(value) for value in (*raw_roots, *forbidden)}
    pending_exceptions = [exc]
    seen_exceptions: set[int] = set()

    def inspect_local(root: object, location: str) -> None:
        pending = [root]
        seen: set[int] = set()
        while pending:
            value = pending.pop()
            assert id(value) not in forbidden_ids, location
            if id(value) in seen:
                continue
            seen.add(id(value))
            if type(value) in (str, bytes, bytearray):
                assert len(value) <= max_local_size
            elif type(value) is dict:
                pending.extend(value.keys())
                pending.extend(value.values())
            elif type(value) in (list, tuple, set, frozenset):
                pending.extend(value)
            elif type(value) is MappingProxyType:
                pending.extend(value.keys())
                pending.extend(value.values())

    while pending_exceptions:
        current = pending_exceptions.pop()
        if id(current) in seen_exceptions:
            continue
        seen_exceptions.add(id(current))
        assert type(current.args) is tuple
        assert all(
            type(argument) in (str, int, type(None))
            and (
                type(argument) is not str
                or len(argument) <= _TRUSTED_EXCEPTION_TEXT_LIMIT
            )
            for argument in current.args
        )
        traceback = current.__traceback__
        while traceback is not None:
            frame_name = traceback.tb_frame.f_code.co_name
            for local_name, local in traceback.tb_frame.f_locals.items():
                inspect_local(local, f"{frame_name}.{local_name}")
            traceback = traceback.tb_next
        if current.__cause__ is not None:
            pending_exceptions.append(current.__cause__)
        if current.__context__ is not None:
            pending_exceptions.append(current.__context__)


def _capture_manifest_failure(
    identity: Any,
    manifest: dict[str, Any],
    census: dict[str, Any],
    projections: dict[str, Any],
) -> BaseException:
    try:
        try:
            identity.validate_identity_manifest(manifest, census, projections)
        except BaseException as caught:
            return caught
        raise AssertionError("hostile manifest unexpectedly validated")
    finally:
        manifest = {}
        census = {}
        projections = {}


def test_all_manifest_failures_detach_raw_caller_roots_from_exception_graphs() -> None:
    identity = _identity()
    huge_markers = tuple(bytearray(32 * 1024 * 1024) for _ in range(3))
    attackers = tuple(_TracebackAttacker() for _ in range(3))
    cases: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []

    # First-root normalization rejection must not retain either unvisited later root.
    manifest, census, projections = _fixture()
    manifest["schema"] = attackers[0]
    census["entries"] = huge_markers[1]
    projections[COMPONENTS[-1]][f"/fixture/{COMPONENTS[-1]}"] = huge_markers[2]
    cases.append((manifest, census, projections))

    # The first root is detached before the second rejects; the third stays unvisited.
    manifest, census, projections = _fixture()
    census["schema"] = attackers[1]
    projections[COMPONENTS[-1]]["/later-huge"] = huge_markers[2]
    cases.append((manifest, census, projections))

    # A third-root rejection follows successful detachment of the first two roots.
    manifest, census, projections = _fixture()
    projections[COMPONENTS[-1]]["/reject"] = attackers[2]
    cases.append((manifest, census, projections))

    # Root cardinality refusal precedes traversal but still clears all caller roots.
    manifest, census, projections = _fixture()
    manifest["extra"] = huge_markers[0]
    census["later_huge"] = huge_markers[1]
    projections[COMPONENTS[-1]]["/later-huge"] = huge_markers[2]
    cases.append((manifest, census, projections))

    # Nested hostile, oversized, and cyclic values exercise normalizer frame cleanup.
    manifest, census, projections = _fixture()
    cycle: list[object] = []
    cycle.append(cycle)
    projections[COMPONENTS[-1]]["/nested"] = [
        attackers[2],
        huge_markers[2],
        cycle,
    ]
    cases.append((manifest, census, projections))

    # A semantic failure after all roots normalize may retain only bounded copies.
    manifest, census, projections = _fixture()
    census["owners"] = list(reversed(COMPONENTS))
    cases.append((manifest, census, projections))

    _TracebackAttacker.calls = 0
    started = __import__("time").perf_counter()
    for raw_manifest, raw_census, raw_projections in cases:
        caught = _capture_manifest_failure(
            identity, raw_manifest, raw_census, raw_projections
        )
        assert type(caught) is identity.IdentityValidationError
        _assert_manifest_exception_graph_is_detached(
            caught,
            raw_roots=(raw_manifest, raw_census, raw_projections),
            forbidden=(*huge_markers, *attackers),
            max_local_size=identity.MAX_CANONICAL_UTF8_PAYLOAD_BYTES,
        )
    assert __import__("time").perf_counter() - started < 2.0
    assert _TracebackAttacker.calls == 0


@pytest.mark.parametrize("argument", [(), UserDict(), ChainMap(), MappingProxyType({})])
def test_rejects_nonexact_root_containers(argument: object) -> None:
    identity = _identity()
    manifest, census, projections = _fixture()
    for args in (
        (argument, census, projections),
        (manifest, argument, projections),
        (manifest, census, argument),
    ):
        with pytest.raises(identity.IdentityValidationError, match="ordinary object"):
            identity.validate_identity_manifest(*args)


def test_census_and_projection_ownership_join_is_exact_and_ordered() -> None:
    identity = _identity()
    manifest, census, projections = _fixture()
    cases = []
    wrong_owners = deepcopy(census)
    wrong_owners["owners"] = list(reversed(COMPONENTS))
    cases.append((wrong_owners, projections, "owners"))
    unsorted = deepcopy(census)
    unsorted["entries"][0], unsorted["entries"][1] = (
        unsorted["entries"][1],
        unsorted["entries"][0],
    )
    cases.append((unsorted, projections, "sorted"))
    duplicate = deepcopy(census)
    duplicate["entries"].insert(1, deepcopy(duplicate["entries"][0]))
    cases.append((duplicate, projections, "unique"))
    missing_census = deepcopy(census)
    missing_census["entries"].pop(0)
    cases.append((missing_census, projections, "ownership"))
    extra_projection = deepcopy(projections)
    extra_projection[COMPONENTS[0]]["/extra"] = 1
    cases.append((census, extra_projection, "ownership"))
    missing_projection = deepcopy(projections)
    missing_projection[COMPONENTS[0]].clear()
    cases.append((census, missing_projection, "ownership"))
    for candidate_census, candidate_projections, message in cases:
        with pytest.raises(identity.IdentityValidationError, match=message):
            identity.validate_identity_manifest(
                manifest, candidate_census, candidate_projections
            )


def test_rejects_projection_mismatch_with_refreshed_attacker_digests() -> None:
    identity = _identity()
    manifest, census, projections = _fixture()
    attacker_projections = deepcopy(projections)
    attacker_projections[COMPONENTS[0]][f"/fixture/{COMPONENTS[0]}"] = 999
    _refresh_downstream(manifest, attacker_projections)
    with pytest.raises(identity.IdentityValidationError, match="projection ownership"):
        identity.validate_identity_manifest(manifest, census, projections)


@pytest.mark.parametrize(
    ("serialized_value", "independent_value"),
    [
        (False, 0),
        (0, False),
        ({"nested": [False]}, {"nested": [0]}),
        ({"nested": {"value": False}}, {"nested": {"value": 0}}),
    ],
)
def test_projection_values_require_exact_recursive_json_types(
    serialized_value: Any, independent_value: Any
) -> None:
    identity = _identity()
    manifest, census, projections = _fixture()
    path = f"/fixture/{COMPONENTS[0]}"
    manifest["components"][COMPONENTS[0]]["owned_projection"][path] = serialized_value
    projections[COMPONENTS[0]][path] = independent_value

    with pytest.raises(identity.IdentityValidationError, match="projection ownership"):
        identity.validate_identity_manifest(manifest, census, projections)


def test_bool_projection_mismatch_with_refreshed_attacker_digests_is_rejected() -> None:
    identity = _identity()
    manifest, census, projections = _fixture()
    attacker_projections = deepcopy(projections)
    attacker_projections[COMPONENTS[0]][f"/fixture/{COMPONENTS[0]}"] = False
    _refresh_downstream(manifest, attacker_projections)

    with pytest.raises(identity.IdentityValidationError, match="projection ownership"):
        identity.validate_identity_manifest(manifest, census, projections)


@pytest.mark.parametrize(
    "contract_path",
    [
        ("schema_version",),
        ("requirement_kinds", "exact_digest"),
        ("decision_predicates", "can_runtime_replay"),
    ],
)
def test_compatibility_contract_versions_reject_bool_substitution(
    contract_path: tuple[str, ...],
) -> None:
    identity = _identity()
    manifest, census, projections = _fixture()
    target = manifest["compatibility_contract"]
    for key in contract_path[:-1]:
        target = target[key]
    target[contract_path[-1]] = True

    with pytest.raises(identity.IdentityValidationError, match="compatibility contract"):
        identity.validate_identity_manifest(manifest, census, projections)


@pytest.mark.parametrize(
    ("record_group", "record_name", "message"),
    [
        ("components", COMPONENTS[0], "component card_schema_content"),
        ("composites", "runtime_bundle", "composite runtime_bundle"),
    ],
)
def test_serialized_identity_versions_reject_bool_substitution(
    record_group: str, record_name: str, message: str
) -> None:
    identity = _identity()
    manifest, census, projections = _fixture()
    manifest[record_group][record_name]["schema_version"] = True

    with pytest.raises(identity.IdentityValidationError, match=message):
        identity.validate_identity_manifest(manifest, census, projections)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda manifest: manifest.update(extra=True), "identity manifest fields"),
        (lambda manifest: manifest["components"].pop(COMPONENTS[-1]), "components"),
        (lambda manifest: manifest["composites"].pop("runtime_bundle"), "composites"),
        (
            lambda manifest: manifest["components"][COMPONENTS[0]].update(
                domain="operatebench.digest.component.wrong.v1"
            ),
            "complete valid identity",
        ),
        (
            lambda manifest: manifest["composites"]["runtime_bundle"].update(
                bundle_digest_sha256="0" * 64
            ),
            "composite runtime_bundle",
        ),
        (
            lambda manifest: manifest.update(manifest_digest_sha256="0" * 64),
            "manifest digest",
        ),
    ],
)
def test_rejects_manifest_closure_and_stale_records(mutate, message: str) -> None:
    identity = _identity()
    manifest, census, projections = _fixture()
    mutate(manifest)
    with pytest.raises(identity.IdentityValidationError, match=message):
        identity.validate_identity_manifest(manifest, census, projections)


def test_requirement_satisfaction_is_deterministic_without_decisions() -> None:
    identity = _identity()
    for compatible, message in (
        ({"kind": "exact_digest", "content_digest_sha256": "0" * 64}, "digest"),
        (
            {
                "kind": "closed_schema_version_range",
                "minimum_schema_version": 2,
                "maximum_schema_version": 3,
            },
            "range",
        ),
        (
            {
                "kind": "named_predicate",
                "predicate": "future.rule",
                "predicate_version": 1,
            },
            "named predicate",
        ),
    ):
        manifest, census, projections = _fixture()
        manifest["components"]["operation_core_content"]["requires"][0]["compatible"] = (
            compatible
        )
        _refresh_downstream(manifest, projections)
        with pytest.raises(identity.IdentityValidationError, match=message):
            identity.validate_identity_manifest(manifest, census, projections)


def test_projection_census_schema_is_pinned_strict_and_byte_identical() -> None:
    identity = _identity()
    jsonschema = pytest.importorskip("jsonschema")
    name = "identity-manifest-projection-census-v1.schema.json"
    docs = Path(__file__).parents[1] / "docs" / "schemas" / name
    packaged = files("operatebench.resources.identity").joinpath(name)
    assert docs.read_bytes() == packaged.read_bytes()
    assert (
        identity._RESOURCE_SHA256[name]
        == __import__("hashlib").sha256(docs.read_bytes()).hexdigest()
    )
    schema = identity.load_identity_manifest_projection_census_schema()
    manifest_schema = identity.load_identity_schema()
    assert (
        schema["$defs"]["fieldPath"]["pattern"]
        == manifest_schema["$defs"]["componentIdentity"]["properties"]["owned_fields"][
            "items"
        ]["pattern"]
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    manifest, census, _ = _fixture()
    jsonschema.Draft202012Validator(schema).validate(census)
    assert census["manifest_schema"] == manifest["schema"]
    for definition in schema["$defs"].values():
        if definition.get("type") == "object":
            assert definition.get("additionalProperties") is False


def test_direct_module_import_and_validation_stay_phase_isolated() -> None:
    identity = _identity()
    source = Path(identity.__file__).read_text(encoding="utf-8")
    imports = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert {name for name in imports if name.startswith("operatebench.")} == {
        "operatebench.jsonsafe"
    }
    code = """
import json, sys
from operatebench.identity import validate_identity_manifest
blocked = ('cards', 'artifact', 'runtime', 'evaluator', 'provider', 'runner', 'version')
phase_modules = {
    name for name in sys.modules
    if name.startswith('operatebench.') and any(part in name for part in blocked)
}
# Importing the package initializer itself accounts for these two known modules.
assert phase_modules <= {'operatebench.cards', 'operatebench.version'}
print(json.dumps(True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) is True


def test_shipped_docs_state_current_slice5_and_later_b2_ownership() -> None:
    root = Path(__file__).parents[1]
    rfc = root / "docs" / "IDENTITY_REPLAY_RFC.md"
    architecture = root / "docs" / "ARCHITECTURE_RFC.md"
    versioning = root / "docs" / "VERSIONING.md"
    module = root / "src" / "operatebench" / "identity.py"
    prose = " ".join(rfc.read_text(encoding="utf-8").split())
    architecture_prose = " ".join(architecture.read_text(encoding="utf-8").split())
    versioning_prose = " ".join(versioning.read_text(encoding="utf-8").split())
    module_source = module.read_text(encoding="utf-8")
    module_tree = ast.parse(module_source)

    required_contracts = (
        "`validate_identity_manifest` is the implemented B2 Slice 3 trust boundary.",
        "`recompute_manifest_digest` is a hash-only helper: it does not validate "
        "component/composite closure or make a manifest authoritative; full validation "
        "is available through `validate_identity_manifest`.",
        "`evaluate_component_requirements(requirements, resolved_components)` is the "
        "implemented B2 Slice 4 primitive.",
        "The empty requirement list is only the true empty conjunction",
        "does not authorize replay, regrading, analysis recomputation, comparison, "
        "or pooling.",
        "Predicate vocabulary is not executable predicate semantics.",
        "Slice 4 consumes the unchanged vocabulary-only compatibility registry; no "
        "mirrored registry resource or packaged-resource integrity pin changed;",
        "it does not change component, composite, compatibility-contract, or golden "
        "manifest bytes/digests.",
        "The `identity.py` module itself has no direct import of B1 cards or any "
        "Artifact, runtime, evaluator, provider, or version module",
        "Normal Python package initialization may load the package's existing B1 "
        "top-level exports before the submodule",
    )
    for contract in required_contracts:
        assert contract in prose

    dash = "\N{EN DASH}"
    assert (
        f"B2 Identity Manifest Slices 1{dash}5 construction, validation, census, "
        "requirement evaluator, and identity diff implemented; final compatibility "
        "Decisions and domain adapters pending" in prose
    )
    assert (
        "Slice 5 implements `identity_diff` over exact `ValidatedIdentityManifest` "
        "inputs." in prose
    )
    assert (
        "`OWNED_MUTATION_CONTRACT_VERSION` is the exact integer `1`. `OwnedMutation` "
        "is a frozen, slotted dataclass" in prose
    )
    assert (
        "The diff invokes no requirement evaluator, compatibility Decision, replay, "
        "comparability, provider, or filesystem logic, has no JSON projection, and "
        "retains no input projection or requirement values." in prose
    )
    assert "Domain projection adapters and Artifact 9 remain later work." in prose

    assert f"Phase B2 Identity Manifest Slices 1{dash}5" in architecture_prose
    assert (
        f"Slices 1{dash}5 implement identity construction, recomputation, and "
        "projection-backed manifest validation" in architecture_prose
    )
    assert (
        "The five final compatibility Decisions remain specified-only"
        in architecture_prose
    )
    assert (
        "Slice 5 adds the frozen `OwnedMutation` and structurally revalidated, "
        "owner-ordered `identity_diff`; domain projection adapters remain pending."
        in architecture_prose
    )
    assert (
        "B3 owns the semantics and canonical inputs of both `can_runtime_replay` and "
        "`can_evaluator_regrade`" in architecture_prose
    )
    assert (
        "B4 owns only the Artifact 9 adapters/serialization and full census; it cannot "
        "redefine replay or regrade semantics or canonical inputs." in architecture_prose
    )
    assert (
        "The Slice 5 identity diff now reports detached owner rows for concrete "
        "projection and direct requirement-metadata changes, with actual reverse-"
        "composite consequences; it makes no replay or comparability decision."
        in versioning_prose
    )
    assert (
        "Final compatibility Decisions, domain projection adapters, and Artifact 9 "
        "support remain pending." in versioning_prose
    )

    assert ast.get_docstring(module_tree, clean=False) == (
        "Pure-offline Identity Manifest v1 identities and trust-boundary validation.\n\n"
        "This module provides construction, recomputation, projection-backed manifest\n"
        "validation, and bounded component-requirement evaluation.  It does not import\n"
        "replay, Artifact, evaluator, provider, result, or analysis logic.\n"
    )
    public_nodes = {
        node.name: type(node)
        for node in module_tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    assert public_nodes["OwnedMutation"] is ast.ClassDef
    assert public_nodes["identity_diff"] is ast.FunctionDef
    assert "CompatibilityDecisionV1" not in public_nodes
    assert "evaluate_compatibility" not in public_nodes

    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    sdist_include = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    assert "/docs" in sdist_include
    compatibility_registry = root / "docs/schemas/identity-compatibility-v1.registry.json"
    assert (
        __import__("hashlib").sha256(compatibility_registry.read_bytes()).hexdigest()
        == "bcb84fdac5a545e34f002113eed1ac7231d7387c0cd61a6043acc6895282ba44"
    )
    assert (
        compatibility_registry.read_bytes()
        == files("operatebench.resources.identity")
        .joinpath(compatibility_registry.name)
        .read_bytes()
    )
    for unclaimed_resource in (
        "identity-diff-v1.schema.json",
        "owned-mutation-v1.schema.json",
    ):
        assert not (root / "docs" / "schemas" / unclaimed_resource).exists()
        assert (
            not files("operatebench.resources.identity")
            .joinpath(unclaimed_resource)
            .is_file()
        )

    stale_claims = (
        "A future manifest validator will establish trust",
        "Full manifest validation remains Slice 3 work.",
        "B2 Slice 3 owns the future three-argument manifest validator, identity diff",
        f"Slices 1{dash}2 implemented, later slices pending",
        f"Slices 1{dash}4 construction, validation, census, and requirement evaluator "
        "implemented; final compatibility Decisions, diff, and domain adapters pending",
        "all five registered decision predicates",
        "CompatibilityDecisionV1",
        "evaluate_compatibility(",
        "identity-diff-v1.schema.json",
        "owned-mutation-v1.schema.json",
    )
    for claim in stale_claims:
        assert claim not in prose
    assert f"Phase B2 Identity Manifest Slices 1{dash}2" not in architecture_prose
    assert f"Phase B2 Identity Manifest Slices 1{dash}4" not in architecture_prose
    assert "This slice constructs identities only." not in architecture_prose
    assert "It does not validate a complete manifest" not in architecture_prose
