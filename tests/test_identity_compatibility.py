# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import zipfile
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from types import TracebackType
from typing import Any

import pytest

import operatebench.identity as identity
from operatebench.identity import (
    IdentityValidationError,
    RequirementEvaluationV1,
    evaluate_component_requirements,
    recompute_component,
)


def _component():
    return recompute_component("card_schema_content", {"/schema": 1}, [])


def _requirement(compatible: dict[str, Any]) -> dict[str, Any]:
    return {"component": "card_schema_content", "compatible": compatible}


def _requirement_for(component: Any) -> dict[str, Any]:
    return {
        "component": component,
        "compatible": {
            "kind": "exact_digest",
            "content_digest_sha256": "0" * 64,
        },
    }


def _exact(component=None, digest: str | None = None) -> dict[str, Any]:
    component = _component() if component is None else component
    return _requirement(
        {
            "kind": "exact_digest",
            "content_digest_sha256": digest or component.content_digest_sha256,
        }
    )


def _exception_graph(exc: BaseException):
    pending = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


def _tracebacks(exc: BaseException):
    for current in _exception_graph(exc):
        traceback: TracebackType | None = current.__traceback__
        while traceback is not None:
            yield traceback
            traceback = traceback.tb_next


def _capture_resolved_components_failure(
    components: dict[Any, Any],
) -> IdentityValidationError:
    try:
        evaluate_component_requirements([], components)
    except IdentityValidationError as exc:
        return exc
    finally:
        components = {}
    raise AssertionError("malformed resolved components were accepted")


def test_false_generic_decision_surface_is_absent() -> None:
    assert not hasattr(identity, "CompatibilityDecisionV1")
    assert not hasattr(identity, "evaluate_compatibility")
    assert "CompatibilityDecisionV1" not in identity.__all__
    assert "evaluate_compatibility" not in identity.__all__
    for name in (
        "can_runtime_replay",
        "can_evaluator_regrade",
        "can_analysis_recompute",
        "can_compare",
        "can_pool",
    ):
        assert not hasattr(identity, name)


def test_empty_requirements_are_only_the_true_empty_conjunction() -> None:
    evaluation = evaluate_component_requirements([], {})
    assert evaluation == RequirementEvaluationV1(1, True, ())
    assert [field.name for field in fields(evaluation)] == [
        "requirement_contract_version",
        "satisfied",
        "reasons",
    ]
    assert not hasattr(evaluation, "allowed")
    assert not hasattr(evaluation, "predicate")
    assert not hasattr(evaluation, "reason_codes")
    with pytest.raises(FrozenInstanceError):
        evaluation.satisfied = False


def test_exact_digest_and_inclusive_closed_range_endpoints() -> None:
    component = _component()
    resolved = {component.name: component}
    assert evaluate_component_requirements([_exact(component)], resolved).satisfied

    mismatch = evaluate_component_requirements([_exact(component, "0" * 64)], resolved)
    assert mismatch == RequirementEvaluationV1(1, False, ("content_digest_mismatch",))

    for minimum, maximum in ((1, 1), (1.0, 1.0)):
        requirement = _requirement(
            {
                "kind": "closed_schema_version_range",
                "minimum_schema_version": minimum,
                "maximum_schema_version": maximum,
            }
        )
        assert evaluate_component_requirements([requirement], resolved).satisfied

    for minimum, maximum in ((2, 3), (1, 0)):
        requirement = _requirement(
            {
                "kind": "closed_schema_version_range",
                "minimum_schema_version": minimum,
                "maximum_schema_version": maximum,
            }
        )
        if minimum > maximum:
            with pytest.raises(IdentityValidationError):
                evaluate_component_requirements([requirement], resolved)
        else:
            assert evaluate_component_requirements([requirement], resolved).reasons == (
                "schema_version_out_of_range",
            )


def test_named_predicate_allowlist_uses_stable_immutable_derived_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity._clear_resource_caches()
    monkeypatch.setattr(
        identity,
        "load_compatibility_registry",
        lambda: pytest.fail("public mutable registry loader called during evaluation"),
    )
    monkeypatch.setattr(
        identity,
        "_plain",
        lambda _value: pytest.fail("frozen registry deep-copied during evaluation"),
    )

    first = identity._named_requirement_predicate_allowlist()
    assert first == frozenset()
    assert isinstance(first, frozenset)

    named = _requirement(
        {
            "kind": "named_predicate",
            "predicate": "operatebench.future.rule",
            "predicate_version": 1,
        }
    )
    for _ in range(3):
        assert evaluate_component_requirements([named], {}).reasons == (
            "unknown_named_requirement_predicate",
            "missing_component",
        )
        assert identity._named_requirement_predicate_allowlist() is first


def test_clearing_resource_caches_rebuilds_named_predicate_allowlist() -> None:
    identity._clear_resource_caches()
    first = identity._named_requirement_predicate_allowlist()

    identity._clear_resource_caches()

    second = identity._named_requirement_predicate_allowlist()
    assert second == first == frozenset()
    assert second is not first


def test_resource_caches_clear_derived_state_before_validated_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleared: list[str] = []

    class CacheStub:
        def __init__(self, name: str) -> None:
            self.name = name

        def cache_clear(self) -> None:
            cleared.append(self.name)

    monkeypatch.setattr(
        identity, "_named_requirement_predicate_allowlist", CacheStub("derived")
    )
    monkeypatch.setattr(identity, "_cached_resource", CacheStub("resource"))

    identity._clear_resource_caches()

    assert cleared == ["derived", "resource"]


def test_missing_named_and_mismatches_are_ordered_and_deduplicated() -> None:
    component = _component()
    named = _requirement(
        {
            "kind": "named_predicate",
            "predicate": "operatebench.future.rule",
            "predicate_version": 1,
        }
    )
    out_of_range = _requirement(
        {
            "kind": "closed_schema_version_range",
            "minimum_schema_version": 2,
            "maximum_schema_version": 3,
        }
    )
    evaluation = evaluate_component_requirements(
        [out_of_range, _exact(component, "0" * 64), named, named],
        {component.name: component},
    )
    assert evaluation.reasons == (
        "unknown_named_requirement_predicate",
        "content_digest_mismatch",
        "schema_version_out_of_range",
    )
    assert evaluate_component_requirements([_exact(component)], {}).reasons == (
        "missing_component",
    )


@pytest.mark.parametrize(
    "component",
    [
        "future_component",
        "operatebench.future.component",
        f"future.{('namespace.' * 100)}component",
        "card_schema_contents",
        "runtime_bundle",
        "experiment_bundle",
    ],
)
def test_unregistered_component_requirements_are_malformed(component: str) -> None:
    with pytest.raises(IdentityValidationError, match=r"^unknown component requirement$"):
        evaluate_component_requirements([_requirement_for(component)], {})


@pytest.mark.parametrize("component", ["", "root", ".", "_"])
def test_empty_and_root_like_component_requirements_are_malformed(component: str) -> None:
    with pytest.raises(IdentityValidationError):
        evaluate_component_requirements([_requirement_for(component)], {})


def test_registered_missing_null_and_complete_component_semantics_are_distinct() -> None:
    component = _component()
    requirement = _exact(component)

    assert evaluate_component_requirements([requirement], {}) == RequirementEvaluationV1(
        1, False, ("missing_component",)
    )
    malformed_components: Any = {component.name: None}
    with pytest.raises(IdentityValidationError):
        evaluate_component_requirements([requirement], malformed_components)
    assert evaluate_component_requirements(
        [requirement], {component.name: component}
    ) == RequirementEvaluationV1(1, True, ())


def test_hostile_unknown_component_string_is_not_dispatched_or_retained() -> None:
    calls = 0

    class HostileString(str):
        def __str__(self):
            nonlocal calls
            calls += 1
            raise AssertionError("str called")

        def __repr__(self):
            nonlocal calls
            calls += 1
            raise AssertionError("repr called")

        def __hash__(self):
            nonlocal calls
            calls += 1
            raise AssertionError("hash called")

        def __eq__(self, other: object) -> bool:
            nonlocal calls
            calls += 1
            raise AssertionError("eq called")

    hostile = HostileString("future_component")
    requirements = [_requirement_for(hostile)]
    with pytest.raises(IdentityValidationError) as caught:
        evaluate_component_requirements(requirements, {})

    for traceback in _tracebacks(caught.value):
        if traceback.tb_frame.f_globals.get("__name__") == "operatebench.identity":
            locals_ = traceback.tb_frame.f_locals
            assert all(value is not hostile for value in locals_.values())
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert calls == 0


@pytest.mark.parametrize(
    "requirements",
    [
        (),
        [None],
        [{}],
        [{"component": "card_schema_content", "compatible": {}, "extra": 1}],
        [_requirement({"kind": "future_kind"})],
        [_requirement({"kind": "exact_digest"})],
        [_requirement({"kind": "exact_digest", "content_digest_sha256": "bad"})],
        [_requirement({"kind": "exact_digest", "content_digest_sha256": None})],
        [
            _requirement(
                {
                    "kind": "closed_schema_version_range",
                    "minimum_schema_version": 2,
                    "maximum_schema_version": 1,
                }
            )
        ],
        [
            _requirement(
                {
                    "kind": "named_predicate",
                    "predicate": "not_namespaced",
                    "predicate_version": 1,
                }
            )
        ],
    ],
)
def test_malformed_requirement_shapes_raise(requirements: Any) -> None:
    with pytest.raises(IdentityValidationError):
        evaluate_component_requirements(requirements, {})


@pytest.mark.parametrize("components", [(), {1: None}, {"card_schema_content": None}])
def test_malformed_resolved_component_shapes_raise(components: Any) -> None:
    with pytest.raises(IdentityValidationError):
        evaluate_component_requirements([], components)


@pytest.mark.parametrize(
    "name",
    [
        "future_component",
        "card_schema_contents",
        "operatebench.future.component",
        "runtime_bundle",
        "root",
        "",
        "attacker-marker-" + "x" * (1024 * 1024),
    ],
    ids=["unknown", "typo", "dotted", "composite", "root", "empty", "one-mib"],
)
def test_resolved_component_names_are_closed_before_record_validation(name: str) -> None:
    with pytest.raises(
        IdentityValidationError,
        match=(
            r"^resolved component name is outside the closed Identity v1 component set$"
        ),
    ) as caught:
        evaluate_component_requirements([], {name: object()})  # type: ignore[dict-item]
    assert "attacker-marker" not in str(caught.value)
    assert len(str(caught.value)) < 100


def test_oversized_resolved_component_name_is_not_copied_or_retained() -> None:
    name = "attacker-marker-" + "x" * (8 * 1024 * 1024)
    components = {name: object()}

    caught = _capture_resolved_components_failure(components)

    assert caught.args == (
        "resolved component name is outside the closed Identity v1 component set",
    )
    assert caught.__cause__ is None
    assert caught.__context__ is None
    for traceback in _tracebacks(caught):
        if traceback.tb_frame.f_globals.get("__name__") == "operatebench.identity":
            assert all(
                value is not name for value in traceback.tb_frame.f_locals.values()
            )
            assert all(
                value is not components for value in traceback.tb_frame.f_locals.values()
            )


def test_hostile_resolved_component_name_is_rejected_without_hooks() -> None:
    calls: list[str] = []

    class HostileString(str):
        def __str__(self) -> str:
            calls.append("str")
            raise AssertionError("str called")

        def __repr__(self) -> str:
            calls.append("repr")
            raise AssertionError("repr called")

        def __hash__(self) -> int:
            calls.append("hash")
            return str.__hash__(self)

        def __eq__(self, other: object) -> bool:
            calls.append("eq")
            return str.__eq__(self, other)

    name = HostileString("card_schema_content")
    components = {name: object()}
    calls.clear()

    caught = _capture_resolved_components_failure(components)

    assert caught.args == ("resolved component names must be exact strings",)
    assert calls == []
    for traceback in _tracebacks(caught):
        if traceback.tb_frame.f_globals.get("__name__") == "operatebench.identity":
            assert all(
                value is not name for value in traceback.tb_frame.f_locals.values()
            )


def test_forged_and_wrong_name_component_records_raise() -> None:
    component = _component()
    forged = replace(component, content_digest_sha256="0" * 64)
    wrong_name = {"build_provenance_content": component}
    with pytest.raises(IdentityValidationError, match="complete valid identity"):
        evaluate_component_requirements([_exact(component)], {component.name: forged})
    with pytest.raises(IdentityValidationError, match="wrong identity name"):
        evaluate_component_requirements([], wrong_name)


def test_aggregate_bounds_and_hostile_subclasses_fail_without_hooks() -> None:
    calls = 0

    class HostileDict(dict):
        def items(self):
            nonlocal calls
            calls += 1
            raise AssertionError("hook called")

        def __repr__(self):
            nonlocal calls
            calls += 1
            raise AssertionError("repr called")

    with pytest.raises(IdentityValidationError):
        evaluate_component_requirements([HostileDict()], {})
    with pytest.raises(IdentityValidationError, match="resource limits"):
        evaluate_component_requirements(
            [
                _requirement(
                    {
                        "kind": "exact_digest",
                        "content_digest_sha256": "x" * (5 * 1024 * 1024),
                    }
                )
            ],
            {},
        )
    with pytest.raises(IdentityValidationError, match="resource limits"):
        evaluate_component_requirements([{}] * 4097, {})
    assert calls == 0


def test_failure_graph_drops_raw_roots_and_oversized_markers() -> None:
    marker = bytearray(b"z" * (8 * 1024 * 1024))
    requirements: list[dict[str, Any]] = [{"attacker": marker}]
    components: dict[str, Any] = {}
    with pytest.raises(IdentityValidationError) as caught:
        evaluate_component_requirements(requirements, components)

    for traceback in _tracebacks(caught.value):
        if traceback.tb_frame.f_globals.get("__name__") == "operatebench.identity":
            locals_ = traceback.tb_frame.f_locals
            assert all(value is not requirements for value in locals_.values())
            assert all(value is not components for value in locals_.values())
            assert all(value is not marker for value in locals_.values())
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_unknown_component_failure_drops_raw_name_and_oversized_sibling() -> None:
    unknown = "future_component"
    marker = "z" * (8 * 1024 * 1024)
    requirements = [_requirement_for(unknown), {"attacker": marker}]
    with pytest.raises(IdentityValidationError) as caught:
        evaluate_component_requirements(requirements, {})

    for traceback in _tracebacks(caught.value):
        if traceback.tb_frame.f_globals.get("__name__") == "operatebench.identity":
            locals_ = traceback.tb_frame.f_locals
            assert all(value is not unknown for value in locals_.values())
            assert all(value is not marker for value in locals_.values())
            assert all(value is not requirements for value in locals_.values())
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_identity_module_keeps_later_owner_import_boundaries() -> None:
    source = Path("src/operatebench/identity.py").read_text(encoding="utf-8")
    imports = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported = {alias.name for node in imports for alias in node.names}
    forbidden = (
        "operatebench.artifact",
        "operatebench.runner",
        "operatebench.agents",
        "operatebench.providers",
        "operatebench.domains",
        "operatebench.core.evaluation",
        "operatebench.analysis",
        "operatebench.result",
    )
    assert not any(name.startswith(forbidden) for name in imported)
    dynamic_calls = {
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert dynamic_calls.isdisjoint({"eval", "exec"})


def test_registry_is_vocabulary_only_and_documentation_bytes_are_mirrored() -> None:
    packaged = Path(
        "src/operatebench/resources/identity/identity-compatibility-v1.registry.json"
    ).read_bytes()
    documented = Path("docs/schemas/identity-compatibility-v1.registry.json").read_bytes()
    assert packaged == documented
    registry = json.loads(packaged)
    assert registry["contract"]["decision_predicates"] == {
        "can_runtime_replay": 1,
        "can_evaluator_regrade": 1,
        "can_analysis_recompute": 1,
        "can_compare": 1,
        "can_pool": 1,
    }
    assert registry["contract"]["named_requirement_predicates"] == []
    assert registry["reason_codes"] == []
    assert identity.REQUIREMENT_EVALUATION_REASON_ORDER == (
        "unknown_named_requirement_predicate",
        "missing_component",
        "content_digest_mismatch",
        "schema_version_out_of_range",
    )

    digest = hashlib.sha256(packaged).hexdigest()
    assert digest == "bcb84fdac5a545e34f002113eed1ac7231d7387c0cd61a6043acc6895282ba44"
    source = Path("src/operatebench/identity.py").read_text(encoding="utf-8")
    assert f'"{digest}"' in source

    prose = " ".join(
        Path("docs/IDENTITY_REPLAY_RFC.md")
        .read_text(encoding="utf-8")
        .replace("`", "")
        .split()
    )
    assert (
        "Slice 4 consumes the unchanged vocabulary-only compatibility registry; "
        "no mirrored registry resource or packaged-resource integrity pin changed;"
        in prose
    )
    assert "six reason_codes" not in prose
    assert "Slice 4 changes the compatibility registry" not in prose


def test_registry_validator_accepts_only_empty_decision_reason_vocabulary() -> None:
    packaged = Path(
        "src/operatebench/resources/identity/identity-compatibility-v1.registry.json"
    ).read_text(encoding="utf-8")
    registry = json.loads(packaged)
    identity._validate_compatibility_registry(registry)

    registry["reason_codes"] = [
        "unknown_decision_predicate",
        "unsupported_requirement_kind",
        "unknown_named_predicate",
        "missing_dependency",
        "digest_mismatch",
        "schema_version_out_of_range",
    ]
    with pytest.raises(
        IdentityValidationError,
        match=r"^compatibility registry reason codes are not the closed v1 vocabulary$",
    ):
        identity._validate_compatibility_registry(registry)


def test_replay_predicate_ownership_is_consistent_across_rfcs() -> None:
    identity_rfc = " ".join(
        Path("docs/IDENTITY_REPLAY_RFC.md")
        .read_text(encoding="utf-8")
        .replace("`", "")
        .split()
    )
    architecture_rfc = " ".join(
        Path("docs/ARCHITECTURE_RFC.md")
        .read_text(encoding="utf-8")
        .replace("`", "")
        .split()
    )
    ownership = (
        "B3 owns the semantics and canonical inputs of both can_runtime_replay "
        "and can_evaluator_regrade"
    )
    assert ownership in identity_rfc
    assert ownership in architecture_rfc
    assert "B4 owns only the Artifact 9 adapters/serialization" in architecture_rfc
    assert "B4 for runtime replay" not in identity_rfc
    assert "B4 owns runtime replay" not in identity_rfc
    assert "B4 for runtime replay" not in architecture_rfc
    assert "B4 owns runtime replay" not in architecture_rfc


def test_built_wheel_exposes_strict_external_typing_surface(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--wheel", "--offline", "--out-dir", str(dist)],
        check=True,
        cwd=Path(__file__).parents[1],
    )
    wheel = next(dist.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        assert "operatebench/py.typed" in archive.namelist()
        assert "boundarybench/py.typed" in archive.namelist()

    environment = tmp_path / "consumer-env"
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(environment)], check=True
    )
    python = environment / "bin" / "python"
    clean_environment = os.environ.copy()
    clean_environment.pop("PYTHONPATH", None)
    clean_environment.pop("MYPYPATH", None)
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)],
        check=True,
    )
    provenance = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.util, pathlib; "
            "operate = importlib.util.find_spec('operatebench'); "
            "boundary = importlib.util.find_spec('boundarybench'); "
            "assert operate is not None and operate.origin is not None; "
            "assert boundary is not None and boundary.origin is not None; "
            "print(pathlib.Path(operate.origin).resolve()); "
            "print(pathlib.Path(boundary.origin).resolve())",
        ],
        check=True,
        capture_output=True,
        env=clean_environment,
        text=True,
    ).stdout.splitlines()
    assert len(provenance) == 2
    assert "site-packages/operatebench" in provenance[0]
    assert "site-packages/boundarybench" in provenance[1]
    assert all(Path(location).is_relative_to(environment) for location in provenance)

    runtime_probe = subprocess.run(
        [
            str(python),
            "-c",
            textwrap.dedent(
                """
                from operatebench.identity import (
                    IdentityValidationError,
                    evaluate_component_requirements,
                )

                def capture(components):
                    try:
                        evaluate_component_requirements([], components)
                    except IdentityValidationError as exc:
                        return exc
                    finally:
                        components = {}
                    raise AssertionError("malformed resolved components were accepted")

                requirement = {
                    "component": "future_component",
                    "compatible": {
                        "kind": "exact_digest",
                        "content_digest_sha256": "0" * 64,
                    },
                }
                try:
                    evaluate_component_requirements([requirement], {})
                except IdentityValidationError:
                    pass
                else:
                    raise AssertionError("unknown component was treated as missing")

                name = "installed-attacker-marker-" + "x" * (8 * 1024 * 1024)
                components = {name: object()}
                exc = capture(components)
                expected = (
                    "resolved component name is outside the closed Identity v1 "
                    "component set"
                )
                assert exc.args == (expected,)
                assert exc.__cause__ is None
                assert exc.__context__ is None
                traceback = exc.__traceback__
                while traceback is not None:
                    if (
                        traceback.tb_frame.f_globals.get("__name__")
                        == "operatebench.identity"
                    ):
                        assert all(
                            value is not name
                            for value in traceback.tb_frame.f_locals.values()
                        )
                        assert all(
                            value is not components
                            for value in traceback.tb_frame.f_locals.values()
                        )
                    traceback = traceback.tb_next
                """
            ),
        ],
        check=False,
        capture_output=True,
        env=clean_environment,
        text=True,
    )
    assert runtime_probe.returncode == 0, runtime_probe.stdout + runtime_probe.stderr

    positive = tmp_path / "positive_consumer.py"
    positive.write_text(
        textwrap.dedent(
            """
            from boundarybench import BENCHMARK_VERSION
            from operatebench.identity import (
                ComponentIdentityV1,
                RequirementEvaluationV1,
                evaluate_component_requirements,
            )

            requirements: list[dict[str, object]] = []
            components: dict[str, ComponentIdentityV1] = {}
            result: RequirementEvaluationV1 = evaluate_component_requirements(
                requirements, components
            )
            version: int = result.requirement_contract_version
            satisfied: bool = result.satisfied
            reasons: tuple[str, ...] = result.reasons
            reveal_type(result)
            reveal_type(version)
            reveal_type(satisfied)
            reveal_type(reasons)
            reveal_type(BENCHMARK_VERSION)
            """
        ),
        encoding="utf-8",
    )
    negative = tmp_path / "negative_consumer.py"
    negative.write_text(
        textwrap.dedent(
            """
            from collections.abc import Mapping
            from operatebench.identity import (
                ComponentIdentityV1,
                evaluate_component_requirements,
            )

            requirements: list[dict[str, object]] = []
            requirement_tuple: tuple[dict[str, object], ...] = ()
            requirement_mapping: Mapping[str, object] = {}
            component_mapping: Mapping[str, ComponentIdentityV1] = {}
            component_pairs: tuple[tuple[str, ComponentIdentityV1], ...] = ()
            evaluate_component_requirements(requirement_tuple, {})
            evaluate_component_requirements(requirement_mapping, {})
            evaluate_component_requirements(requirements, component_mapping)
            evaluate_component_requirements(requirements, component_pairs)
            """
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "mypy",
        "--strict",
        "--no-incremental",
        "--python-executable",
        str(python),
    ]
    accepted = subprocess.run(
        [*command, str(positive)],
        cwd=tmp_path,
        env=clean_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    for revealed in (
        "operatebench.identity.RequirementEvaluationV1",
        "int",
        "bool",
        "tuple[str, ...]",
        "str",
    ):
        assert f'Revealed type is "{revealed}"' in accepted.stdout

    rejected = subprocess.run(
        [*command, str(negative)],
        cwd=tmp_path,
        env=clean_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 1, rejected.stdout + rejected.stderr
    assert rejected.stdout.count("[arg-type]") == 4
    assert "Skipping analyzing" not in rejected.stdout
