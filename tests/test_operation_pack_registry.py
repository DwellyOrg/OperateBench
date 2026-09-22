"""Static operation-pack registry identity and mutation boundaries."""

from __future__ import annotations

import copy
import json
import pickle
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, astuple, fields, replace
from pathlib import Path
from types import ModuleType

import pytest

import operatebench.sdk.registry as registry_module
from operatebench.jsonsafe import (
    MAX_JSON_DEPTH,
    CyclicStructureError,
    JsonSafetyError,
    NestingDepthError,
    TextEncodingError,
)
from operatebench.sdk import (
    CheckRequest,
    CommandResult,
    OperationPackMetadata,
    OperationPackRegistry,
    OperationPackRegistryError,
    ReplayRequest,
    RunRequest,
    UnknownOperationPackError,
    ValidateRequest,
)
from operatebench.sdk._validation import (
    canonical_owner_display_name_key,
    is_safe_owner_display_name,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakePack:
    def __init__(
        self,
        pack_id: str,
        operation_type: str,
        *,
        operation_id: str | None = None,
        aliases: tuple[str, ...] = (),
        owner_id: str = "prospire",
        owner_display_name: str = "PROSPIRE TECHNOLOGIES LTD",
        contribution_kind: str = "maintainer",
    ) -> None:
        self.metadata = OperationPackMetadata(
            pack_id=pack_id,
            operation_type=operation_type,
            operation_id=operation_id or pack_id.replace(".", "_"),
            pack_version="0.1.0",
            display_name=pack_id,
            owner_id=owner_id,
            owner_display_name=owner_display_name,
            contribution_kind=contribution_kind,
            status="incubator",
            privacy_status="SYNTHETIC_ONLY",
            default_spec=None,
            agent_ids=("reference",),
            aliases=aliases,
        )

    def validate(self, request: ValidateRequest) -> CommandResult:
        return CommandResult(True, {"spec": str(request.spec_path)})

    def run(self, request: RunRequest) -> CommandResult:
        return CommandResult(True, {"scenario": request.scenario_id})

    def replay(self, request: ReplayRequest) -> CommandResult:
        return CommandResult(True, {"run": str(request.run_path)})

    def check(self, request: CheckRequest) -> CommandResult:
        return CommandResult(True, {"agents": request.agent_ids})


def test_owner_display_validation_requires_exact_nfc_str() -> None:
    assert is_safe_owner_display_name("Étoile")
    assert is_safe_owner_display_name("技術株式会社")
    assert not is_safe_owner_display_name("E\u0301toile")
    assert not is_safe_owner_display_name(True)
    assert canonical_owner_display_name_key("ÉTOILE") == (
        canonical_owner_display_name_key("E\u0301toile")
    )
    with pytest.raises(TypeError, match="exact str"):
        canonical_owner_display_name_key(True)  # type: ignore[arg-type]


def test_sdk_v2_owner_metadata_and_deterministic_queries() -> None:
    alpha = FakePack("alpha.synthetic", "alpha.synthetic")
    zeta = FakePack("zeta.synthetic", "zeta.synthetic")
    registry = OperationPackRegistry((zeta, alpha))
    assert alpha.metadata.sdk_api_version == 2
    assert alpha.metadata.as_dict()["operation_id"] == "alpha_synthetic"
    assert list(alpha.metadata.as_dict())[5:8] == [
        "owner_id",
        "owner_display_name",
        "contribution_kind",
    ]
    assert registry.owner_ids() == ("prospire",)
    assert registry.metadata_for_owner("prospire") == (
        alpha.metadata,
        zeta.metadata,
    )


def test_registry_rejects_duplicate_operation_id() -> None:
    with pytest.raises(
        OperationPackRegistryError, match=r"operation_id.*claimed by both"
    ):
        OperationPackRegistry(
            (
                FakePack("alpha.synthetic", "alpha.synthetic", operation_id="shared_v1"),
                FakePack("beta.synthetic", "beta.synthetic", operation_id="shared_v1"),
            )
        )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ({"aliases": ("shared",)}, {"operation_id": "shared"}),
        ({"pack_id": "shared"}, {"operation_id": "shared"}),
        ({"operation_type": "shared"}, {"operation_id": "shared"}),
        ({"operation_id": "shared"}, {"pack_id": "shared"}),
        ({"operation_id": "shared"}, {"operation_type": "shared"}),
        ({"operation_id": "shared"}, {"aliases": ("shared",)}),
    ],
)
def test_registry_rejects_cross_pack_cross_kind_identity_collisions(
    left: dict[str, object], right: dict[str, object]
) -> None:
    defaults = (
        {
            "pack_id": "alpha.synthetic",
            "operation_type": "alpha.synthetic",
            "operation_id": "alpha_v1",
        },
        {
            "pack_id": "beta.synthetic",
            "operation_type": "beta.synthetic",
            "operation_id": "beta_v1",
        },
    )
    packs = tuple(
        FakePack(**(base | values))  # type: ignore[arg-type]
        for base, values in zip(defaults, (left, right), strict=True)
    )
    with pytest.raises(OperationPackRegistryError) as caught:
        OperationPackRegistry(packs)
    message = str(caught.value)
    assert "identity 'shared'" in message
    assert packs[0].metadata.pack_id in message
    assert packs[1].metadata.pack_id in message
    assert any(kind in message for kind in ("alias", "pack_id", "operation_type"))
    assert "operation_id" in message


@pytest.mark.parametrize("own_kind", ["pack_id", "operation_type", "alias"])
def test_registry_rejects_operation_id_equal_to_own_other_identity_kind(
    own_kind: str,
) -> None:
    kwargs: dict[str, object] = {
        "pack_id": "alpha.synthetic",
        "operation_type": "alpha.type",
        "operation_id": "shared",
    }
    if own_kind == "alias":
        kwargs["aliases"] = ("shared",)
    else:
        kwargs[own_kind] = "shared"
    with pytest.raises(OperationPackRegistryError) as caught:
        OperationPackRegistry((FakePack(**kwargs),))  # type: ignore[arg-type]
    assert "identity 'shared'" in str(caught.value)
    assert "operation_id" in str(caught.value)
    assert own_kind in str(caught.value)


def test_registry_accepts_same_pack_id_and_operation_type() -> None:
    pack = FakePack("alpha.synthetic", "alpha.synthetic", operation_id="alpha_v1")
    assert OperationPackRegistry((pack,)).resolve("alpha.synthetic") is pack


def test_registry_snapshots_metadata_and_refuses_dispatch_after_drift() -> None:
    pack = FakePack("alpha.synthetic", "alpha.synthetic", aliases=("alpha",))
    original = pack.metadata
    registry = OperationPackRegistry((pack,))
    pack.metadata = replace(
        original, owner_display_name="Forged Ltd", contribution_kind="partner"
    )
    assert registry.metadata() == (original,)
    assert registry.metadata_for_owner("prospire") == (original,)
    with pytest.raises(OperationPackRegistryError, match="drift"):
        registry.resolve("alpha")


def test_registry_resolve_rejects_later_metadata_descriptor_without_invoking_it() -> None:
    calls = 0

    class MutablePack(FakePack):
        pass

    pack = MutablePack("alpha.synthetic", "alpha.synthetic")
    registry = OperationPackRegistry((pack,))

    def hostile_metadata(self: object) -> object:
        nonlocal calls
        calls += 1
        raise RuntimeError("hostile drift getter invoked")

    MutablePack.metadata = property(hostile_metadata)  # type: ignore[attr-defined]
    with pytest.raises(OperationPackRegistryError, match="drift"):
        registry.resolve("alpha.synthetic")
    assert calls == 0


def test_registry_resolve_rejects_deleted_metadata() -> None:
    pack = FakePack("alpha.synthetic", "alpha.synthetic")
    registry = OperationPackRegistry((pack,))
    del pack.metadata
    with pytest.raises(OperationPackRegistryError, match="drift"):
        registry.resolve("alpha.synthetic")


def test_registry_rejects_metadata_subclass_before_its_hooks() -> None:
    class HostileMetadata(OperationPackMetadata):
        def as_dict(self) -> dict[str, object]:
            raise AssertionError("hostile metadata hook invoked")

    pack = FakePack("alpha.synthetic", "alpha.synthetic")
    pack.metadata = HostileMetadata(**asdict(pack.metadata))
    with pytest.raises(OperationPackRegistryError, match="expected exact"):
        OperationPackRegistry((pack,))


def test_reserved_manifest_exactly_matches_trusted_builtins() -> None:
    from operatebench.sdk.builtins import BUILTIN_PACKS
    from tools.check_contribution_isolation import load_reserved_operation_identities

    runtime = tuple(
        {
            "owner_id": pack.metadata.owner_id,
            "owner_display_name": pack.metadata.owner_display_name,
            "contribution_kind": pack.metadata.contribution_kind,
            "pack_id": pack.metadata.pack_id,
            "operation_type": pack.metadata.operation_type,
            "operation_id": pack.metadata.operation_id,
            "aliases": pack.metadata.aliases,
            "source_module": type(pack).__module__,
            "source_path": "src/"
            + type(pack).__module__.removesuffix(".pack").replace(".", "/"),
        }
        for pack in BUILTIN_PACKS
    )
    reserved = load_reserved_operation_identities(REPO_ROOT)
    assert runtime == reserved


@pytest.mark.parametrize("owner_id", [True, 1, "PROSPIRE", "pro-spire", " prospire"])
def test_metadata_for_owner_rejects_non_exact_or_malformed_owner_ids(
    owner_id: object,
) -> None:
    registry = OperationPackRegistry((FakePack("alpha.synthetic", "alpha.synthetic"),))
    with pytest.raises((TypeError, ValueError)):
        registry.metadata_for_owner(owner_id)  # type: ignore[arg-type]


def test_metadata_for_owner_rejects_hostile_str_subclasses() -> None:
    class Hostile(str):
        def __eq__(self, other: object) -> bool:
            raise AssertionError("hostile equality invoked")

        def __hash__(self) -> int:
            raise AssertionError("hostile hash invoked")

    registry = OperationPackRegistry((FakePack("alpha.synthetic", "alpha.synthetic"),))
    with pytest.raises(TypeError):
        registry.metadata_for_owner(Hostile("prospire"))
    assert registry.metadata_for_owner("unknown") == ()


def test_all_builtins_report_the_same_prospire_maintainer_identity() -> None:
    from operatebench.sdk.builtins import BUILTIN_PACKS

    assert len(BUILTIN_PACKS.metadata()) == 3
    assert {
        (item.owner_id, item.owner_display_name, item.contribution_kind)
        for item in BUILTIN_PACKS.metadata()
    } == {("prospire", "PROSPIRE TECHNOLOGIES LTD", "maintainer")}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("owner_id", True),
        ("owner_id", "NorthStar"),
        ("owner_id", "class"),
        ("owner_id", "con"),
        ("owner_id", "a" * 64),
        ("owner_display_name", True),
        ("owner_display_name", ""),
        ("owner_display_name", "unsafe\x00name"),
        ("owner_display_name", "a" * 129),
        ("contribution_kind", True),
        ("contribution_kind", "community"),
    ],
)
def test_registry_rejects_invalid_owner_metadata(field: str, value: object) -> None:
    pack = FakePack("alpha.synthetic", "alpha.synthetic")
    pack.metadata = replace(pack.metadata, **{field: value})
    with pytest.raises(OperationPackRegistryError, match=field):
        OperationPackRegistry((pack,))


@pytest.mark.parametrize("candidate", [object(), type("MissingMetadata", (), {})()])
def test_registry_wraps_malformed_candidate_protocol(candidate: object) -> None:
    with pytest.raises(OperationPackRegistryError, match="metadata"):
        OperationPackRegistry((candidate,))  # type: ignore[arg-type]


def test_registry_rejects_metadata_property_without_invoking_it() -> None:
    calls = 0

    class Broken:
        @property
        def metadata(self) -> object:
            nonlocal calls
            calls += 1
            raise RuntimeError("hostile metadata getter invoked")

    with pytest.raises(OperationPackRegistryError, match="metadata"):
        OperationPackRegistry((Broken(),))  # type: ignore[arg-type]
    assert calls == 0


def test_registry_rejects_custom_metadata_descriptor_without_invoking_it() -> None:
    calls = 0

    class HostileDescriptor:
        def __get__(self, instance: object, owner: type[object]) -> object:
            nonlocal calls
            calls += 1
            raise RuntimeError("hostile metadata descriptor invoked")

    class Broken:
        metadata = HostileDescriptor()

    with pytest.raises(OperationPackRegistryError, match="metadata"):
        OperationPackRegistry((Broken(),))  # type: ignore[arg-type]
    assert calls == 0


def test_registry_does_not_broadly_catch_static_lookup_internal_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_static_lookup(candidate: object, name: str) -> object:
        raise RuntimeError("trusted static lookup failure")

    monkeypatch.setattr(registry_module, "_static_attribute", broken_static_lookup)
    with pytest.raises(RuntimeError, match="trusted static lookup failure"):
        OperationPackRegistry((object(),))  # type: ignore[arg-type]


def test_registry_rejects_validate_property_without_invoking_candidate_hooks() -> None:
    getter_calls = 0
    attribute_calls = 0
    repr_calls = 0
    str_calls = 0

    class Broken(FakePack):
        def __getattribute__(self, name: str) -> object:
            nonlocal attribute_calls
            attribute_calls += 1
            raise RuntimeError(f"hostile attribute lookup invoked for {name}")

        def __repr__(self) -> str:
            nonlocal repr_calls
            repr_calls += 1
            raise RuntimeError("hostile repr invoked")

        def __str__(self) -> str:
            nonlocal str_calls
            str_calls += 1
            raise RuntimeError("hostile str invoked")

        @property
        def validate(self) -> object:
            nonlocal getter_calls
            getter_calls += 1
            raise RuntimeError("hostile validate getter invoked")

    pack = Broken("alpha.synthetic", "alpha.synthetic")
    with pytest.raises(OperationPackRegistryError, match=r"validate.*callable"):
        OperationPackRegistry((pack,))
    assert (getter_calls, attribute_calls, repr_calls, str_calls) == (0, 0, 0, 0)


def test_registry_rejects_custom_capability_descriptor_without_invoking_it() -> None:
    calls = 0

    class HostileDescriptor:
        def __get__(self, instance: object, owner: type[object]) -> object:
            nonlocal calls
            calls += 1
            raise RuntimeError("hostile descriptor invoked")

    class Broken(FakePack):
        validate = HostileDescriptor()  # type: ignore[assignment]

    with pytest.raises(OperationPackRegistryError, match=r"validate.*callable"):
        OperationPackRegistry((Broken("alpha.synthetic", "alpha.synthetic"),))
    assert calls == 0


def test_registry_accepts_ordinary_pack_without_dynamic_admission_lookup() -> None:
    class Ordinary(FakePack):
        pass

    # Use object access only to prepare the hostile lookup guard itself.
    pack = Ordinary("alpha.synthetic", "alpha.synthetic")
    metadata = object.__getattribute__(pack, "metadata")
    pack.guarded_names = {"metadata", *metadata.capabilities}

    def guarded_getattribute(self: Ordinary, name: str) -> object:
        if name in object.__getattribute__(self, "guarded_names"):
            raise RuntimeError("dynamic admission lookup invoked")
        return object.__getattribute__(self, name)

    Ordinary.__getattribute__ = guarded_getattribute  # type: ignore[method-assign]
    registry = OperationPackRegistry((pack,))
    assert registry.canonical_ids() == ("alpha.synthetic",)


@pytest.mark.parametrize("hostile_field", ("__module__", "__name__"))
def test_partner_class_identity_avoids_hostile_metaclass_lookup(
    hostile_field: str,
) -> None:
    calls = 0

    class HostileMeta(type):
        def __getattribute__(cls, name: str) -> object:
            nonlocal calls
            calls += 1
            if name == hostile_field:
                raise RuntimeError("candidate code ran")
            return super().__getattribute__(name)

    class HostilePartner(FakePack, metaclass=HostileMeta):
        pass

    pack = HostilePartner(
        "partner.northstar.sample",
        "northstar.sample.synthetic",
        owner_id="northstar",
        owner_display_name="Northstar Fictional Company",
        contribution_kind="partner",
    )
    pack.metadata = replace(
        pack.metadata,
        default_spec="examples/contributions/northstar/sample/operation.yaml",
    )
    with pytest.raises(OperationPackRegistryError, match="implementation module"):
        OperationPackRegistry((pack,))
    assert calls == 0


@pytest.mark.parametrize("field", ("__module__", "__name__"))
@pytest.mark.parametrize("value", (object(), "descriptor", 7))
def test_registry_rejects_non_string_class_identity_namespace_without_execution(
    field: str, value: object
) -> None:
    calls = 0

    class HostileDescriptor:
        def __get__(self, instance: object, owner: type[object]) -> object:
            nonlocal calls
            calls += 1
            raise RuntimeError("descriptor ran")

    stored = HostileDescriptor() if value == "descriptor" else value
    pack_type = type("MalformedPack", (FakePack,), {field: stored})
    with pytest.raises(OperationPackRegistryError, match=field):
        OperationPackRegistry((pack_type("alpha.synthetic", "alpha.synthetic"),))
    assert calls == 0


@pytest.mark.parametrize("field", ("__module__", "__name__"))
def test_registry_rejects_metaclass_identity_descriptor_without_execution(
    field: str,
) -> None:
    calls = 0

    class HostileDescriptor:
        def __get__(self, instance: object, owner: type[object]) -> object:
            nonlocal calls
            calls += 1
            raise RuntimeError("metaclass descriptor ran")

        def __set__(self, instance: object, value: object) -> None:
            raise RuntimeError("metaclass descriptor set")

    metaclass = type("IdentityMeta", (type,), {field: HostileDescriptor()})
    pack_type = metaclass("HostilePack", (FakePack,), {})
    with pytest.raises(OperationPackRegistryError, match=field):
        OperationPackRegistry((pack_type("alpha.synthetic", "alpha.synthetic"),))
    assert calls == 0


def test_registry_propagates_class_identity_helper_programmer_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("trusted class identity programmer error")

    def fail_identity(candidate: object) -> tuple[type[object], str, str]:
        raise failure

    monkeypatch.setattr(registry_module, "_candidate_class_identity", fail_identity)
    with pytest.raises(RuntimeError) as caught:
        OperationPackRegistry((FakePack("alpha.synthetic", "alpha.synthetic"),))
    assert caught.value is failure


def test_registry_preserves_static_method_dispatch_after_admission() -> None:
    class StaticPack(FakePack):
        @staticmethod
        def validate(request: ValidateRequest) -> CommandResult:
            return CommandResult(True, {"spec": str(request.spec_path)})

    pack = StaticPack("alpha.synthetic", "alpha.synthetic")
    resolved = OperationPackRegistry((pack,)).resolve("alpha.synthetic")
    result = resolved.validate(ValidateRequest(Path("fixture.yaml")))
    assert result.contract_passed is True
    assert result.payload == {"spec": "fixture.yaml"}


@pytest.mark.parametrize(
    "value",
    ["unsafe\u202ename", "unsafe\u202dname", "unsafe\u2066name", "zero\u200bwidth"],
)
def test_registry_rejects_unicode_format_owner_display(value: str) -> None:
    pack = FakePack("alpha.synthetic", "alpha.synthetic")
    pack.metadata = replace(pack.metadata, owner_display_name=value)
    with pytest.raises(OperationPackRegistryError, match="owner_display_name"):
        OperationPackRegistry((pack,))


def test_registry_accepts_ordinary_international_owner_display() -> None:
    pack = FakePack(
        "alpha.synthetic", "alpha.synthetic", owner_display_name="Étoile 技術株式会社 № ٢"
    )
    assert OperationPackRegistry((pack,)).metadata() == (pack.metadata,)


def test_registry_rejects_non_nfc_owner_display() -> None:
    pack = FakePack(
        "alpha.synthetic", "alpha.synthetic", owner_display_name="E\u0301toile"
    )
    with pytest.raises(OperationPackRegistryError, match="owner_display_name"):
        OperationPackRegistry((pack,))


def test_registry_canonicalizes_owner_collision_if_validator_is_bypassed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        registry_module, "is_safe_owner_display_name", lambda _value: True
    )
    left = FakePack("alpha.synthetic", "alpha.synthetic", owner_display_name="Étoile")
    right = FakePack(
        "beta.synthetic",
        "beta.synthetic",
        owner_id="another",
        owner_display_name="E\u0301toile",
    )
    with pytest.raises(OperationPackRegistryError, match="display name"):
        OperationPackRegistry((left, right))


@pytest.mark.parametrize("field", ("owner_display_name", "contribution_kind"))
def test_registry_rejects_incoherent_identity_for_owner(field: str) -> None:
    left = FakePack("alpha.synthetic", "alpha.synthetic")
    kwargs = {field: "Another Company" if field == "owner_display_name" else "partner"}
    pack_id = (
        "partner.prospire.beta" if field == "contribution_kind" else "beta.synthetic"
    )
    pack_type = FakePack
    if field == "contribution_kind":
        pack_type = type("PartnerPack", (FakePack,), {})
        pack_type.__module__ = "operatebench.contributions.prospire.beta.pack"
    right = pack_type(pack_id, "beta.synthetic", **kwargs)
    if field == "contribution_kind":
        right.metadata = replace(
            right.metadata,
            default_spec="examples/contributions/prospire/beta/operation.yaml",
        )
    with pytest.raises(OperationPackRegistryError, match="owner_id"):
        OperationPackRegistry((left, right))


def test_registry_rejects_case_insensitive_display_name_impersonation() -> None:
    left = FakePack("alpha.synthetic", "alpha.synthetic")
    right = FakePack(
        "beta.synthetic",
        "beta.synthetic",
        owner_id="another",
        owner_display_name="prospire technologies ltd",
    )
    with pytest.raises(OperationPackRegistryError, match="display name"):
        OperationPackRegistry((left, right))


def test_partner_identity_must_match_pack_and_implementation_module() -> None:
    wrong_id = FakePack(
        "wrong.synthetic",
        "wrong.synthetic",
        owner_id="northstar",
        owner_display_name="Northstar Example Ltd",
        contribution_kind="partner",
    )
    with pytest.raises(OperationPackRegistryError, match=r"partner\.northstar"):
        OperationPackRegistry((wrong_id,))

    class PartnerPack(FakePack):
        pass

    PartnerPack.__module__ = "operatebench.contributions.riverside.sample.pack"
    wrong_module = PartnerPack(
        "partner.northstar.sample",
        "sample.synthetic",
        owner_id="northstar",
        owner_display_name="Northstar Example Ltd",
        contribution_kind="partner",
    )
    wrong_module.metadata = replace(
        wrong_module.metadata,
        default_spec="examples/contributions/northstar/sample/operation.yaml",
    )
    with pytest.raises(OperationPackRegistryError, match="implementation module"):
        OperationPackRegistry((wrong_module,))


@pytest.mark.parametrize(
    "default_spec",
    [None, "examples/contributions/riverside/sample/operation.yaml"],
)
def test_partner_default_spec_is_required_and_owner_bound(
    default_spec: str | None,
) -> None:
    class PartnerPack(FakePack):
        pass

    PartnerPack.__module__ = "operatebench.contributions.northstar.sample.pack"
    pack = PartnerPack(
        "partner.northstar.sample",
        "sample.synthetic",
        owner_id="northstar",
        owner_display_name="Northstar Example Ltd",
        contribution_kind="partner",
    )
    pack.metadata = replace(pack.metadata, default_spec=default_spec)
    with pytest.raises(OperationPackRegistryError, match="default spec"):
        OperationPackRegistry((pack,))


def test_registry_resolves_canonical_and_alias_and_lists_canonical_sorted() -> None:
    zeta = FakePack("zeta.synthetic", "zeta.synthetic", aliases=("zeta",))
    alpha = FakePack("alpha.synthetic", "alpha.synthetic", aliases=("alpha",))
    registry = OperationPackRegistry((zeta, alpha))
    assert registry.canonical_ids() == ("alpha.synthetic", "zeta.synthetic")
    assert registry.resolve("alpha") is alpha
    assert registry.resolve("zeta.synthetic") is zeta


@pytest.mark.parametrize(
    "left,right",
    [
        (
            FakePack("alpha.synthetic", "alpha.synthetic", aliases=("shared",)),
            FakePack("beta.synthetic", "beta.synthetic", aliases=("shared",)),
        ),
        (
            FakePack("alpha.synthetic", "same.synthetic"),
            FakePack("beta.synthetic", "same.synthetic"),
        ),
    ],
)
def test_registry_refuses_name_and_operation_type_collisions(
    left: FakePack, right: FakePack
) -> None:
    with pytest.raises(OperationPackRegistryError):
        OperationPackRegistry((left, right))


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("collision_kind", ("canonical", "alias"))
def test_registry_refuses_lookup_name_owned_as_another_pack_operation_type(
    reverse: bool, collision_kind: str
) -> None:
    operation_owner = FakePack("alpha.synthetic", "shared.operation.synthetic")
    if collision_kind == "canonical":
        lookup_owner = FakePack("shared.operation.synthetic", "beta.operation.synthetic")
    else:
        lookup_owner = FakePack(
            "beta.synthetic",
            "beta.operation.synthetic",
            aliases=("shared.operation.synthetic",),
        )
    packs = (
        (lookup_owner, operation_owner) if reverse else (operation_owner, lookup_owner)
    )
    with pytest.raises(OperationPackRegistryError, match="lookup name"):
        OperationPackRegistry(packs)


def test_registry_allows_own_operation_type_as_alias() -> None:
    pack = FakePack(
        "reviewed.synthetic",
        "commerce.return_refund.synthetic",
        aliases=("commerce.return_refund.synthetic",),
    )
    registry = OperationPackRegistry((pack,))
    assert registry.resolve(pack.metadata.operation_type) is pack


def test_registry_refuses_claim_of_evidence_admission() -> None:
    pack = FakePack("alpha.synthetic", "alpha.synthetic")
    pack.metadata = replace(pack.metadata, evidence_eligible=True)
    with pytest.raises(OperationPackRegistryError, match="evidence_eligible"):
        OperationPackRegistry((pack,))


def test_registry_refuses_unsorted_or_duplicate_agents() -> None:
    pack = FakePack("alpha.synthetic", "alpha.synthetic")
    pack.metadata = replace(pack.metadata, agent_ids=("reference", "aaa"))
    with pytest.raises(OperationPackRegistryError, match="sorted"):
        OperationPackRegistry((pack,))


def test_unknown_pack_is_named_and_sorted() -> None:
    registry = OperationPackRegistry((FakePack("alpha.synthetic", "alpha.synthetic"),))
    with pytest.raises(UnknownOperationPackError, match=r"alpha\.synthetic"):
        registry.resolve("missing")


def test_registry_has_no_runtime_mutation_surface() -> None:
    registry = OperationPackRegistry((FakePack("alpha.synthetic", "alpha.synthetic"),))
    assert not hasattr(registry, "register")
    assert not hasattr(registry, "unregister")


def test_registry_refuses_present_but_non_callable_capability() -> None:
    pack = FakePack("alpha.synthetic", "alpha.synthetic")
    pack.run = 1  # type: ignore[method-assign,assignment]
    with pytest.raises(OperationPackRegistryError, match=r"run.*callable"):
        OperationPackRegistry((pack,))


@pytest.mark.parametrize(
    "default_spec",
    ("./fixture.yaml", "fixtures//fixture.yaml", "fixtures\\fixture.yaml"),
)
def test_registry_refuses_non_normalized_default_spec(default_spec: str) -> None:
    pack = FakePack("alpha.synthetic", "alpha.synthetic")
    pack.metadata = replace(pack.metadata, default_spec=default_spec)
    with pytest.raises(OperationPackRegistryError, match="normalized relative path"):
        OperationPackRegistry((pack,))


@pytest.mark.parametrize("field", ("display_name", "default_spec"))
@pytest.mark.parametrize("value", ("invalid\ud800text", "invalid\udffftext"))
def test_registry_translates_invalid_metadata_text(field: str, value: str) -> None:
    pack = FakePack("alpha.synthetic", "alpha.synthetic")
    pack.metadata = replace(pack.metadata, **{field: value})
    with pytest.raises(OperationPackRegistryError, match=field) as caught:
        OperationPackRegistry((pack,))
    assert isinstance(caught.value.__cause__, TextEncodingError)
    assert value not in str(caught.value)


def test_registry_translates_exact_reviewer_default_spec_reproduction() -> None:
    pack = FakePack("alpha.synthetic", "alpha.synthetic")
    pack.metadata = replace(pack.metadata, default_spec="\ud800")
    with pytest.raises(OperationPackRegistryError, match="default_spec"):
        OperationPackRegistry((pack,))


@pytest.mark.parametrize("value", ("\ud800", "\udfff"))
def test_registry_translates_invalid_final_metadata_projection(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    original = OperationPackMetadata.as_dict

    def invalid_projection(metadata: OperationPackMetadata) -> dict[str, object]:
        projected = original(metadata)
        projected["mutant"] = value
        return projected

    monkeypatch.setattr(OperationPackMetadata, "as_dict", invalid_projection)
    pack = FakePack("alpha.synthetic", "alpha.synthetic")
    with pytest.raises(OperationPackRegistryError, match="metadata projection") as caught:
        OperationPackRegistry((pack,))
    assert isinstance(caught.value.__cause__, TextEncodingError)


def test_registry_accepts_valid_unicode_display_and_default_spec() -> None:
    pack = FakePack("alpha.synthetic", "alpha.synthetic")
    pack.metadata = replace(
        pack.metadata,
        display_name="Étoile 技術 № ٢",
        default_spec="examples/合法/操作.yaml",
    )
    assert OperationPackRegistry((pack,)).metadata() == (pack.metadata,)


@pytest.mark.parametrize("field", ("display_name", "default_spec"))
def test_registry_propagates_check_text_programmer_errors(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    failure = RuntimeError("trusted check_text programmer error")

    def fail_check_text(value: str, context: str) -> str:
        raise failure

    monkeypatch.setattr(registry_module, "check_text", fail_check_text)
    pack = FakePack("alpha.synthetic", "alpha.synthetic")
    if field == "default_spec":
        pack.metadata = replace(pack.metadata, default_spec="examples/valid.yaml")
    with pytest.raises(RuntimeError) as caught:
        OperationPackRegistry((pack,))
    assert caught.value is failure


def test_registry_propagates_json_safety_programmer_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = TypeError("trusted ensure_json_safe programmer error")

    def fail_json_safety(value: object, context: str) -> object:
        raise failure

    monkeypatch.setattr(registry_module, "ensure_json_safe", fail_json_safety)
    with pytest.raises(TypeError) as caught:
        OperationPackRegistry((FakePack("alpha.synthetic", "alpha.synthetic"),))
    assert caught.value is failure


def test_command_result_refuses_non_json_or_non_utf8_evidence() -> None:
    with pytest.raises(TypeError, match="contract_passed must be bool"):
        CommandResult(1)  # type: ignore[arg-type]
    with pytest.raises(JsonSafetyError):
        CommandResult(True, {"bad": {1, 2}})
    with pytest.raises(JsonSafetyError):
        CommandResult(True, {}, ("\ud800",))


@pytest.mark.parametrize("kind", ("dict", "list", "custom-mapping"))
def test_command_result_names_cyclic_payload_before_projection(kind: str) -> None:
    if kind == "dict":
        cyclic: object = {}
        cyclic["self"] = cyclic  # type: ignore[index]
    elif kind == "list":
        cyclic = []
        cyclic.append(cyclic)  # type: ignore[attr-defined]
    else:

        class CyclicMapping(Mapping[str, object]):
            def __getitem__(self, key: str) -> object:
                if key != "self":
                    raise KeyError(key)
                return self

            def __iter__(self):  # type: ignore[no-untyped-def]
                yield "self"

            def __len__(self) -> int:
                return 1

        cyclic = CyclicMapping()

    with pytest.raises(CyclicStructureError):
        CommandResult(True, {"cyclic": cyclic})


def test_command_result_names_excessive_depth_before_projection_recurses() -> None:
    over_depth: object = "leaf"
    for _ in range(max(MAX_JSON_DEPTH + 1, 1_500)):
        over_depth = [over_depth]

    with pytest.raises(NestingDepthError):
        CommandResult(True, {"deep": over_depth})


def test_command_result_post_validates_computed_nested_mapping_projection() -> None:
    class ChangingMapping(Mapping[str, object]):
        def __init__(self) -> None:
            self.reads = 0

        def __getitem__(self, key: str) -> object:
            if key != "value":
                raise KeyError(key)
            self.reads += 1
            return "safe on validation" if self.reads == 1 else {"not-json"}

        def __iter__(self):  # type: ignore[no-untyped-def]
            yield "value"

        def __len__(self) -> int:
            return 1

    changing = ChangingMapping()
    with pytest.raises(JsonSafetyError):
        CommandResult(True, {"nested": changing})
    assert changing.reads == 2


@pytest.mark.parametrize("kind", ("mapping", "sequence"))
def test_command_result_names_cycle_that_appears_during_projection(kind: str) -> None:
    class ChangingMapping(Mapping[str, object]):
        def __init__(self) -> None:
            self.reads = 0

        def __getitem__(self, key: str) -> object:
            if key != "value":
                raise KeyError(key)
            return "safe on pre-validation"

        def __iter__(self):  # type: ignore[no-untyped-def]
            yield "value"

        def __len__(self) -> int:
            return 1

        def items(self):  # type: ignore[no-untyped-def]
            self.reads += 1
            yield "value", "safe on pre-validation" if self.reads == 1 else self

    class ChangingSequence(Sequence[object]):
        def __init__(self) -> None:
            self.reads = 0

        def __getitem__(self, index: int) -> object:  # type: ignore[override]
            if index != 0:
                raise IndexError(index)
            return "safe on pre-validation"

        def __len__(self) -> int:
            return 1

        def __iter__(self):
            self.reads += 1
            yield "safe on pre-validation" if self.reads == 1 else self

    changing = ChangingMapping() if kind == "mapping" else ChangingSequence()

    with pytest.raises(CyclicStructureError, match=r"JSON projection.*cyclic reference"):
        CommandResult(True, {"nested": changing})

    assert changing.reads == 2


@pytest.mark.parametrize("kind", ("mapping", "sequence"))
def test_command_result_names_depth_that_appears_during_projection(kind: str) -> None:
    over_depth: object = "leaf"
    for _ in range(max(MAX_JSON_DEPTH + 1, 1_500)):
        over_depth = [over_depth]

    class ChangingMapping(Mapping[str, object]):
        def __init__(self) -> None:
            self.reads = 0

        def __getitem__(self, key: str) -> object:
            if key != "value":
                raise KeyError(key)
            return "safe on pre-validation"

        def __iter__(self):  # type: ignore[no-untyped-def]
            yield "value"

        def __len__(self) -> int:
            return 1

        def items(self):  # type: ignore[no-untyped-def]
            self.reads += 1
            yield "value", "safe on pre-validation" if self.reads == 1 else over_depth

    class ChangingSequence(Sequence[object]):
        def __init__(self) -> None:
            self.reads = 0

        def __getitem__(self, index: int) -> object:  # type: ignore[override]
            if index != 0:
                raise IndexError(index)
            return "safe on pre-validation"

        def __len__(self) -> int:
            return 1

        def __iter__(self):
            self.reads += 1
            yield "safe on pre-validation" if self.reads == 1 else over_depth

    changing = ChangingMapping() if kind == "mapping" else ChangingSequence()

    with pytest.raises(NestingDepthError, match=rf"deeper than {MAX_JSON_DEPTH} levels"):
        CommandResult(True, {"nested": changing})

    assert changing.reads == 2


def test_command_result_recursively_detaches_input_and_each_payload_access() -> None:
    source = {"outer": {"rows": [{"value": "original"}]}}
    result = CommandResult(True, source)

    source["outer"]["rows"][0]["value"] = "mutated input"
    first = result.payload
    first["outer"]["rows"][0]["value"] = "mutated output"
    second = result.payload
    second["outer"]["rows"].append({"value": "new output row"})

    assert result.payload == {"outer": {"rows": [{"value": "original"}]}}


def test_command_result_preserves_public_dataclass_contract() -> None:
    source = {"outer": {"rows": [{"value": "original"}]}}
    result = CommandResult(True, source, ("done",))
    source["outer"]["rows"][0]["value"] = "mutated input"

    assert [(item.name, item.init) for item in fields(result)] == [
        ("contract_passed", True),
        ("payload", True),
        ("text_lines", True),
    ]
    assert repr(result) == (
        "CommandResult(contract_passed=True, "
        "payload={'outer': {'rows': [{'value': 'original'}]}}, "
        "text_lines=('done',))"
    )
    assert result == CommandResult(
        True, {"outer": {"rows": [{"value": "original"}]}}, ("done",)
    )


def test_command_result_supports_dataclass_copy_and_pickle_operations() -> None:
    expected = {"outer": {"rows": [{"value": "original"}]}}
    result = CommandResult(True, expected, ("done",))
    unchanged = replace(result)
    replaced = replace(result, payload={"replacement": [1, 2]})
    deepcopied = copy.deepcopy(result)
    roundtripped = pickle.loads(pickle.dumps(result))

    assert asdict(result) == {
        "contract_passed": True,
        "payload": expected,
        "text_lines": ("done",),
    }
    assert astuple(result) == (True, expected, ("done",))
    assert unchanged == result
    assert replaced.payload == {"replacement": [1, 2]}
    assert deepcopied == result
    assert roundtripped == result

    for candidate in (result, unchanged, replaced, deepcopied, roundtripped):
        projection = candidate.payload
        projection.setdefault("outer", {}).setdefault("rows", []).append("mutation")
    assert result.payload == expected
    assert unchanged.payload == expected
    assert replaced.payload == {"replacement": [1, 2]}
    assert deepcopied.payload == expected
    assert roundtripped.payload == expected


def test_command_result_normalizes_hostile_scalar_subclasses_for_public_operations() -> (
    None
):
    class HostileStr(str):
        __hash__ = str.__hash__

        def __str__(self) -> str:
            raise AssertionError("hostile __str__ executed")

        def __repr__(self) -> str:
            raise AssertionError("hostile __repr__ executed")

        def __eq__(self, other: object) -> bool:
            raise AssertionError("hostile __eq__ executed")

        def __copy__(self) -> str:
            raise AssertionError("hostile __copy__ executed")

        def __deepcopy__(self, memo: object) -> str:
            raise AssertionError("hostile __deepcopy__ executed")

        def __reduce__(self) -> object:
            raise AssertionError("hostile __reduce__ executed")

        def __reduce_ex__(self, protocol: int) -> object:
            raise AssertionError("hostile __reduce_ex__ executed")

    class HostileInt(int):
        def __int__(self) -> int:
            raise AssertionError("hostile __int__ executed")

        def __repr__(self) -> str:
            raise AssertionError("hostile __repr__ executed")

        def __eq__(self, other: object) -> bool:
            raise AssertionError("hostile __eq__ executed")

        def __copy__(self) -> int:
            raise AssertionError("hostile __copy__ executed")

        def __deepcopy__(self, memo: object) -> int:
            raise AssertionError("hostile __deepcopy__ executed")

        def __reduce__(self) -> object:
            raise AssertionError("hostile __reduce__ executed")

        def __reduce_ex__(self, protocol: int) -> object:
            raise AssertionError("hostile __reduce_ex__ executed")

    class HostileFloat(float):
        def __float__(self) -> float:
            raise AssertionError("hostile __float__ executed")

        def __repr__(self) -> str:
            raise AssertionError("hostile __repr__ executed")

        def __eq__(self, other: object) -> bool:
            raise AssertionError("hostile __eq__ executed")

        def __copy__(self) -> float:
            raise AssertionError("hostile __copy__ executed")

        def __deepcopy__(self, memo: object) -> float:
            raise AssertionError("hostile __deepcopy__ executed")

        def __reduce__(self) -> object:
            raise AssertionError("hostile __reduce__ executed")

        def __reduce_ex__(self, protocol: int) -> object:
            raise AssertionError("hostile __reduce_ex__ executed")

    source = {
        HostileStr("outer"): (
            HostileStr("value"),
            {HostileStr("numbers"): [HostileInt(7), HostileFloat(2.5)]},
        )
    }
    expected = {"outer": ["value", {"numbers": [7, 2.5]}]}

    result = CommandResult(True, source, ("done",))
    first = result.payload
    second = result.payload
    unchanged = replace(result)
    copied = copy.copy(result)
    deepcopied = copy.deepcopy(result)
    mapping = asdict(result)
    sequence = astuple(result)
    roundtripped = pickle.loads(pickle.dumps(result))

    match result:
        case CommandResult(contract_passed=True, payload=matched, text_lines=("done",)):
            pass
        case _:
            raise AssertionError("CommandResult pattern did not match")

    assert first == expected
    assert second == expected
    assert first is not second
    assert first["outer"] is not second["outer"]
    assert repr(result) == (
        "CommandResult(contract_passed=True, "
        "payload={'outer': ['value', {'numbers': [7, 2.5]}]}, "
        "text_lines=('done',))"
    )
    assert result == CommandResult(True, expected, ("done",))
    assert unchanged == result
    assert copied == result
    assert deepcopied == result
    assert mapping == {
        "contract_passed": True,
        "payload": expected,
        "text_lines": ("done",),
    }
    assert sequence == (True, expected, ("done",))
    assert roundtripped == result
    assert matched == expected
    assert json.dumps(result.payload, sort_keys=True, separators=(",", ":")) == (
        '{"outer":["value",{"numbers":[7,2.5]}]}'
    )

    for projection in (first, second, matched, result.payload):
        assert type(next(iter(projection))) is str
        assert type(projection["outer"][0]) is str
        assert type(next(iter(projection["outer"][1]))) is str
        assert type(projection["outer"][1]["numbers"][0]) is int
        assert type(projection["outer"][1]["numbers"][1]) is float


def test_command_result_rejects_hostile_non_json_value_without_copy_hooks() -> None:
    hooks: list[str] = []

    class HookedList(list[object]):
        def __deepcopy__(self, memo: object) -> object:
            hooks.append("deepcopy")
            raise AssertionError("user deepcopy hook executed")

        def __reduce__(self) -> object:
            hooks.append("reduce")
            raise AssertionError("user reduce hook executed")

        def __reduce_ex__(self, protocol: int) -> object:
            hooks.append("reduce_ex")
            raise AssertionError("user reduce hook executed")

    class Hostile:
        def __deepcopy__(self, memo: object) -> object:
            hooks.append("deepcopy")
            raise AssertionError("user deepcopy hook executed")

        def __reduce__(self) -> object:
            hooks.append("reduce")
            raise AssertionError("user reduce hook executed")

        def __reduce_ex__(self, protocol: int) -> object:
            hooks.append("reduce_ex")
            raise AssertionError("user reduce hook executed")

    result = CommandResult(True, {"safe": HookedList([{"value": "plain"}])})
    assert result.payload == {"safe": [{"value": "plain"}]}
    with pytest.raises(JsonSafetyError):
        CommandResult(True, {"bad": Hostile()})
    assert hooks == []


def test_static_registration_defaults_to_explicit_non_evidence_admission() -> None:
    metadata = OperationPackMetadata(
        pack_id="development.synthetic",
        operation_type="development.synthetic",
        operation_id="development_synthetic_v1",
        pack_version="0.1.0",
        display_name="Development pack",
        owner_id="prospire",
        owner_display_name="PROSPIRE TECHNOLOGIES LTD",
        contribution_kind="maintainer",
        status="development",
        privacy_status="SYNTHETIC_ONLY",
        default_spec=None,
        agent_ids=("reference",),
    )

    assert metadata.evidence_eligible is False
    assert metadata.as_dict()["evidence_eligible"] is False


def test_partner_pack_rejects_forged_external_class_module() -> None:
    forged = FakePack(
        "partner.acme.forged",
        "acme.forged.synthetic",
        owner_id="acme",
        owner_display_name="Acme Fictional Company",
        contribution_kind="partner",
    )
    forged.metadata = replace(
        forged.metadata,
        default_spec="examples/contributions/acme/forged/operation.yaml",
    )
    type(forged).__module__ = "operatebench.contributions.acme.forged"
    try:
        with pytest.raises(OperationPackRegistryError, match="module binding"):
            OperationPackRegistry((forged,))
    finally:
        type(forged).__module__ = __name__


def test_partner_pack_accepts_exact_ephemeral_module_binding() -> None:
    module_name = "operatebench.contributions.acme.ephemeral"
    module = ModuleType(module_name)
    pack_type = type("EphemeralPack", (FakePack,), {"__module__": module_name})
    pack = pack_type(
        "partner.acme.ephemeral",
        "acme.ephemeral.synthetic",
        owner_id="acme",
        owner_display_name="Acme Fictional Company",
        contribution_kind="partner",
    )
    pack.metadata = replace(
        pack.metadata,
        default_spec="examples/contributions/acme/ephemeral/operation.yaml",
    )
    module.EphemeralPack = pack_type
    module.PACK = pack
    assert module_name not in sys.modules
    sys.modules[module_name] = module
    try:
        assert OperationPackRegistry((pack,)).resolve(pack.metadata.pack_id) is pack
    finally:
        assert sys.modules.pop(module_name) is module
    assert module_name not in sys.modules


def test_sdk_docs_runnable_commerce_example_uses_checkout_root() -> None:
    text = (REPO_ROOT / "docs" / "OPERATION_PACK_SDK.md").read_text(encoding="utf-8")
    runnable_anchor = (
        "The repository's completed Commerce pack demonstrates that transition."
    )
    scaffold_section, runnable_section = text.split(runnable_anchor, 1)

    scaffold_block = scaffold_section.split("```bash", 1)[1].split("```", 1)[0]
    scaffold_tokens = " ".join(
        line.removesuffix("\\").strip() for line in scaffold_block.strip().splitlines()
    ).split()
    hypothetical_pack_id = "partner.example_company.return_refund.synthetic"
    assert scaffold_tokens == [
        "operatebench",
        "init-operation",
        "--pack-id",
        hypothetical_pack_id,
        "--operation-id",
        "example_company_return_refund_synthetic_v1",
        "--owner-id",
        "example_company",
        "--owner-display-name",
        '"Example',
        "Company",
        'Ltd"',
        "--contribution-kind",
        "partner",
        "--destination",
        "return_refund",
    ]
    assert f"--pack-id {hypothetical_pack_id}.v1" not in scaffold_section
    assert "hypothetical, unregistered candidate" in scaffold_section
    assert "not expected to resolve through the registry" in scaffold_section

    introduction, fenced_commands = runnable_section.split("```bash", 1)
    command_block = fenced_commands.split("```", 1)[0]
    assert "From the root of a source checkout" in introduction
    assert (
        "# validate is static only: no scenario runs and no terminal is proven reachable."
        in command_block
    )
    assert "# Static only: no scenario runs" not in command_block

    uncommented_block = "\n".join(
        line for line in command_block.strip().splitlines() if not line.startswith("#")
    )
    command_paragraphs = uncommented_block.split("\n\n")
    assert len(command_paragraphs) == 4
    actual_commands = [
        " ".join(
            line.removesuffix("\\").strip() for line in paragraph.splitlines()
        ).split()
        for paragraph in command_paragraphs
    ]

    runnable_pack_id = "commerce.return_refund.synthetic.v1"
    fixture = "examples/operatebench/commerce_return_refund_v0_1.yaml"
    assert (REPO_ROOT / fixture).is_file()
    assert actual_commands == [
        [
            "uv",
            "run",
            "operatebench",
            "validate",
            "--pack",
            runnable_pack_id,
            fixture,
        ],
        [
            "uv",
            "run",
            "operatebench",
            "run",
            "--pack",
            runnable_pack_id,
            "--spec",
            fixture,
            "--scenario",
            "V1",
            "--agent",
            "reference",
            "--output",
            "return-v1.json",
        ],
        [
            "uv",
            "run",
            "operatebench",
            "replay",
            "--pack",
            runnable_pack_id,
            "--spec",
            fixture,
            "--run",
            "return-v1.json",
        ],
        [
            "uv",
            "run",
            "operatebench",
            "check",
            "--pack",
            runnable_pack_id,
            "--spec",
            fixture,
        ],
    ]
    pack_values = [
        command[command.index("--pack") + 1]
        for command in actual_commands
        if "--pack" in command
    ]
    assert hypothetical_pack_id not in pack_values


def test_sdk_docs_separate_registration_from_unavailable_evidence_admission() -> None:
    text = (REPO_ROOT / "docs" / "OPERATION_PACK_SDK.md").read_text(encoding="utf-8")

    assert "## Development registration" in text
    assert "## Evidence admission" in text
    assert "does not implement an evidence-admission route" in text
    assert "privacy and independent outcome review are recorded" not in text
