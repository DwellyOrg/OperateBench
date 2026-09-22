"""Immutable lookup for reviewed, built-in operation packs.

Partner module binding is a defense-in-depth nominal check, not in-process
attestation or a Python sandbox. Source-path/AST checks and maintainer review
remain the admission authority.
"""

from __future__ import annotations

import keyword
import re
import sys
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace
from pathlib import PurePosixPath
from types import (
    BuiltinFunctionType,
    BuiltinMethodType,
    FunctionType,
    GetSetDescriptorType,
    MappingProxyType,
    MethodType,
    ModuleType,
)

from operatebench.jsonsafe import JsonSafetyError, check_text, ensure_json_safe
from operatebench.sdk._validation import (
    canonical_owner_display_name_key,
    is_operation_identity,
    is_pack_version,
    is_safe_display_name,
    is_safe_owner_display_name,
)
from operatebench.sdk.api import (
    PACK_API_VERSION,
    PACK_CAPABILITIES,
    OperationPack,
    OperationPackMetadata,
)
from operatebench.sdk.errors import (
    OperationPackRegistryError,
    UnknownOperationPackError,
)

_STATUSES = frozenset(("development", "incubator"))
_CONTRIBUTION_KINDS = frozenset(("maintainer", "partner"))
_OWNER_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
_WINDOWS_RESERVED = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
)
_TYPE_NAMESPACE_DESCRIPTOR = type.__dict__["__dict__"]
_TYPE_NAME_DESCRIPTOR = type.__dict__["__name__"]
_TYPE_MRO_DESCRIPTOR = type.__dict__["__mro__"]
_MISSING = object()


def _problem(message: str) -> OperationPackRegistryError:
    return OperationPackRegistryError(
        f"invalid static operation-pack registry: {message}"
    )


def _static_attribute(candidate: object, name: str) -> object:
    """Read candidate shape without entering candidate-controlled lookup code."""

    candidate_type = type(candidate)
    mro = _TYPE_MRO_DESCRIPTOR.__get__(candidate_type, type)
    class_value = _MISSING
    for base in mro:
        namespace = _TYPE_NAMESPACE_DESCRIPTOR.__get__(base, type)
        value = namespace.get(name, _MISSING)
        if value is not _MISSING:
            class_value = value
            break
    if class_value is not _MISSING and _is_data_descriptor(class_value):
        return class_value
    for base in mro:
        namespace = _TYPE_NAMESPACE_DESCRIPTOR.__get__(base, type)
        dictionary_descriptor = namespace.get("__dict__", _MISSING)
        if type(dictionary_descriptor) is GetSetDescriptorType:
            instance_namespace = dictionary_descriptor.__get__(candidate, candidate_type)
            value = instance_namespace.get(name, _MISSING)
            if value is not _MISSING:
                return value
            break
    if class_value is not _MISSING:
        return class_value
    raise _problem(f"operation-pack candidate is missing required {name}")


def _is_data_descriptor(value: object) -> bool:
    value_type = type(value)
    for base in _TYPE_MRO_DESCRIPTOR.__get__(value_type, type):
        namespace = _TYPE_NAMESPACE_DESCRIPTOR.__get__(base, type)
        if "__set__" in namespace or "__delete__" in namespace:
            return True
    return False


def _candidate_class_identity(candidate: object) -> tuple[type[object], str, str]:
    """Read exact class identity without invoking its metaclass lookup hooks."""

    candidate_type = type(candidate)
    namespace = _TYPE_NAMESPACE_DESCRIPTOR.__get__(candidate_type, type)
    module = namespace.get("__module__", _MISSING)
    name = _TYPE_NAME_DESCRIPTOR.__get__(candidate_type, type)
    namespace_name = namespace.get("__name__", _MISSING)
    metaclass = type(candidate_type)
    for base in _TYPE_MRO_DESCRIPTOR.__get__(metaclass, type):
        if base is type:
            break
        metaclass_namespace = _TYPE_NAMESPACE_DESCRIPTOR.__get__(base, type)
        for field in ("__module__", "__name__"):
            override = metaclass_namespace.get(field, _MISSING)
            if override is not _MISSING and _is_data_descriptor(override):
                raise _problem(
                    f"operation-pack candidate metaclass overrides class {field}"
                )
    if (
        type(module) is not str
        or not module
        or any(
            not component.isidentifier() or keyword.iskeyword(component)
            for component in module.split(".")
        )
    ):
        raise _problem(
            "operation-pack candidate class __module__ must be an exact safe dotted str"
        )
    if (
        type(name) is not str
        or not name.isidentifier()
        or keyword.iskeyword(name)
        or (
            namespace_name is not _MISSING
            and (type(namespace_name) is not str or namespace_name != name)
        )
    ):
        raise _problem(
            "operation-pack candidate class __name__ must be an exact safe identifier"
        )
    return candidate_type, module, name


def _is_static_callable(value: object) -> bool:
    """Admit ordinary callable storage forms, but never a custom descriptor."""

    if type(value) in {
        FunctionType,
        MethodType,
        BuiltinFunctionType,
        BuiltinMethodType,
    }:
        return True
    if type(value) in {staticmethod, classmethod}:
        return callable(object.__getattribute__(value, "__func__"))
    return False


def _require_id(value: str, where: str) -> None:
    if not is_operation_identity(value):
        raise _problem(
            f"{where} must be 1-128 lowercase ASCII letters/digits separated by "
            "'.', '_' or '-', starting with a letter"
        )


def _validate_default_spec(
    value: str | None, pack_id: str, owner_id: str, contribution_kind: str
) -> None:
    if value is None:
        if contribution_kind == "partner":
            raise _problem(f"default spec for partner pack {pack_id!r} is required")
        return
    if type(value) is not str:
        raise _problem(f"default spec for pack {pack_id!r} must be text or null")
    try:
        check_text(value, f"default spec for pack {pack_id!r}")
    except JsonSafetyError as exc:
        raise _problem(f"default_spec for pack {pack_id!r} is not valid text") from exc
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value in ("", ".")
        or ".." in path.parts
        or "\\" in value
        or value != path.as_posix()
    ):
        raise _problem(
            f"default spec for pack {pack_id!r} must be a normalized relative path"
        )
    if contribution_kind == "partner":
        prefix = f"examples/contributions/{owner_id}/"
        if not value.startswith(prefix):
            raise _problem(
                f"default spec for partner pack {pack_id!r} must be beneath {prefix!r}"
            )


def _validate_metadata(metadata: OperationPackMetadata) -> None:
    _require_id(metadata.pack_id, "pack_id")
    _require_id(metadata.operation_type, f"operation_type for {metadata.pack_id!r}")
    _require_id(metadata.operation_id, f"operation_id for {metadata.pack_id!r}")
    if not is_pack_version(metadata.pack_version):
        raise _problem(
            f"pack_version for {metadata.pack_id!r} must be a three-part version"
        )
    if type(metadata.display_name) is str:
        try:
            check_text(
                metadata.display_name,
                f"display_name for pack {metadata.pack_id!r}",
            )
        except JsonSafetyError as exc:
            raise _problem(
                f"display_name for pack {metadata.pack_id!r} is not valid text"
            ) from exc
    if not is_safe_display_name(metadata.display_name):
        raise _problem(
            f"display_name for {metadata.pack_id!r} must be non-empty safe text "
            "of at most 128 characters"
        )
    if (
        type(metadata.owner_id) is not str
        or not _OWNER_ID_RE.fullmatch(metadata.owner_id)
        or keyword.iskeyword(metadata.owner_id)
        or metadata.owner_id in _WINDOWS_RESERVED
    ):
        raise _problem(
            f"owner_id for {metadata.pack_id!r} must be a 1-63 character lowercase "
            "ASCII Python-safe identifier that is not reserved"
        )
    if not is_safe_owner_display_name(metadata.owner_display_name):
        raise _problem(
            f"owner_display_name for {metadata.pack_id!r} must be non-empty safe "
            "text of at most 128 characters"
        )
    if (
        type(metadata.contribution_kind) is not str
        or metadata.contribution_kind not in _CONTRIBUTION_KINDS
    ):
        raise _problem(
            f"contribution_kind for {metadata.pack_id!r} must be one of "
            f"{sorted(_CONTRIBUTION_KINDS)}"
        )
    if metadata.contribution_kind == "partner":
        prefix = f"partner.{metadata.owner_id}."
        if not metadata.pack_id.startswith(prefix):
            raise _problem(
                f"partner pack {metadata.pack_id!r} must start with {prefix!r}"
            )
    if type(metadata.status) is not str or metadata.status not in _STATUSES:
        raise _problem(
            f"status for {metadata.pack_id!r} must be one of {sorted(_STATUSES)}"
        )
    if metadata.privacy_status != "SYNTHETIC_ONLY":
        raise _problem(
            f"pack {metadata.pack_id!r} declares privacy status "
            f"{metadata.privacy_status!r}; SDK v2 admits SYNTHETIC_ONLY only"
        )
    if type(metadata.evidence_eligible) is not bool or metadata.evidence_eligible:
        raise _problem(
            f"pack {metadata.pack_id!r} sets evidence_eligible=True; SDK v2 is a "
            "development dispatch layer and cannot confer research admission"
        )
    if (
        type(metadata.sdk_api_version) is not int
        or metadata.sdk_api_version != PACK_API_VERSION
    ):
        raise _problem(
            f"pack {metadata.pack_id!r} uses SDK API {metadata.sdk_api_version}; "
            f"this build supports {PACK_API_VERSION}"
        )
    if type(metadata.capabilities) is not tuple or metadata.capabilities != (
        PACK_CAPABILITIES
    ):
        raise _problem(
            f"pack {metadata.pack_id!r} capabilities must be exactly "
            f"{list(PACK_CAPABILITIES)} in SDK v2"
        )
    if type(metadata.agent_ids) is not tuple or not metadata.agent_ids:
        raise _problem(f"agent_ids for {metadata.pack_id!r} must be non-empty and unique")
    for agent_id in metadata.agent_ids:
        _require_id(agent_id, f"agent id in {metadata.pack_id!r}")
    if len(set(metadata.agent_ids)) != len(metadata.agent_ids):
        raise _problem(f"agent_ids for {metadata.pack_id!r} must be non-empty and unique")
    if tuple(sorted(metadata.agent_ids)) != metadata.agent_ids:
        raise _problem(f"agent_ids for {metadata.pack_id!r} must be sorted")
    if type(metadata.aliases) is not tuple:
        raise _problem(f"aliases for {metadata.pack_id!r} must be an immutable tuple")
    for alias in metadata.aliases:
        _require_id(alias, f"alias for {metadata.pack_id!r}")
    if len(set(metadata.aliases)) != len(metadata.aliases):
        raise _problem(f"aliases for {metadata.pack_id!r} contain a duplicate")
    _validate_default_spec(
        metadata.default_spec,
        metadata.pack_id,
        metadata.owner_id,
        metadata.contribution_kind,
    )
    try:
        ensure_json_safe(metadata.as_dict(), f"metadata for pack {metadata.pack_id!r}")
    except JsonSafetyError as exc:
        raise _problem(
            f"metadata projection for pack {metadata.pack_id!r} is not JSON-safe"
        ) from exc


class OperationPackRegistry:
    """A deterministic registry with no mutation or discovery surface."""

    def __init__(self, packs: Iterable[OperationPack]) -> None:
        canonical: dict[str, OperationPack] = {}
        names: dict[str, OperationPack] = {}
        identity_owners: dict[str, tuple[str, str]] = {}
        owners: dict[str, tuple[str, str]] = {}
        display_names: dict[str, str] = {}
        snapshots: dict[str, OperationPackMetadata] = {}
        pack_ids: dict[int, str] = {}
        for pack in packs:
            metadata = _static_attribute(pack, "metadata")
            if type(metadata) is not OperationPackMetadata:
                raise _problem(
                    "operation-pack candidate metadata is not the expected exact "
                    "OperationPackMetadata type"
                )
            _validate_metadata(metadata)
            snapshot = replace(metadata)
            pack_type, implementation_module, pack_type_name = _candidate_class_identity(
                pack
            )
            for capability in PACK_CAPABILITIES:
                static_capability = _static_attribute(pack, capability)
                if not _is_static_callable(static_capability):
                    raise _problem(
                        f"operation-pack candidate capability {capability!r} must be "
                        "callable"
                    )
            if metadata.contribution_kind == "partner":
                expected_module = f"operatebench.contributions.{metadata.owner_id}."
                if not implementation_module.startswith(expected_module):
                    raise _problem(
                        f"partner pack {metadata.pack_id!r} implementation module "
                        f"{implementation_module!r} must be beneath {expected_module!r}"
                    )
            identity = (metadata.owner_display_name, metadata.contribution_kind)
            prior_identity = owners.get(metadata.owner_id)
            if prior_identity is not None and prior_identity != identity:
                raise _problem(
                    f"owner_id {metadata.owner_id!r} has inconsistent display name "
                    "or contribution kind"
                )
            folded_display = canonical_owner_display_name_key(metadata.owner_display_name)
            prior_owner = display_names.get(folded_display)
            if prior_owner is not None and prior_owner != metadata.owner_id:
                raise _problem(
                    f"owner display name {metadata.owner_display_name!r} is claimed "
                    f"by both {prior_owner!r} and {metadata.owner_id!r}"
                )
            owners[metadata.owner_id] = identity
            display_names[folded_display] = metadata.owner_id
            if metadata.contribution_kind == "partner":
                implementation = sys.modules.get(implementation_module)
                if (
                    type(implementation) is not ModuleType
                    or implementation.__dict__.get("PACK") is not pack
                    or implementation.__dict__.get(pack_type_name) is not pack_type
                ):
                    raise _problem(
                        f"partner pack {metadata.pack_id!r} has no exact runtime "
                        "module binding; source-path and AST review remain the "
                        "admission authority"
                    )
            if metadata.pack_id in canonical:
                raise _problem(f"duplicate canonical pack id {metadata.pack_id!r}")
            claims = (
                (metadata.pack_id, "pack_id"),
                (metadata.operation_type, "operation_type"),
                *((alias, "alias") for alias in metadata.aliases),
                (metadata.operation_id, "operation_id"),
            )
            for value, kind in claims:
                prior_identity_owner = identity_owners.get(value)
                if prior_identity_owner is None:
                    identity_owners[value] = (metadata.pack_id, kind)
                    continue
                prior_pack_id, prior_kind = prior_identity_owner
                compatible_own_kinds = {prior_kind, kind} == {
                    "pack_id",
                    "operation_type",
                } or {prior_kind, kind} == {"alias", "operation_type"}
                if prior_pack_id == metadata.pack_id and compatible_own_kinds:
                    continue
                if prior_kind == kind:
                    raise _problem(
                        f"{kind} {value!r} is claimed by both {prior_pack_id!r} "
                        f"and {metadata.pack_id!r} as {kind}"
                    )
                prior_label = (
                    f"lookup name ({prior_kind})"
                    if prior_kind in {"pack_id", "alias"}
                    else prior_kind
                )
                kind_label = (
                    f"lookup name ({kind})" if kind in {"pack_id", "alias"} else kind
                )
                raise _problem(
                    f"identity {value!r} is {prior_label} for {prior_pack_id!r} "
                    f"and {kind_label} for {metadata.pack_id!r}"
                )
            canonical[metadata.pack_id] = pack
            snapshots[metadata.pack_id] = snapshot
            pack_ids[id(pack)] = metadata.pack_id
            for name in (metadata.pack_id, *metadata.aliases):
                prior = names.get(name)
                if prior is not None:
                    prior_pack_id = pack_ids[id(prior)]
                    raise _problem(
                        f"lookup name {name!r} is claimed by both "
                        f"{prior_pack_id!r} and {metadata.pack_id!r}"
                    )
                names[name] = pack
        self._canonical: Mapping[str, OperationPack] = MappingProxyType(
            dict(sorted(canonical.items()))
        )
        self._names: Mapping[str, OperationPack] = MappingProxyType(dict(names))
        self._owners = MappingProxyType(dict(sorted(owners.items())))
        self._metadata: Mapping[str, OperationPackMetadata] = MappingProxyType(
            dict(sorted(snapshots.items()))
        )
        self._pack_ids: Mapping[int, str] = MappingProxyType(dict(pack_ids))

    def resolve(self, name: str) -> OperationPack:
        pack = self._names.get(name)
        if pack is None:
            raise UnknownOperationPackError(
                f"operation pack {name!r} is not statically registered; available "
                f"packs are {list(self.canonical_ids())}"
            )
        snapshot = self._metadata[self._pack_ids[id(pack)]]
        try:
            current = _static_attribute(pack, "metadata")
        except OperationPackRegistryError:
            current = None
        if type(current) is not OperationPackMetadata or current != snapshot:
            raise OperationPackRegistryError(
                f"operation pack {snapshot.pack_id!r} metadata drifted after registry "
                "construction; dispatch refused"
            )
        return pack

    def canonical_ids(self) -> tuple[str, ...]:
        return tuple(self._canonical)

    def metadata(self) -> tuple[OperationPackMetadata, ...]:
        return tuple(self._metadata.values())

    def owner_ids(self) -> tuple[str, ...]:
        return tuple(self._owners)

    def metadata_for_owner(self, owner_id: str) -> tuple[OperationPackMetadata, ...]:
        if type(owner_id) is not str:
            raise TypeError("owner_id must be an exact str")
        if (
            not _OWNER_ID_RE.fullmatch(owner_id)
            or keyword.iskeyword(owner_id)
            or owner_id in _WINDOWS_RESERVED
        ):
            raise ValueError("owner_id must be a safe lowercase identifier")
        return tuple(item for item in self.metadata() if item.owner_id == owner_id)

    def __iter__(self) -> Iterator[OperationPack]:
        return iter(self._canonical.values())

    def __len__(self) -> int:
        return len(self._canonical)


__all__ = ["OperationPackRegistry"]
