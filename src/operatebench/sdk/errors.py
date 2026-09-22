"""Named failures at the static operation-pack boundary."""

from __future__ import annotations

from operatebench.core.errors import OperateBenchError


class OperationPackError(OperateBenchError):
    """An operation pack cannot be selected or used as declared."""


class OperationPackRegistryError(OperationPackError):
    """The immutable built-in pack registry is internally inconsistent."""


class UnknownOperationPackError(OperationPackError):
    """A command names no statically registered operation pack."""


class OperationPackMismatchError(OperationPackError):
    """A pack was offered a spec for another operation type."""


class OperationScaffoldError(OperationPackError):
    """A contributor scaffold cannot be created safely."""


__all__ = [
    "OperationPackError",
    "OperationPackMismatchError",
    "OperationPackRegistryError",
    "OperationScaffoldError",
    "UnknownOperationPackError",
]
