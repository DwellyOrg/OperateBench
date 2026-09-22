"""The named domain errors Core raises, and the root every caller can catch.

Nothing in this build is allowed to reach an operator as a raw traceback: an
unknown event type, a malformed timestamp, a duplicate identity and an actor
acting outside its authority are all *statements about the input*, and each one
gets a name that says which statement was false. The CLI catches
:class:`OperateBenchError` once, at the top, and prints it.

Domain packs subclass these rather than inventing a parallel hierarchy, so a
caller that wants "did this run fail because of the artefact or because of the
agent?" can answer it from the exception type alone.
"""

from __future__ import annotations


class OperateBenchError(Exception):
    """Root of every error this build raises deliberately."""


class SpecError(OperateBenchError):
    """An operation specification cannot be read as what it claims to be."""


class SpecFormatError(SpecError):
    """A specification file is not readable, not YAML, or not a mapping."""


class SpecSchemaError(SpecError):
    """A specification is well-formed YAML but violates the strict schema."""


class SpecIdentityError(SpecError):
    """A specification no longer hashes to the identity it claims.

    Raised when a spec's recorded digest and the digest re-derived from what it
    currently holds disagree — the residue of semantics changed after loading.
    An episode refuses rather than executing changed rules under a stale
    identity, because every artefact it wrote would then assert that identity as
    provenance for a run it does not describe.
    """


class UnknownFieldError(SpecSchemaError):
    """A specification declares a field this build does not know."""


class UnknownTypeError(SpecSchemaError):
    """A specification names an event, action or actor type that does not exist."""


class DuplicateIdentityError(SpecSchemaError):
    """A specification declares the same identity twice."""


class ClockError(OperateBenchError):
    """Simulated time was asked to do something it cannot do."""


class MalformedTimestampError(ClockError, SpecError):
    """A timestamp is not the canonical ``YYYY-MM-DDTHH:MM:SSZ`` instant."""


class ClockRewindError(ClockError):
    """Simulated time was asked to move backwards."""


class EventError(OperateBenchError):
    """An event cannot be queued, delivered or reduced as offered."""


class DuplicateEventError(EventError, DuplicateIdentityError):
    """Two events claim the same identity, so one of them cannot be recorded."""


class OutcomeError(OperateBenchError):
    """An agent outcome violates the Core outcome contract."""


class WaitContractError(OutcomeError):
    """A WAIT declares neither a wake condition nor a fallback deadline."""


class ActionError(OperateBenchError):
    """A proposed action was refused before it could commit."""


class AuthorityError(ActionError):
    """An actor proposed something its authority does not cover."""


class EvidenceError(ActionError):
    """An action cited evidence that does not establish what it must."""


class StateError(ActionError):
    """An action is well-formed and authorised but the state forbids it now."""


class DuplicateActionError(ActionError):
    """An action repeats an identity that has already committed."""


class TerminalGuardError(ActionError):
    """A proposed terminal transition does not satisfy its guard."""


class ArtifactError(OperateBenchError):
    """A run artefact cannot be read as the run it claims to record."""


class ReplayMismatchError(ArtifactError):
    """A replay reconstructed a different run from the same inputs."""


class LedgerPayloadError(OperateBenchError):
    """A trajectory payload is malformed at the ledger admission boundary."""


class LedgerFieldError(LedgerPayloadError):
    """A trajectory payload tries to supply a field owned by its ledger."""


class AgentRegistryError(OperateBenchError):
    """An agent was requested by a name this build does not ship."""


class OracleManifestError(OperateBenchError):
    """A oracle-declared oracle manifest cannot be read as what it claims to be.

    Kept apart from :class:`SpecError` because the two documents fail for
    different reasons and a caller should be able to tell them apart: a spec
    describes the case to execute, an oracle manifest describes what a control
    run against that case is required to produce.
    """


__all__ = [
    "ActionError",
    "AgentRegistryError",
    "ArtifactError",
    "AuthorityError",
    "ClockError",
    "ClockRewindError",
    "DuplicateActionError",
    "DuplicateEventError",
    "DuplicateIdentityError",
    "EventError",
    "EvidenceError",
    "LedgerFieldError",
    "LedgerPayloadError",
    "MalformedTimestampError",
    "OperateBenchError",
    "OracleManifestError",
    "OutcomeError",
    "ReplayMismatchError",
    "SpecError",
    "SpecFormatError",
    "SpecIdentityError",
    "SpecSchemaError",
    "StateError",
    "TerminalGuardError",
    "UnknownFieldError",
    "UnknownTypeError",
    "WaitContractError",
]
