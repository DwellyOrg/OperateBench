"""The append-only trajectory ledger, and the record types it distinguishes.

The distinctions in :data:`RECORD_TYPES` are the point of the file. A model
outcome, the validation that outcome had to survive, the state transition that
was actually accepted and the external side effect that followed the commit are
four different facts about one moment. Collapsing them would make the two
questions this benchmark exists to answer — *was this accepted, and on whose
authority?* — unanswerable from the artefact.

Append-only in the strict sense: :meth:`TrajectoryLedger.append` is the only
mutator, indices are assigned by it, and :meth:`digest` is taken over the
canonical encoding of the whole sequence, so a reordering is a different run.
"""

from __future__ import annotations

import gc
import hashlib
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, cast

from operatebench.core.errors import LedgerFieldError, LedgerPayloadError
from operatebench.jsonsafe import (
    _to_json_preserving_tuples_with_root_key_admission,
    _type_mro,
    canonical_json_bytes,
    to_json_preserving_tuples,
)

_LEDGER_OWNED_FIELDS = ("index", "at", "record_type")
_LEDGER_OWNED_FIELD_MESSAGE = "trajectory payload contains a ledger-owned field"
_MALFORMED_PAYLOAD_MESSAGE = "trajectory payload must be a mapping"
_MAPPING_PROXY_TYPE: type[Any] = type(MappingProxyType({}))


def _admit_payload_key(key: str) -> None:
    if key in _LEDGER_OWNED_FIELDS:
        raise LedgerFieldError(_LEDGER_OWNED_FIELD_MESSAGE)


def _has_concrete_mapping_shape(payload: object) -> bool:
    """Admit only real dict/Mapping inheritance, without caller or ABC hooks."""
    payload_type = type(payload)
    if payload_type is _MAPPING_PROXY_TYPE:
        # CPython exposes no public mappingproxy unwrap. Its GC traversal has one
        # referent: the wrapped mapping. Fail closed if that runtime invariant
        # changes, and inspect only the direct referent so proxy chains cannot
        # launder an otherwise inadmissible mapping shape.
        referents = gc.get_referents(payload)
        if len(referents) != 1:
            return False
        payload_type = type(referents[0])
    return any(base is dict or base is Mapping for base in _type_mro(payload_type))


#: What a tool-mediated read leaves behind: one provenance row per result, one
#: row for a batch that was refused before anything was served, and one row each
#: time derived context stopped being usable. Named as their own tuple because a
#: reader has to be able to ask which record types a *contract version* wrote,
#: and the answer for every contract before 5 is "these three, none of them".
RETRIEVAL_RECORD_TYPES: tuple[str, ...] = (
    "retrieval_served",
    "retrieval_refused",
    "derived_context_invalidated",
)

#: Every record type this build writes. Grouped by what they are evidence of.
RECORD_TYPES: tuple[str, ...] = (
    # The world moving, and whether it was allowed to.
    "event_observed",
    "event_audit_only",
    "event_rejected",
    "event_after_terminal",
    # The agent being asked, and what it said.
    "agent_invoked",
    "agent_outcome",
    # An outcome the engine could not read as one of the five. Kept apart from
    # ``action_rejected``: nothing was proposed, so there is no proposal for a
    # refusal to answer, and the record says so rather than inventing one.
    "outcome_rejected",
    "wait_declared",
    "wait_rejected",
    # A declared wait was still standing when the operational horizon ran out.
    # Written when the episode ends, not when the wait was declared: whether an
    # event was ever coming is not a fact anyone holds at declaration time.
    "wait_unresolved_at_horizon",
    # Proposal → validation → accepted effect → post-commit side effect.
    "action_proposed",
    "action_rejected",
    "effect_accepted",
    "side_effect",
    "side_effect_failed",
    # Human checkpoints.
    "checkpoint_opened",
    "checkpoint_resolved",
    "checkpoint_rejected",
    # Obligations and timers.
    "obligation_created",
    "obligation_discharged",
    "obligation_cancelled",
    "obligation_breached",
    "timer_scheduled",
    "timer_cancelled",
    # Terminal handling.
    "terminal_proposed",
    "terminal_rejected",
    "provisional_close_entered",
    "operation_reopened",
    "terminal_accepted",
    "episode_ended",
    # Invariants.
    "critical_violation",
    # Tool-mediated reads. Appended rather than interleaved so the vocabulary a
    # pre-retrieval artefact contract was written under is the prefix of this
    # one and can be derived from it rather than restated.
    *RETRIEVAL_RECORD_TYPES,
)


class TrajectoryLedger:
    """An append-only sequence of typed records with a canonical digest."""

    def __init__(
        self, observer: Callable[[Mapping[str, Any]], None] | None = None
    ) -> None:
        self._observer = observer
        self._records: list[dict[str, Any]] = []
        self._append_in_progress = False

    def append(
        self, at: str, record_type: str, payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Write one record and hand back the detached row that was written."""
        if self._append_in_progress:
            raise RuntimeError("trajectory append already in progress")
        self._append_in_progress = True
        try:
            if payload is not None and not _has_concrete_mapping_shape(payload):
                raise LedgerPayloadError(_MALFORMED_PAYLOAD_MESSAGE)
            record: dict[str, Any] = {
                "index": len(self._records),
                "at": to_json_preserving_tuples(at),
                "record_type": to_json_preserving_tuples(record_type),
            }
            if payload is not None:
                projected_payload = _to_json_preserving_tuples_with_root_key_admission(
                    payload, _admit_payload_key
                )
                if type(projected_payload) is not dict:
                    raise LedgerPayloadError(_MALFORMED_PAYLOAD_MESSAGE)
                record.update(cast(dict[str, Any], projected_payload))
            canonical_json_bytes([record], "operatebench trajectory")
            self._records.append(record)
            if self._observer is not None:
                self._observer(cast(dict[str, Any], to_json_preserving_tuples(record)))
            return cast(dict[str, Any], to_json_preserving_tuples(record))
        finally:
            self._append_in_progress = False

    def as_list(self) -> list[dict[str, Any]]:
        """A detached projection of the whole trajectory."""
        return cast(list[dict[str, Any]], to_json_preserving_tuples(self._records))

    def digest(self) -> str:
        """SHA-256 over the canonical encoding of the whole trajectory."""
        return hashlib.sha256(
            canonical_json_bytes(self.as_list(), "operatebench trajectory")
        ).hexdigest()

    def __len__(self) -> int:
        return len(self._records)


__all__ = ["RECORD_TYPES", "RETRIEVAL_RECORD_TYPES", "TrajectoryLedger"]
