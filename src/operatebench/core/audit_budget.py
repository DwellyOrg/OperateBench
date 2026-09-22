# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0

"""Core-local atomic budgeting for domain-generated audit intents.

This private primitive owns only the assigned-runtime refusal rows in the frozen
registry. Source-content admission, the six pre-identity rows, successor
integration, and Engine transactionality remain with their declared owners.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import re
import secrets
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Generic, Literal, NoReturn, TypeVar

_CONTRACT_ID = "operatebench.domain-generated-audit-budget.v1"
_INTENT_DIGEST_DOMAIN = "operatebench.digest.domain-generated-audit-intent.v1"
_COUNTED_KINDS = frozenset(
    {"scheduled_audit_event_intent", "immediate_audit_event_intent"}
)
_PORTABLE_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_INSTANCE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9._:-]{0,254}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_INT64_MAX = 9_223_372_036_854_775_807
_V1_CAPACITY = 99_998


def _exact_int(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact built-in int")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside {minimum}..{maximum}")
    return value


def _portable_id(value: object, name: str, *, instance: bool = False) -> str:
    pattern = _INSTANCE_ID if instance else _PORTABLE_ID
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact built-in str")
    if len(value) > 255 or pattern.fullmatch(value) is None:
        raise ValueError(f"{name} is not a portable identifier")
    return value


def _checked_add(left: object, right: object) -> int:
    """Add two exact non-negative signed-int64 values or fail closed."""

    lhs = _exact_int(left, "left", 0, _INT64_MAX)
    rhs = _exact_int(right, "right", 0, _INT64_MAX)
    if lhs > _INT64_MAX - rhs:
        raise OverflowError("checked signed int64 addition overflow")
    return lhs + rhs


def _checked_multiply(left: object, right: object) -> int:
    """Multiply two exact non-negative signed-int64 values or fail closed."""

    lhs = _exact_int(left, "left", 0, _INT64_MAX)
    rhs = _exact_int(right, "right", 0, _INT64_MAX)
    if lhs and rhs > _INT64_MAX // lhs:
        raise OverflowError("checked signed int64 multiplication overflow")
    return lhs * rhs


@dataclass(frozen=True, slots=True)
class AuditBudgetSource:
    source_id: str
    counted_kind: str
    max_accepted_occurrences: int
    max_events_per_transition: int

    def __post_init__(self) -> None:
        _portable_id(self.source_id, "source_id")
        if type(self.counted_kind) is not str or self.counted_kind not in _COUNTED_KINDS:
            raise ValueError("counted_kind is not in the closed v1 taxonomy")
        occurrences = _exact_int(
            self.max_accepted_occurrences, "max_accepted_occurrences", 0, _INT64_MAX
        )
        fanout = _exact_int(
            self.max_events_per_transition, "max_events_per_transition", 1, _INT64_MAX
        )
        try:
            _checked_multiply(occurrences, fanout)
        except OverflowError as exc:
            raise ValueError("source maximum product exceeds signed int64") from exc
        if occurrences > _V1_CAPACITY or fanout > _V1_CAPACITY:
            raise ValueError("source maximum exceeds protocol v1 capacity")


@dataclass(frozen=True, slots=True)
class AuditBudgetDeclaration:
    contract_id: str
    operation_core_content_digest_sha256: str
    operation_instance_id: str
    declared_max_events: int
    sources: tuple[AuditBudgetSource, ...]

    def __post_init__(self) -> None:
        if type(self.contract_id) is not str or self.contract_id != _CONTRACT_ID:
            raise ValueError("unsupported or wrong audit-budget contract")
        if (
            type(self.operation_core_content_digest_sha256) is not str
            or _DIGEST.fullmatch(self.operation_core_content_digest_sha256) is None
        ):
            raise ValueError("operation Core content digest must be lowercase SHA-256")
        _portable_id(self.operation_instance_id, "operation_instance_id", instance=True)
        declared = _exact_int(
            self.declared_max_events, "declared_max_events", 0, _V1_CAPACITY
        )
        if type(self.sources) is not tuple:
            raise TypeError("sources must be an exact tuple")
        if len(self.sources) > 4096 or any(
            type(source) is not AuditBudgetSource for source in self.sources
        ):
            raise ValueError("sources must contain at most 4096 exact source rows")
        source_ids = tuple(source.source_id for source in self.sources)
        if source_ids != tuple(sorted(source_ids)) or len(set(source_ids)) != len(
            source_ids
        ):
            raise ValueError("source IDs must be sorted and unique")
        total = 0
        for source in self.sources:
            source.__post_init__()
            try:
                product = _checked_multiply(
                    source.max_accepted_occurrences, source.max_events_per_transition
                )
                total = _checked_add(total, product)
            except OverflowError as exc:
                raise ValueError("declaration aggregate exceeds signed int64") from exc
            if total > _V1_CAPACITY:
                raise ValueError("declaration aggregate exceeds protocol v1 capacity")
        if total != declared:
            raise ValueError("declared maximum does not equal checked source aggregate")


@dataclass(frozen=True, slots=True)
class AuditIntent:
    """Exact domain-supplied identity; payloads never enter audit authority."""

    intent_id: str

    def __post_init__(self) -> None:
        _portable_id(self.intent_id, "intent_id")


@dataclass(frozen=True, slots=True)
class CommittedAuditIntent:
    record_type: Literal["committed_intent"]
    contract_id: str
    intent_digest_sha256: str
    intent_id: str
    source_id: str
    counted_kind: str
    transition_id: str
    batch_ordinal: int
    operation_committed_count: int


@dataclass(frozen=True, slots=True)
class AuditIntegrityFinding:
    record_type: Literal["integrity_finding"]
    contract_id: str
    integrity_code: str
    refusal_class: str
    source_id: str | None
    requested_count: int
    operation_committed_before: int
    source_committed_before: int | None
    declared_max_events: int
    transition_id: str | None
    invocation_index: int | None


@dataclass(frozen=True, slots=True)
class AuditTerminalSummary:
    record_type: Literal["terminal_summary"]
    contract_id: str
    declared_max_events: int
    committed_count: int
    result: Literal["SUCCESS", "ERROR"]
    integrity_code: str | None


@dataclass(frozen=True, slots=True)
class AuditBudgetSnapshot:
    operation_committed_count: int
    source_committed_counts: tuple[tuple[str, int], ...]
    source_occurrence_counts: tuple[tuple[str, int], ...]
    committed_intents: tuple[CommittedAuditIntent, ...]
    final: bool
    halted: bool


@dataclass(slots=True)
class AuditIntegrityHalt(Exception):
    """Fresh bounded outward integrity stop; the runtime retains no exception."""

    contract_id: str
    integrity_code: str
    refusal_class: str
    source_id: str | None
    requested_count: int
    operation_committed_before: int
    source_committed_before: int | None
    declared_max_events: int
    transition_id: str | None
    invocation_index: int | None

    def __copy__(self) -> AuditIntegrityHalt:
        raise TypeError("audit integrity halts cannot be copied")

    def __deepcopy__(self, memo: object) -> AuditIntegrityHalt:
        raise TypeError("audit integrity halts cannot be deep-copied")

    def __reduce__(self) -> tuple[object, ...]:
        raise TypeError("audit integrity halts cannot be pickled")


@dataclass(frozen=True, slots=True)
class _HaltWitness:
    cause: str
    facts: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _HaltRecord:
    integrity_code: str
    refusal_class: str
    source_id: str | None
    requested_count: int
    operation_committed_before: int
    source_committed_before: int | None
    transition_id: str | None
    invocation_index: int | None
    witness: _HaltWitness


@dataclass(frozen=True, slots=True)
class _ActiveHaltFacts:
    transition_id: str
    invocation_index: int
    identity: str
    batches: tuple[tuple[str, str, int, int], ...]


@dataclass(frozen=True, slots=True)
class _HaltAuthority:
    """Process-local provenance minted while publishing the first stop."""

    record: _HaltRecord
    malformed_record: _HaltRecord
    cause: str
    nonce: str
    final_before: bool
    source_counts_before: _PersistentMap[int]
    source_occurrences_before: _PersistentMap[int]
    active_facts: _ActiveHaltFacts | None


@dataclass(frozen=True, slots=True)
class AuditReservation:
    """Opaque process-local authority; all mutable consume state stays in runtime."""

    _runtime: AuditBudgetRuntime
    _transition_identity: str
    _nonce: str
    _creator_pid: int

    @property
    def process_id(self) -> int:
        return self._creator_pid

    def __copy__(self) -> AuditReservation:
        raise TypeError("audit reservations cannot be copied")

    def __deepcopy__(self, memo: object) -> AuditReservation:
        raise TypeError("audit reservations cannot be deep-copied")

    def __reduce__(self) -> tuple[object, ...]:
        raise TypeError("audit reservations cannot be pickled")

    def consume(self, intent_id: str) -> None:
        if self._creator_pid != os.getpid():
            raise RuntimeError("audit reservation belongs to another process")
        self._runtime._consume(self, intent_id)


@dataclass(frozen=True, slots=True, eq=False)
class AuditTransitionScope:
    _runtime: AuditBudgetRuntime
    _transition_id: str
    _invocation_index: int
    _identity: str
    _creator_pid: int

    def __enter__(self) -> AuditTransitionScope:
        self._ensure_pid()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self._ensure_pid()
        self._runtime._abort_if_active(self)
        return False

    def _ensure_pid(self) -> None:
        if self._creator_pid != os.getpid():
            raise RuntimeError("audit transition scope belongs to another process")

    def __copy__(self) -> AuditTransitionScope:
        raise TypeError("audit transition scopes cannot be copied")

    def __deepcopy__(self, memo: object) -> AuditTransitionScope:
        raise TypeError("audit transition scopes cannot be deep-copied")

    def __reduce__(self) -> tuple[object, ...]:
        raise TypeError("audit transition scopes cannot be pickled")

    def reserve_batch(
        self, source_id: str, counted_kind: str, intent_ids: tuple[str, ...]
    ) -> AuditReservation:
        self._ensure_pid()
        return self._runtime._reserve(self, source_id, counted_kind, intent_ids)

    def consume(self, reservation: AuditReservation, intent_id: str) -> None:
        self._ensure_pid()
        self._runtime._consume_from_scope(self, reservation, intent_id)

    def commit(self) -> None:
        self._ensure_pid()
        self._runtime._commit(self)

    def abort(self) -> None:
        self._ensure_pid()
        self._runtime._abort(self)


@dataclass(frozen=True, slots=True)
class _BatchState:
    source: AuditBudgetSource
    reservation: AuditReservation
    nonce: str
    intents: tuple[str, ...]
    digests: tuple[str, ...]
    next_index: int


@dataclass(frozen=True, slots=True)
class _TransitionState:
    scope: AuditTransitionScope
    transition_id: str
    invocation_index: int
    identity: str
    batches: tuple[_BatchState, ...]
    reexecution: bool = False


_V = TypeVar("_V")


@dataclass(frozen=True, slots=True)
class _TreapNode(Generic[_V]):
    key: str
    value: _V
    priority: bytes
    left: _TreapNode[_V] | None
    right: _TreapNode[_V] | None
    size: int


def _node(
    key: str,
    value: _V,
    priority: bytes,
    left: _TreapNode[_V] | None,
    right: _TreapNode[_V] | None,
) -> _TreapNode[_V]:
    return _TreapNode(
        key,
        value,
        priority,
        left,
        right,
        1 + (left.size if left else 0) + (right.size if right else 0),
    )


def _treap_priority(key: str) -> bytes:
    return hashlib.sha256(b"operatebench.audit.treap.v1\0" + key.encode()).digest()


def _treap_insert(
    root: _TreapNode[_V] | None, key: str, value: _V
) -> tuple[_TreapNode[_V], bool]:
    """Persistent deterministic treap insertion; return root and whether key was new."""

    priority = _treap_priority(key)

    def insert(current: _TreapNode[_V] | None) -> tuple[_TreapNode[_V], bool]:
        _treap_visit()
        if current is None:
            return _node(key, value, priority, None, None), True
        if key == current.key:
            return _node(key, value, current.priority, current.left, current.right), False
        if key < current.key:
            left, added = insert(current.left)
            candidate = _node(
                current.key,
                current.value,
                current.priority,
                left,
                current.right,
            )
            if left.priority < candidate.priority:
                candidate = _node(
                    left.key,
                    left.value,
                    left.priority,
                    left.left,
                    _node(
                        candidate.key,
                        candidate.value,
                        candidate.priority,
                        left.right,
                        candidate.right,
                    ),
                )
            return candidate, added
        right, added = insert(current.right)
        candidate = _node(
            current.key,
            current.value,
            current.priority,
            current.left,
            right,
        )
        if right.priority < candidate.priority:
            candidate = _node(
                right.key,
                right.value,
                right.priority,
                _node(
                    candidate.key,
                    candidate.value,
                    candidate.priority,
                    candidate.left,
                    right.left,
                ),
                right.right,
            )
        return candidate, added

    return insert(root)


def _treap_visit() -> None:
    """Instrumentation seam for deterministic structural-complexity tests."""


@dataclass(frozen=True, slots=True)
class _PersistentMap(Mapping[str, _V], Generic[_V]):
    root: _TreapNode[_V] | None = None

    def __getitem__(self, key: str) -> _V:
        current = self.root
        while current is not None:
            if key == current.key:
                return current.value
            current = current.left if key < current.key else current.right
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        stack: list[_TreapNode[_V]] = []
        current = self.root
        while current is not None or stack:
            while current is not None:
                stack.append(current)
                current = current.left
            current = stack.pop()
            yield current.key
            current = current.right

    def __len__(self) -> int:
        return self.root.size if self.root else 0

    def set(self, key: str, value: _V) -> _PersistentMap[_V]:
        root, _ = _treap_insert(self.root, key, value)
        return _PersistentMap(root)

    def intersection(self, keys: tuple[str, ...]) -> builtins.set[str]:
        return {key for key in keys if key in self}


@dataclass(frozen=True, slots=True)
class _HistoryNode:
    previous: _HistoryNode | None
    rows: tuple[CommittedAuditIntent, ...]
    count: int


def _flatten_history(tail: _HistoryNode | None) -> tuple[CommittedAuditIntent, ...]:
    chunks: list[tuple[CommittedAuditIntent, ...]] = []
    seen: set[int] = set()
    current = tail
    expected_count = current.count if current is not None else 0
    while current is not None:
        if (
            type(current) is not _HistoryNode
            or id(current) in seen
            or type(current.rows) is not tuple
            or current.count != expected_count
            or current.count < len(current.rows)
        ):
            raise ValueError("corrupt committed history")
        seen.add(id(current))
        chunks.append(current.rows)
        expected_count -= len(current.rows)
        current = current.previous
    if expected_count != 0:
        raise ValueError("corrupt committed history count")
    return tuple(row for chunk in reversed(chunks) for row in chunk)


@dataclass(frozen=True, slots=True)
class _RuntimeState:
    committed: _HistoryNode | None
    committed_count: int
    committed_ids: _PersistentMap[bool]
    committed_digests: _PersistentMap[bool]
    committed_transitions: _PersistentMap[bool]
    source_counts: _PersistentMap[int]
    source_occurrences: _PersistentMap[int]
    final: bool
    halt: _HaltRecord | None
    active: _TransitionState | None


@dataclass(frozen=True, slots=True)
class _RuntimeControl:
    state: _RuntimeState
    trusted_state: _RuntimeState
    halt_authority: _HaltAuthority | None


def _publish_state(state: _RuntimeState) -> _RuntimeState:
    """Private test seam: exceptions here occur before the single publication."""

    return state


def _publish_halt(
    authority: _HaltAuthority, state: _RuntimeState
) -> tuple[_HaltAuthority, _RuntimeState]:
    """Private test seam: exceptions occur before authority/state publication."""

    return authority, state


def _intent_digest(
    declaration: AuditBudgetDeclaration,
    intent_id: str,
    source_id: str,
    transition_id: str,
    batch_ordinal: int,
) -> str:
    content = {
        "contract_id": _CONTRACT_ID,
        "operation_core_content_digest_sha256": (
            declaration.operation_core_content_digest_sha256
        ),
        "operation_instance_id": declaration.operation_instance_id,
        "intent_id": intent_id,
        "source_id": source_id,
        "transition_id": transition_id,
        "batch_ordinal": batch_ordinal,
    }
    wrapped = {"content": content, "domain": _INTENT_DIGEST_DOMAIN}
    encoded = json.dumps(
        wrapped,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class AuditBudgetRuntime:
    _creator_pid: int
    _declaration: AuditBudgetDeclaration
    _identity: str
    _lock: threading.RLock
    _sources: tuple[AuditBudgetSource, ...]
    _control: _RuntimeControl

    __slots__ = (
        "_control",
        "_creator_pid",
        "_declaration",
        "_identity",
        "_lock",
        "_sources",
    )

    def __init__(self, declaration: AuditBudgetDeclaration) -> None:
        if type(declaration) is not AuditBudgetDeclaration:
            raise TypeError("declaration must be an exact AuditBudgetDeclaration")
        try:
            declaration.__post_init__()
        except AttributeError as exc:
            raise ValueError("declaration is not completely initialized") from exc
        object.__setattr__(self, "_creator_pid", os.getpid())
        object.__setattr__(self, "_identity", secrets.token_hex(32))
        object.__setattr__(self, "_lock", threading.RLock())
        object.__setattr__(self, "_declaration", declaration)
        object.__setattr__(self, "_sources", tuple(declaration.sources))
        zero_counts: _PersistentMap[int] = _PersistentMap()
        for source in self._sources:
            zero_counts = zero_counts.set(source.source_id, 0)
        state = _RuntimeState(
            None,
            0,
            _PersistentMap(),
            _PersistentMap(),
            _PersistentMap(),
            zero_counts,
            zero_counts,
            False,
            None,
            None,
        )
        object.__setattr__(self, "_control", _RuntimeControl(state, state, None))

    @property
    def _state(self) -> _RuntimeState:
        return self._control.state

    @property
    def _trusted_state(self) -> _RuntimeState:
        return self._control.trusted_state

    @property
    def _halt_authority(self) -> _HaltAuthority | None:
        return self._control.halt_authority

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("audit budget runtime is sealed")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("audit budget runtime is sealed")

    def __copy__(self) -> AuditBudgetRuntime:
        raise TypeError("audit budget runtimes cannot be copied")

    def __deepcopy__(self, memo: object) -> AuditBudgetRuntime:
        raise TypeError("audit budget runtimes cannot be deep-copied")

    def __reduce__(self) -> tuple[object, ...]:
        raise TypeError("audit budget runtimes cannot be pickled")

    def _ensure_pid(self) -> None:
        if self._creator_pid != os.getpid():
            raise RuntimeError("audit budget runtime belongs to another process")

    def _raise_record(self, record: _HaltRecord) -> NoReturn:
        error = AuditIntegrityHalt(
            _CONTRACT_ID,
            record.integrity_code,
            record.refusal_class,
            record.source_id,
            record.requested_count,
            record.operation_committed_before,
            record.source_committed_before,
            self._declaration.declared_max_events,
            record.transition_id,
            record.invocation_index,
        )
        raise error.with_traceback(None)

    def _raise_if_halted_locked(self) -> None:
        halt = self._state.halt
        if halt is not None:
            self._raise_record(halt)

    def _incremental_valid_locked(
        self, predecessor: _RuntimeState, candidate: _RuntimeState
    ) -> bool:
        """Prove one internally authored non-halt successor from the trusted state."""

        if (
            predecessor is not self._trusted_state
            or type(candidate) is not _RuntimeState
            or candidate.halt is not None
            or candidate.committed_count < 0
        ):
            return False
        unchanged_history = (
            candidate.committed is predecessor.committed
            and candidate.committed_count == predecessor.committed_count
            and candidate.committed_ids is predecessor.committed_ids
            and candidate.committed_digests is predecessor.committed_digests
            and candidate.committed_transitions is predecessor.committed_transitions
            and candidate.source_counts is predecessor.source_counts
            and candidate.source_occurrences is predecessor.source_occurrences
        )
        before = predecessor.active
        after = candidate.active
        if unchanged_history:
            if before is None and after is not None:
                return (
                    not predecessor.final
                    and not candidate.final
                    and after.batches == ()
                    and self._active_shallow_valid(after)
                )
            if before is not None and after is None:
                return not candidate.final
            if before is None and after is None:
                return candidate.final and not predecessor.final
            if before is None or after is None or candidate.final:
                return False
            if (
                after.scope is not before.scope
                or after.transition_id != before.transition_id
                or after.invocation_index != before.invocation_index
                or after.identity != before.identity
                or after.reexecution is not before.reexecution
            ):
                return False
            if len(after.batches) == len(before.batches) + 1:
                return after.batches[:-1] == before.batches and self._active_valid(
                    after, candidate
                )
            if len(after.batches) != len(before.batches):
                return False
            changed = [
                index
                for index, (old, new) in enumerate(
                    zip(before.batches, after.batches, strict=True)
                )
                if old is not new
            ]
            if len(changed) != 1:
                return False
            index = changed[0]
            old = before.batches[index]
            new = after.batches[index]
            return (
                new.source is old.source
                and new.reservation is old.reservation
                and new.nonce == old.nonce
                and new.intents is old.intents
                and new.digests is old.digests
                and new.next_index == old.next_index + 1
                and new.next_index <= len(new.intents)
            )
        if before is None or after is not None or candidate.final:
            return False
        rows = (
            candidate.committed.rows
            if type(candidate.committed) is _HistoryNode
            and candidate.committed.previous is predecessor.committed
            else ()
        )
        if (
            not rows
            or candidate.committed_count != predecessor.committed_count + len(rows)
            or candidate.committed is None
            or candidate.committed.count != candidate.committed_count
        ):
            return False
        expected_ids = predecessor.committed_ids
        expected_digests = predecessor.committed_digests
        for offset, row in enumerate(rows, 1):
            if (
                type(row) is not CommittedAuditIntent
                or row.operation_committed_count != predecessor.committed_count + offset
                or row.transition_id != before.transition_id
                or row.intent_digest_sha256
                != _intent_digest(
                    self._declaration,
                    row.intent_id,
                    row.source_id,
                    row.transition_id,
                    row.batch_ordinal,
                )
            ):
                return False
            expected_ids = expected_ids.set(row.intent_id, True)
            expected_digests = expected_digests.set(row.intent_digest_sha256, True)
        expected_transitions = predecessor.committed_transitions.set(
            before.transition_id, True
        )
        expected_counts = predecessor.source_counts
        expected_occurrences = predecessor.source_occurrences
        for batch in before.batches:
            source_id = batch.source.source_id
            expected_counts = expected_counts.set(
                source_id, expected_counts[source_id] + len(batch.intents)
            )
            expected_occurrences = expected_occurrences.set(
                source_id, expected_occurrences[source_id] + 1
            )
        return (
            candidate.committed_ids == expected_ids
            and candidate.committed_digests == expected_digests
            and candidate.committed_transitions == expected_transitions
            and candidate.source_counts == expected_counts
            and candidate.source_occurrences == expected_occurrences
        )

    def _publish_locked(
        self,
        candidate: _RuntimeState,
        *,
        reservation_nonce: str | None = None,
        validate_full: bool = False,
    ) -> None:
        """Publish one validated successor; this is the sole runtime-state writer."""

        owned = getattr(self._lock, "_is_owned", None)
        if owned is not None and not owned():
            raise RuntimeError("audit runtime state publication requires its lock")
        predecessor = self._state
        if reservation_nonce is not None:
            if (
                len(reservation_nonce) != 64
                or _DIGEST.fullmatch(reservation_nonce) is None
            ):
                raise ValueError("invalid internal reservation nonce")
            active = candidate.active
            matching = (
                sum(batch.nonce == reservation_nonce for batch in active.batches)
                if active is not None
                else 0
            )
            if matching > 1:
                raise ValueError("duplicate internal reservation nonce")
            if matching != 1 or active is None:
                raise ValueError("invalid internal reservation nonce")
            if len({batch.nonce for batch in active.batches}) != len(active.batches):
                raise ValueError("duplicate internal reservation nonce")
        if not self._incremental_valid_locked(predecessor, candidate):
            raise ValueError("invalid runtime-state successor")
        if validate_full and not self._integrity_valid_locked(candidate):
            raise ValueError("invalid runtime-state successor")
        control = _RuntimeControl(candidate, candidate, self._halt_authority)
        published = _publish_state(candidate)
        if published is not candidate:
            raise ValueError("invalid state publication")
        object.__setattr__(self, "_control", control)

    def _malformed_record(self) -> _HaltRecord:
        return _HaltRecord(
            "DOMAIN_AUDIT_COUNTER_INTEGRITY",
            "malformed_runtime_counter",
            None,
            0,
            0,
            None,
            None,
            None,
            _HaltWitness("malformed_counter", ()),
        )

    def _active_halt_facts(self, state: _RuntimeState) -> _ActiveHaltFacts | None:
        active = state.active
        if active is None:
            return None
        return _ActiveHaltFacts(
            active.transition_id,
            active.invocation_index,
            active.identity,
            tuple(
                (
                    batch.source.source_id,
                    batch.nonce,
                    len(batch.intents),
                    batch.next_index,
                )
                for batch in active.batches
            ),
        )

    def _publish_halt_locked(
        self, authority: _HaltAuthority, candidate: _RuntimeState
    ) -> None:
        """Validate, then publish authority and state with exception-free stores."""

        if self._halt_authority is not None:
            raise ValueError("halt authority was already minted")
        if not self._integrity_valid_locked(candidate, halt_authority=authority):
            raise ValueError("invalid halted runtime-state successor")
        control = _RuntimeControl(candidate, candidate, authority)
        published_authority, published_state = _publish_halt(authority, candidate)
        if published_authority is not authority or published_state is not candidate:
            raise ValueError("invalid halt publication")
        object.__setattr__(self, "_control", control)

    def _zero_counts(self) -> _PersistentMap[int]:
        result: _PersistentMap[int] = _PersistentMap()
        for source in self._sources:
            result = result.set(source.source_id, 0)
        return result

    def _malformed_locked(self) -> NoReturn:
        zero_counts = self._zero_counts()
        authority = self._halt_authority
        record = authority.malformed_record if authority else self._malformed_record()
        candidate = _RuntimeState(
            None,
            0,
            _PersistentMap(),
            _PersistentMap(),
            _PersistentMap(),
            zero_counts,
            zero_counts,
            False,
            record,
            None,
        )
        if authority is None:
            authority = _HaltAuthority(
                record,
                record,
                "malformed_counter",
                secrets.token_hex(32),
                False,
                zero_counts,
                zero_counts,
                None,
            )
            self._publish_halt_locked(authority, candidate)
        else:
            if not self._integrity_valid_locked(candidate, halt_authority=authority):
                raise RuntimeError("failed to canonicalize malformed audit runtime")
            object.__setattr__(
                self, "_control", _RuntimeControl(candidate, candidate, authority)
            )
        self._raise_record(record)

    def _stop_locked(
        self,
        code: str,
        refusal_class: str,
        source_id: str | None,
        requested_count: int,
        transition_id: str | None,
        invocation_index: int | None,
        witness: _HaltWitness,
        *,
        operation_before: int | None = None,
        source_before: int | Literal[False] | None = False,
    ) -> NoReturn:
        state = self._state
        if state.halt is not None:
            self._raise_record(state.halt)
        operation = (
            state.committed_count if operation_before is None else operation_before
        )
        if source_before is False:
            source_count = (
                state.source_counts.get(source_id) if source_id is not None else None
            )
        else:
            source_count = source_before
        record = _HaltRecord(
            code,
            refusal_class,
            source_id,
            requested_count,
            operation,
            source_count,
            transition_id,
            invocation_index,
            witness,
        )
        authority = _HaltAuthority(
            record,
            self._malformed_record(),
            witness.cause,
            secrets.token_hex(32),
            state.final,
            state.source_counts,
            state.source_occurrences,
            self._active_halt_facts(state),
        )
        candidate = _RuntimeState(
            state.committed,
            state.committed_count,
            state.committed_ids,
            state.committed_digests,
            state.committed_transitions,
            state.source_counts,
            state.source_occurrences,
            state.final,
            record,
            None,
        )
        self._publish_halt_locked(authority, candidate)
        self._raise_record(record)

    def _integrity_valid_locked(
        self,
        state: _RuntimeState | None = None,
        *,
        deep_active: bool = True,
        halt_authority: _HaltAuthority | None = None,
    ) -> bool:
        if state is None:
            state = self._state
        if halt_authority is None:
            halt_authority = self._halt_authority
        if type(state) is not _RuntimeState:
            return False
        if type(self._declaration) is not AuditBudgetDeclaration:
            return False
        try:
            self._declaration.__post_init__()
            rows = _flatten_history(state.committed)
        except (AttributeError, TypeError, ValueError):
            return False
        if type(self._sources) is not tuple or self._sources != self._declaration.sources:
            return False
        if (
            type(state.committed_count) is not int
            or state.committed_count != len(rows)
            or type(state.committed_ids) is not _PersistentMap
            or type(state.committed_digests) is not _PersistentMap
            or type(state.committed_transitions) is not _PersistentMap
            or type(state.source_counts) is not _PersistentMap
            or type(state.source_occurrences) is not _PersistentMap
            or type(state.final) is not bool
            or (state.halt is not None and type(state.halt) is not _HaltRecord)
            or (state.active is not None and type(state.active) is not _TransitionState)
        ):
            return False
        source_ids = tuple(source.source_id for source in self._sources)
        if set(state.source_counts) != set(source_ids) or set(
            state.source_occurrences
        ) != set(source_ids):
            return False
        if any(
            type(value) is not int or not 0 <= value <= _V1_CAPACITY
            for value in (
                *state.source_counts.values(),
                *state.source_occurrences.values(),
            )
        ):
            return False
        by_id = {source.source_id: source for source in self._sources}
        derived_counts = dict.fromkeys(source_ids, 0)
        occurrence_rows: dict[tuple[str, str], list[CommittedAuditIntent]] = {}
        seen_ids: set[str] = set()
        seen_digests: set[str] = set()
        seen_transitions: set[str] = set()
        for position, row in enumerate(rows, 1):
            if type(row) is not CommittedAuditIntent:
                return False
            source = by_id.get(row.source_id) if type(row.source_id) is str else None
            if (
                row.record_type != "committed_intent"
                or row.contract_id != _CONTRACT_ID
                or type(row.intent_id) is not str
                or _PORTABLE_ID.fullmatch(row.intent_id) is None
                or source is None
                or row.counted_kind != source.counted_kind
                or type(row.transition_id) is not str
                or _PORTABLE_ID.fullmatch(row.transition_id) is None
                or type(row.batch_ordinal) is not int
                or not 0 <= row.batch_ordinal <= _V1_CAPACITY
                or type(row.operation_committed_count) is not int
                or row.operation_committed_count != position
                or type(row.intent_digest_sha256) is not str
                or _DIGEST.fullmatch(row.intent_digest_sha256) is None
                or row.intent_id in seen_ids
                or row.intent_digest_sha256 in seen_digests
                or row.intent_digest_sha256
                != _intent_digest(
                    self._declaration,
                    row.intent_id,
                    row.source_id,
                    row.transition_id,
                    row.batch_ordinal,
                )
            ):
                return False
            seen_ids.add(row.intent_id)
            seen_digests.add(row.intent_digest_sha256)
            seen_transitions.add(row.transition_id)
            derived_counts[row.source_id] += 1
            occurrence_rows.setdefault((row.source_id, row.transition_id), []).append(row)
        derived_occurrences = dict.fromkeys(source_ids, 0)
        for (source_id, _), grouped in occurrence_rows.items():
            if tuple(row.batch_ordinal for row in grouped) != tuple(range(len(grouped))):
                return False
            source = by_id[source_id]
            if len(grouped) > source.max_events_per_transition:
                return False
            derived_occurrences[source_id] += 1
        if (
            set(state.committed_ids) != seen_ids
            or set(state.committed_digests) != seen_digests
            or set(state.committed_transitions) != seen_transitions
            or dict(state.source_counts) != derived_counts
            or dict(state.source_occurrences) != derived_occurrences
            or len(rows) > self._declaration.declared_max_events
        ):
            return False
        if any(
            derived_occurrences[source.source_id] > source.max_accepted_occurrences
            or derived_counts[source.source_id]
            > source.max_accepted_occurrences * source.max_events_per_transition
            for source in self._sources
        ):
            return False
        if state.final and state.active is not None:
            return False
        if (state.halt is None) is not (halt_authority is None):
            return False
        if state.halt is not None and (
            state.active is not None
            or not self._halt_valid(state.halt, state, derived_counts, halt_authority)
            or (state.final and state.halt.refusal_class != "post_finality_attempt")
            or (not state.final and state.halt.refusal_class == "post_finality_attempt")
        ):
            return False
        if state.active is not None:
            if deep_active:
                if not self._active_valid(state.active, state):
                    return False
            elif not self._active_shallow_valid(state.active):
                return False
        return True

    def _active_shallow_valid(self, active: _TransitionState) -> bool:
        if (
            type(active.scope) is not AuditTransitionScope
            or active.scope._runtime is not self
            or active.scope._creator_pid != self._creator_pid
            or active.scope._transition_id != active.transition_id
            or active.scope._invocation_index != active.invocation_index
            or active.scope._identity != active.identity
            or type(active.transition_id) is not str
            or _PORTABLE_ID.fullmatch(active.transition_id) is None
            or type(active.invocation_index) is not int
            or not 0 <= active.invocation_index <= _V1_CAPACITY
            or type(active.identity) is not str
            or type(active.batches) is not tuple
        ):
            return False
        return all(
            type(batch) is _BatchState
            and type(batch.intents) is tuple
            and type(batch.digests) is tuple
            and type(batch.next_index) is int
            and 0 <= batch.next_index <= len(batch.intents)
            and type(batch.reservation) is AuditReservation
            and batch.reservation._runtime is self
            and batch.reservation._transition_identity == active.identity
            and batch.reservation._creator_pid == self._creator_pid
            and batch.reservation._nonce == batch.nonce
            for batch in active.batches
        )

    def _halt_valid(
        self,
        halt: _HaltRecord,
        state: _RuntimeState,
        counts: dict[str, int],
        authority: _HaltAuthority | None,
    ) -> bool:
        expected_codes = {
            "unknown_source_or_kind": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
            "invalid_batch_or_identity": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
            "reservation_lifecycle_violation": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
            "post_finality_attempt": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
            "checked_signed_int64_arithmetic_overflow": (
                "DOMAIN_AUDIT_COUNTER_INTEGRITY"
            ),
            "malformed_runtime_counter": "DOMAIN_AUDIT_COUNTER_INTEGRITY",
            "valid_reservation_above_declared_aggregate": (
                "DOMAIN_AUDIT_BUDGET_EXCEEDED"
            ),
            "valid_reservation_above_declared_source": "DOMAIN_AUDIT_BUDGET_EXCEEDED",
        }
        if (
            type(authority) is not _HaltAuthority
            or not (halt is authority.record or halt is authority.malformed_record)
            or type(authority.nonce) is not str
            or len(authority.nonce) != 64
            or _DIGEST.fullmatch(authority.nonce) is None
            or type(authority.cause) is not str
            or authority.cause != authority.record.witness.cause
            or type(authority.final_before) is not bool
            or type(halt.integrity_code) is not str
            or type(halt.refusal_class) is not str
            or expected_codes.get(halt.refusal_class) != halt.integrity_code
            or type(halt.witness) is not _HaltWitness
            or type(halt.witness.cause) is not str
            or type(halt.witness.facts) is not tuple
            or (
                halt.source_id is not None
                and (
                    type(halt.source_id) is not str
                    or len(halt.source_id) > 255
                    or _PORTABLE_ID.fullmatch(halt.source_id) is None
                )
            )
            or type(halt.requested_count) is not int
            or not 0 <= halt.requested_count <= _V1_CAPACITY
            or type(halt.operation_committed_before) is not int
            or not 0 <= halt.operation_committed_before <= _V1_CAPACITY
            or (
                halt.source_committed_before is not None
                and (
                    type(halt.source_committed_before) is not int
                    or not 0 <= halt.source_committed_before <= _V1_CAPACITY
                )
            )
            or (
                halt.transition_id is not None
                and (
                    type(halt.transition_id) is not str
                    or len(halt.transition_id) > 255
                    or _PORTABLE_ID.fullmatch(halt.transition_id) is None
                )
            )
            or (
                halt.invocation_index is not None
                and (
                    type(halt.invocation_index) is not int
                    or not 0 <= halt.invocation_index <= _V1_CAPACITY
                )
            )
        ):
            return False
        if halt is authority.malformed_record:
            return (
                halt == self._malformed_record()
                and state.committed is None
                and not state.final
            )
        if (
            halt.witness.cause != authority.cause
            or authority.source_counts_before != state.source_counts
            or authority.source_occurrences_before != state.source_occurrences
            or authority.final_before is not state.final
        ):
            return False
        witness = halt.witness
        active_facts = authority.active_facts
        if halt.refusal_class == "post_finality_attempt":
            if active_facts is not None:
                return False
        elif (
            type(active_facts) is not _ActiveHaltFacts
            or (active_facts.transition_id, active_facts.invocation_index)
            != (halt.transition_id, halt.invocation_index)
            or type(active_facts.identity) is not str
            or len(active_facts.identity) != 64
            or _DIGEST.fullmatch(active_facts.identity) is None
            or type(active_facts.batches) is not tuple
            or any(
                type(batch) is not tuple
                or len(batch) != 4
                or type(batch[0]) is not str
                or type(batch[1]) is not str
                or len(batch[1]) != 64
                or _DIGEST.fullmatch(batch[1]) is None
                or type(batch[2]) is not int
                or type(batch[3]) is not int
                or not 0 <= batch[3] <= batch[2] <= _V1_CAPACITY
                for batch in active_facts.batches
            )
        ):
            return False
        if halt.refusal_class == "malformed_runtime_counter":
            return (
                witness == _HaltWitness("malformed_counter", ())
                and halt.source_id is None
                and halt.requested_count == 0
                and halt.operation_committed_before == 0
                and halt.source_committed_before is None
                and halt.transition_id is None
                and halt.invocation_index is None
            )
        committed = sum(counts.values())
        if halt.refusal_class == "post_finality_attempt":
            return (
                witness == _HaltWitness("post_finality", (True,))
                and state.final
                and halt.source_id is None
                and halt.requested_count == 0
                and halt.operation_committed_before == committed
                and halt.source_committed_before is None
                and halt.transition_id is None
                and halt.invocation_index is None
            )
        if (
            halt.operation_committed_before != committed
            or halt.source_id is None
            or halt.transition_id is None
            or halt.invocation_index is None
        ):
            return False
        source = next(
            (item for item in self._sources if item.source_id == halt.source_id), None
        )
        if halt.refusal_class == "unknown_source_or_kind":
            facts = witness.facts
            return (
                witness.cause == "unknown_source_or_kind"
                and len(facts) == 5
                and type(facts[0]) is str
                and len(facts[0]) <= 255
                and _PORTABLE_ID.fullmatch(facts[0]) is not None
                and type(facts[1]) is bool
                and facts[1] is (source is not None)
                and facts[2:]
                == (
                    halt.source_id,
                    halt.transition_id,
                    halt.invocation_index,
                )
                and halt.requested_count > 0
                and halt.source_committed_before == counts.get(halt.source_id)
                and (source is None or facts[0] != source.counted_kind)
            )
        if source is None or halt.source_committed_before != counts[halt.source_id]:
            return False
        if halt.refusal_class == "invalid_batch_or_identity":
            allowed = {
                "empty_or_unknown_batch",
                "batch_shape",
                "duplicate_source_batch",
                "duplicate_intent_identity",
                "digest_identity_collision",
                "duplicate_transition_reexecution",
            }
            return (
                witness.cause == "invalid_batch"
                and witness.facts
                == (
                    witness.facts[0],
                    halt.source_id,
                    halt.transition_id,
                    halt.invocation_index,
                    0,
                )
                and type(witness.facts[0]) is str
                and witness.facts[0] in allowed
                and halt.requested_count == 0
            )
        if halt.refusal_class == "reservation_lifecycle_violation":
            facts = witness.facts
            return (
                witness.cause == "reservation_lifecycle"
                and len(facts) == 7
                and type(facts[0]) is str
                and facts[0]
                in {
                    "foreign_reservation",
                    "consume_order",
                    "incomplete_commit",
                    "active_overlap",
                }
                and type(facts[1]) is str
                and len(facts[1]) == 64
                and _DIGEST.fullmatch(facts[1]) is not None
                and active_facts is not None
                and facts[1] in tuple(batch[1] for batch in active_facts.batches)
                and type(facts[2]) is str
                and len(facts[2]) == 64
                and _DIGEST.fullmatch(facts[2]) is not None
                and facts[3:]
                == (
                    halt.source_id,
                    halt.requested_count,
                    halt.transition_id,
                    halt.invocation_index,
                )
                and halt.requested_count > 0
                and active_facts is not None
                and facts[2] == active_facts.identity
                and (halt.source_id, facts[1], halt.requested_count)
                in tuple(batch[:3] for batch in active_facts.batches)
            )
        if halt.refusal_class == "valid_reservation_above_declared_aggregate":
            facts = witness.facts
            if (
                witness.cause != "aggregate_limit"
                or len(facts) != 7
                or any(type(value) is not int for value in facts[:4])
                or facts[4:]
                != (halt.source_id, halt.transition_id, halt.invocation_index)
            ):
                return False
            staged = facts[1]
            assert type(staged) is int
            return (
                facts[:4]
                == (
                    committed,
                    staged,
                    halt.requested_count,
                    self._declaration.declared_max_events,
                )
                and 0 <= staged <= _V1_CAPACITY
                and committed + staged + halt.requested_count
                > self._declaration.declared_max_events
                and halt.requested_count > 0
            )
        if halt.refusal_class == "valid_reservation_above_declared_source":
            facts = witness.facts
            if (
                witness.cause != "source_limit"
                or len(facts) != 10
                or any(type(value) is not int for value in facts[:7])
                or facts[7:]
                != (halt.source_id, halt.transition_id, halt.invocation_index)
            ):
                return False
            staged = facts[1]
            assert type(staged) is int
            source_limit = (
                source.max_accepted_occurrences * source.max_events_per_transition
            )
            occurrences = dict(state.source_occurrences)[source.source_id]
            return (
                facts[:7]
                == (
                    committed,
                    staged,
                    halt.requested_count,
                    counts[source.source_id],
                    occurrences,
                    source_limit,
                    self._declaration.declared_max_events,
                )
                and 0 <= staged <= _V1_CAPACITY
                and committed + staged + halt.requested_count
                <= self._declaration.declared_max_events
                and (
                    occurrences >= source.max_accepted_occurrences
                    or counts[source.source_id] + halt.requested_count > source_limit
                )
                and halt.requested_count > 0
            )
        if halt.refusal_class == "checked_signed_int64_arithmetic_overflow":
            facts = witness.facts
            if (
                witness.cause != "checked_overflow"
                or len(facts) != 6
                or facts[0] not in {"add", "multiply"}
                or type(facts[1]) is not int
                or type(facts[2]) is not int
                or not 0 <= facts[1] <= _V1_CAPACITY
                or not 0 <= facts[2] <= _V1_CAPACITY
                or facts[2] != halt.requested_count
                or facts[3:]
                != (halt.source_id, halt.transition_id, halt.invocation_index)
                or halt.requested_count <= 0
            ):
                return False
            return (
                facts[1] > _INT64_MAX - facts[2]
                if facts[0] == "add"
                else bool(facts[1] and facts[2] > _INT64_MAX // facts[1])
            )
        return False

    def _active_valid(self, active: _TransitionState, state: _RuntimeState) -> bool:
        if (
            type(active.scope) is not AuditTransitionScope
            or active.scope._runtime is not self
            or active.scope._creator_pid != self._creator_pid
            or active.scope._transition_id != active.transition_id
            or active.scope._invocation_index != active.invocation_index
            or active.scope._identity != active.identity
            or type(active.transition_id) is not str
            or _PORTABLE_ID.fullmatch(active.transition_id) is None
            or type(active.invocation_index) is not int
            or not 0 <= active.invocation_index <= _V1_CAPACITY
            or type(active.identity) is not str
            or type(active.batches) is not tuple
        ):
            return False
        source_ids: set[str] = set()
        staged_ids: set[str] = set()
        staged_digests: set[str] = set()
        for batch in active.batches:
            if (
                type(batch) is not _BatchState
                or type(batch.source) is not AuditBudgetSource
            ):
                return False
            source = next(
                (
                    candidate
                    for candidate in self._sources
                    if candidate.source_id == batch.source.source_id
                ),
                None,
            )
            if (
                source is not batch.source
                or batch.source.source_id in source_ids
                or type(batch.reservation) is not AuditReservation
                or batch.reservation._runtime is not self
                or batch.reservation._transition_identity != active.identity
                or batch.reservation._creator_pid != self._creator_pid
                or batch.reservation._nonce != batch.nonce
                or type(batch.nonce) is not str
                or len(batch.nonce) != 64
                or type(batch.intents) is not tuple
                or not 0 < len(batch.intents) <= source.max_events_per_transition
                or any(
                    type(intent) is not str
                    or len(intent) > 255
                    or _PORTABLE_ID.fullmatch(intent) is None
                    for intent in batch.intents
                )
                or len(set(batch.intents)) != len(batch.intents)
                or type(batch.digests) is not tuple
                or batch.digests
                != tuple(
                    _intent_digest(
                        self._declaration,
                        intent,
                        source.source_id,
                        active.transition_id,
                        ordinal,
                    )
                    for ordinal, intent in enumerate(batch.intents)
                )
                or type(batch.next_index) is not int
                or not 0 <= batch.next_index <= len(batch.intents)
                or staged_ids.intersection(batch.intents)
                or staged_digests.intersection(batch.digests)
                or state.committed_ids.intersection(batch.intents)
                or state.committed_digests.intersection(batch.digests)
            ):
                return False
            source_ids.add(source.source_id)
            staged_ids.update(batch.intents)
            staged_digests.update(batch.digests)
        return True

    def _check_locked(self, *, validate_full: bool = False) -> None:
        if self._state is not self._trusted_state or (
            validate_full and not self._integrity_valid_locked()
        ):
            self._malformed_locked()

    def begin_transition(
        self, transition_id: str, invocation_index: int
    ) -> AuditTransitionScope:
        self._ensure_pid()
        with self._lock:
            self._check_locked()
            self._raise_if_halted_locked()
            _portable_id(transition_id, "transition_id")
            _exact_int(invocation_index, "invocation_index", 0, _V1_CAPACITY)
            state = self._state
            if state.final:
                self._stop_locked(
                    "DOMAIN_AUDIT_CONTRACT_VIOLATION",
                    "post_finality_attempt",
                    None,
                    0,
                    None,
                    None,
                    _HaltWitness("post_finality", (True,)),
                    source_before=None,
                )
            if state.active is not None:
                self._lifecycle_or_programmer_locked(state.active)
            reexecution = transition_id in state.committed_transitions
            identity = secrets.token_hex(32)
            scope = AuditTransitionScope(
                self, transition_id, invocation_index, identity, self._creator_pid
            )
            active = _TransitionState(
                scope, transition_id, invocation_index, identity, (), reexecution
            )
            candidate = _RuntimeState(
                state.committed,
                state.committed_count,
                state.committed_ids,
                state.committed_digests,
                state.committed_transitions,
                state.source_counts,
                state.source_occurrences,
                False,
                None,
                active,
            )
            self._publish_locked(candidate)
            return scope

    def _require_scope_locked(self, scope: AuditTransitionScope) -> _TransitionState:
        state = self._state
        active = state.active
        if active is None or active.scope is not scope:
            if active is not None:
                self._lifecycle_or_programmer_locked(active)
            raise RuntimeError("audit transition scope is not active")
        return active

    def _lifecycle_or_programmer_locked(self, active: _TransitionState) -> NoReturn:
        if not active.batches:
            raise RuntimeError(
                "reservation lifecycle refusal requires a real reservation"
            )
        batch = active.batches[0]
        self._stop_locked(
            "DOMAIN_AUDIT_CONTRACT_VIOLATION",
            "reservation_lifecycle_violation",
            batch.source.source_id,
            len(batch.intents),
            active.transition_id,
            active.invocation_index,
            _HaltWitness(
                "reservation_lifecycle",
                (
                    "active_overlap",
                    batch.nonce,
                    active.identity,
                    batch.source.source_id,
                    len(batch.intents),
                    active.transition_id,
                    active.invocation_index,
                ),
            ),
        )

    def _reserve(
        self,
        scope: AuditTransitionScope,
        source_id: str,
        counted_kind: str,
        intent_ids: tuple[str, ...],
    ) -> AuditReservation:
        self._ensure_pid()
        with self._lock:
            self._check_locked()
            self._raise_if_halted_locked()
            active = self._require_scope_locked(scope)
            _portable_id(source_id, "source_id")
            _portable_id(counted_kind, "counted_kind")
            if type(intent_ids) is tuple:
                for intent_id in intent_ids:
                    _portable_id(intent_id, "intent_id")
            if active.reexecution:
                self._stop_locked(
                    "DOMAIN_AUDIT_CONTRACT_VIOLATION",
                    "invalid_batch_or_identity",
                    source_id,
                    0,
                    active.transition_id,
                    active.invocation_index,
                    _HaltWitness(
                        "invalid_batch",
                        (
                            "duplicate_transition_reexecution",
                            source_id,
                            active.transition_id,
                            active.invocation_index,
                            0,
                        ),
                    ),
                )
            source = next(
                (
                    candidate
                    for candidate in self._sources
                    if candidate.source_id == source_id
                ),
                None,
            )
            requested = len(intent_ids) if type(intent_ids) is tuple else 0
            if (
                source is None
                or counted_kind not in _COUNTED_KINDS
                or source.counted_kind != counted_kind
            ):
                if requested <= 0:
                    self._stop_locked(
                        "DOMAIN_AUDIT_CONTRACT_VIOLATION",
                        "invalid_batch_or_identity",
                        source_id,
                        0,
                        active.transition_id,
                        active.invocation_index,
                        _HaltWitness(
                            "invalid_batch",
                            (
                                "empty_or_unknown_batch",
                                source_id,
                                active.transition_id,
                                active.invocation_index,
                                0,
                            ),
                        ),
                    )
                self._stop_locked(
                    "DOMAIN_AUDIT_CONTRACT_VIOLATION",
                    "unknown_source_or_kind",
                    source_id,
                    requested,
                    active.transition_id,
                    active.invocation_index,
                    _HaltWitness(
                        "unknown_source_or_kind",
                        (
                            counted_kind,
                            source is not None,
                            source_id,
                            active.transition_id,
                            active.invocation_index,
                        ),
                    ),
                )
            assert source is not None
            staged_ids = {intent for batch in active.batches for intent in batch.intents}
            if (
                type(intent_ids) is not tuple
                or not 0 < requested <= source.max_events_per_transition
                or any(batch.source.source_id == source_id for batch in active.batches)
                or len(set(intent_ids)) != requested
                or staged_ids.intersection(intent_ids)
                or self._state.committed_ids.intersection(intent_ids)
            ):
                self._stop_locked(
                    "DOMAIN_AUDIT_CONTRACT_VIOLATION",
                    "invalid_batch_or_identity",
                    source_id,
                    0,
                    active.transition_id,
                    active.invocation_index,
                    _HaltWitness(
                        "invalid_batch",
                        (
                            "batch_shape",
                            source_id,
                            active.transition_id,
                            active.invocation_index,
                            0,
                        ),
                    ),
                )
            staged_count = sum(len(batch.intents) for batch in active.batches)
            try:
                attempted = _checked_add(staged_count, requested)
                aggregate = _checked_add(self._state.committed_count, attempted)
                source_limit = _checked_multiply(
                    source.max_accepted_occurrences, source.max_events_per_transition
                )
                source_after = _checked_add(
                    dict(self._state.source_counts)[source_id], requested
                )
            except OverflowError:
                self._stop_locked(
                    "DOMAIN_AUDIT_COUNTER_INTEGRITY",
                    "checked_signed_int64_arithmetic_overflow",
                    source_id,
                    requested,
                    active.transition_id,
                    active.invocation_index,
                    _HaltWitness(
                        "checked_overflow",
                        (
                            "add",
                            staged_count,
                            requested,
                            source_id,
                            active.transition_id,
                            active.invocation_index,
                        ),
                    ),
                )
            if aggregate > self._declaration.declared_max_events:
                self._stop_locked(
                    "DOMAIN_AUDIT_BUDGET_EXCEEDED",
                    "valid_reservation_above_declared_aggregate",
                    source_id,
                    requested,
                    active.transition_id,
                    active.invocation_index,
                    _HaltWitness(
                        "aggregate_limit",
                        (
                            self._state.committed_count,
                            staged_count,
                            requested,
                            self._declaration.declared_max_events,
                            source_id,
                            active.transition_id,
                            active.invocation_index,
                        ),
                    ),
                )
            occurrences = dict(self._state.source_occurrences)[source_id]
            if (
                occurrences >= source.max_accepted_occurrences
                or source_after > source_limit
            ):
                self._stop_locked(
                    "DOMAIN_AUDIT_BUDGET_EXCEEDED",
                    "valid_reservation_above_declared_source",
                    source_id,
                    requested,
                    active.transition_id,
                    active.invocation_index,
                    _HaltWitness(
                        "source_limit",
                        (
                            self._state.committed_count,
                            staged_count,
                            requested,
                            dict(self._state.source_counts)[source_id],
                            occurrences,
                            source_limit,
                            self._declaration.declared_max_events,
                            source_id,
                            active.transition_id,
                            active.invocation_index,
                        ),
                    ),
                )
            digests = tuple(
                _intent_digest(
                    self._declaration,
                    intent_id,
                    source_id,
                    active.transition_id,
                    ordinal,
                )
                for ordinal, intent_id in enumerate(intent_ids)
            )
            if len(set(digests)) != len(
                digests
            ) or self._state.committed_digests.intersection(digests):
                self._stop_locked(
                    "DOMAIN_AUDIT_CONTRACT_VIOLATION",
                    "invalid_batch_or_identity",
                    source_id,
                    0,
                    active.transition_id,
                    active.invocation_index,
                    _HaltWitness(
                        "invalid_batch",
                        (
                            "batch_shape",
                            source_id,
                            active.transition_id,
                            active.invocation_index,
                            0,
                        ),
                    ),
                )
            nonce = secrets.token_hex(32)
            reservation = AuditReservation(
                self, active.identity, nonce, self._creator_pid
            )
            batch = _BatchState(source, reservation, nonce, intent_ids, digests, 0)
            new_active = _TransitionState(
                active.scope,
                active.transition_id,
                active.invocation_index,
                active.identity,
                (*active.batches, batch),
                active.reexecution,
            )
            state = self._state
            candidate = _RuntimeState(
                state.committed,
                state.committed_count,
                state.committed_ids,
                state.committed_digests,
                state.committed_transitions,
                state.source_counts,
                state.source_occurrences,
                False,
                None,
                new_active,
            )
            self._publish_locked(candidate, reservation_nonce=nonce)
            return reservation

    def _consume(self, reservation: AuditReservation, intent_id: str) -> None:
        self._ensure_pid()
        with self._lock:
            self._check_locked(validate_full=False)
            self._raise_if_halted_locked()
            active = self._state.active
            if active is None:
                raise RuntimeError("audit transition scope is not active")
            self._consume_locked(active.scope, reservation, intent_id)

    def _consume_from_scope(
        self,
        scope: AuditTransitionScope,
        reservation: AuditReservation,
        intent_id: str,
    ) -> None:
        self._ensure_pid()
        with self._lock:
            self._check_locked(validate_full=False)
            self._raise_if_halted_locked()
            active = self._require_scope_locked(scope)
            self._consume_locked(active.scope, reservation, intent_id)

    def _consume_locked(
        self,
        scope: AuditTransitionScope,
        reservation: AuditReservation,
        intent_id: str,
    ) -> None:
        active = self._require_scope_locked(scope)
        _portable_id(intent_id, "intent_id")
        batch_index = (
            next(
                (
                    index
                    for index, batch in enumerate(active.batches)
                    if batch.reservation is reservation
                    and batch.nonce == reservation._nonce
                    and reservation._runtime is self
                    and reservation._transition_identity == active.identity
                    and reservation._creator_pid == self._creator_pid
                ),
                None,
            )
            if type(reservation) is AuditReservation
            else None
        )
        if batch_index is None:
            self._lifecycle_or_programmer_locked(active)
        assert batch_index is not None
        batch = active.batches[batch_index]
        if (
            batch.next_index >= len(batch.intents)
            or intent_id != batch.intents[batch.next_index]
        ):
            self._stop_locked(
                "DOMAIN_AUDIT_CONTRACT_VIOLATION",
                "reservation_lifecycle_violation",
                batch.source.source_id,
                len(batch.intents),
                active.transition_id,
                active.invocation_index,
                _HaltWitness(
                    "reservation_lifecycle",
                    (
                        "consume_order",
                        batch.nonce,
                        active.identity,
                        batch.source.source_id,
                        len(batch.intents),
                        active.transition_id,
                        active.invocation_index,
                    ),
                ),
            )
        new_batch = _BatchState(
            batch.source,
            batch.reservation,
            batch.nonce,
            batch.intents,
            batch.digests,
            batch.next_index + 1,
        )
        batches = (
            *active.batches[:batch_index],
            new_batch,
            *active.batches[batch_index + 1 :],
        )
        new_active = _TransitionState(
            active.scope,
            active.transition_id,
            active.invocation_index,
            active.identity,
            batches,
            active.reexecution,
        )
        state = self._state
        candidate = _RuntimeState(
            state.committed,
            state.committed_count,
            state.committed_ids,
            state.committed_digests,
            state.committed_transitions,
            state.source_counts,
            state.source_occurrences,
            False,
            None,
            new_active,
        )
        self._publish_locked(candidate, validate_full=False)

    def _commit(self, scope: AuditTransitionScope) -> None:
        self._ensure_pid()
        with self._lock:
            self._check_locked()
            self._raise_if_halted_locked()
            state = self._state
            active = self._require_scope_locked(scope)
            if state.final:
                self._stop_locked(
                    "DOMAIN_AUDIT_CONTRACT_VIOLATION",
                    "post_finality_attempt",
                    None,
                    0,
                    None,
                    None,
                    _HaltWitness("post_finality", (True,)),
                    source_before=None,
                )
            if not active.batches and not active.reexecution:
                raise RuntimeError("commit requires at least one real reservation")
            if active.reexecution:
                candidate = _RuntimeState(
                    state.committed,
                    state.committed_count,
                    state.committed_ids,
                    state.committed_digests,
                    state.committed_transitions,
                    state.source_counts,
                    state.source_occurrences,
                    False,
                    None,
                    None,
                )
                self._publish_locked(candidate)
                return
            incomplete = next(
                (
                    batch
                    for batch in active.batches
                    if batch.next_index != len(batch.intents)
                ),
                None,
            )
            if incomplete is not None:
                self._stop_locked(
                    "DOMAIN_AUDIT_CONTRACT_VIOLATION",
                    "reservation_lifecycle_violation",
                    incomplete.source.source_id,
                    len(incomplete.intents),
                    active.transition_id,
                    active.invocation_index,
                    _HaltWitness(
                        "reservation_lifecycle",
                        (
                            "incomplete_commit",
                            incomplete.nonce,
                            active.identity,
                            incomplete.source.source_id,
                            len(incomplete.intents),
                            active.transition_id,
                            active.invocation_index,
                        ),
                    ),
                )
            rows: list[CommittedAuditIntent] = []
            start = state.committed_count
            for batch in active.batches:
                for ordinal, intent_id in enumerate(batch.intents):
                    digest = _intent_digest(
                        self._declaration,
                        intent_id,
                        batch.source.source_id,
                        active.transition_id,
                        ordinal,
                    )
                    rows.append(
                        CommittedAuditIntent(
                            "committed_intent",
                            _CONTRACT_ID,
                            digest,
                            intent_id,
                            batch.source.source_id,
                            batch.source.counted_kind,
                            active.transition_id,
                            ordinal,
                            start + len(rows) + 1,
                        )
                    )
            row_tuple = tuple(rows)
            committed = _HistoryNode(state.committed, row_tuple, start + len(rows))
            committed_ids = state.committed_ids
            committed_digests = state.committed_digests
            committed_transitions = state.committed_transitions.set(
                active.transition_id, True
            )
            for row in rows:
                committed_ids = committed_ids.set(row.intent_id, True)
                committed_digests = committed_digests.set(row.intent_digest_sha256, True)
            source_counts = state.source_counts
            source_occurrences = state.source_occurrences
            for batch in active.batches:
                source_id = batch.source.source_id
                source_counts = source_counts.set(
                    source_id, source_counts[source_id] + len(batch.intents)
                )
                source_occurrences = source_occurrences.set(
                    source_id, source_occurrences[source_id] + 1
                )
            new_state = _RuntimeState(
                committed,
                start + len(rows),
                committed_ids,
                committed_digests,
                committed_transitions,
                source_counts,
                source_occurrences,
                False,
                None,
                None,
            )
            self._publish_locked(new_state)

    def _abort(self, scope: AuditTransitionScope) -> None:
        self._ensure_pid()
        with self._lock:
            self._check_locked()
            self._raise_if_halted_locked()
            self._require_scope_locked(scope)
            state = self._state
            candidate = _RuntimeState(
                state.committed,
                state.committed_count,
                state.committed_ids,
                state.committed_digests,
                state.committed_transitions,
                state.source_counts,
                state.source_occurrences,
                False,
                None,
                None,
            )
            self._publish_locked(candidate)

    def _abort_if_active(self, scope: AuditTransitionScope) -> None:
        self._ensure_pid()
        with self._lock:
            self._check_locked()
            state = self._state
            if state.halt is not None or state.active is None:
                return
            if state.active.scope is scope:
                candidate = _RuntimeState(
                    state.committed,
                    state.committed_count,
                    state.committed_ids,
                    state.committed_digests,
                    state.committed_transitions,
                    state.source_counts,
                    state.source_occurrences,
                    False,
                    None,
                    None,
                )
                self._publish_locked(candidate)

    def mark_finality(self) -> None:
        self._ensure_pid()
        with self._lock:
            self._check_locked(validate_full=True)
            self._raise_if_halted_locked()
            state = self._state
            if state.final:
                self._stop_locked(
                    "DOMAIN_AUDIT_CONTRACT_VIOLATION",
                    "post_finality_attempt",
                    None,
                    0,
                    None,
                    None,
                    _HaltWitness("post_finality", (True,)),
                    source_before=None,
                )
            if state.active is not None:
                self._lifecycle_or_programmer_locked(state.active)
            candidate = _RuntimeState(
                state.committed,
                state.committed_count,
                state.committed_ids,
                state.committed_digests,
                state.committed_transitions,
                state.source_counts,
                state.source_occurrences,
                True,
                None,
                None,
            )
            self._publish_locked(candidate)

    def snapshot(self) -> AuditBudgetSnapshot:
        self._ensure_pid()
        with self._lock:
            self._check_locked(validate_full=True)
            state = self._state
            committed = _flatten_history(state.committed)
            return AuditBudgetSnapshot(
                state.committed_count,
                tuple(
                    (source.source_id, state.source_counts[source.source_id])
                    for source in self._sources
                ),
                tuple(
                    (source.source_id, state.source_occurrences[source.source_id])
                    for source in self._sources
                ),
                committed,
                state.final,
                state.halt is not None,
            )

    def terminal_evidence(
        self,
    ) -> tuple[CommittedAuditIntent | AuditIntegrityFinding | AuditTerminalSummary, ...]:
        self._ensure_pid()
        with self._lock:
            self._check_locked(validate_full=True)
            state = self._state
            committed = _flatten_history(state.committed)
            if not state.final and state.halt is None:
                raise RuntimeError(
                    "audit terminal evidence is unavailable before finality"
                )
            findings: tuple[AuditIntegrityFinding, ...] = ()
            if state.halt is not None:
                halt = state.halt
                findings = (
                    AuditIntegrityFinding(
                        "integrity_finding",
                        _CONTRACT_ID,
                        halt.integrity_code,
                        halt.refusal_class,
                        halt.source_id,
                        halt.requested_count,
                        halt.operation_committed_before,
                        halt.source_committed_before,
                        self._declaration.declared_max_events,
                        halt.transition_id,
                        halt.invocation_index,
                    ),
                )
            result: Literal["SUCCESS", "ERROR"] = (
                "ERROR" if state.halt is not None else "SUCCESS"
            )
            summary = AuditTerminalSummary(
                "terminal_summary",
                _CONTRACT_ID,
                self._declaration.declared_max_events,
                state.committed_count,
                result,
                state.halt.integrity_code if state.halt is not None else None,
            )
            return (*committed, *findings, summary)
