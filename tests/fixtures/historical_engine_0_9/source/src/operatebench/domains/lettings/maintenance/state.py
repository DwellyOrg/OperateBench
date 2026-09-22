"""Canonical business state for the synthetic Maintenance operation.

This is the operation's own record, and it is deliberately not the agent's
memory. Whatever an agent remembers between invocations is derived context; the
facts that decide whether work was done, whether a quote was approved and
whether money moved live here, and they only change through the reducers in
:mod:`operatebench.domains.lettings.maintenance.operation`.

Two structural choices carry most of the weight:

* **Cycles.** A quote, an approval, a visit, an invoice and a payment all belong
  to a *work cycle*. Correlating by cycle is what makes "the approval for the
  work you actually did" checkable, and what lets a warranty revisit be linked
  to the work it repeats instead of looking like a second job.
* **Authoritative records versus assertions.** A verification from an
  authoritative system lands in :attr:`MaintenanceState.authoritative_records`
  and can be cited as evidence. A supplier saying the same thing lands in
  :attr:`MaintenanceState.assertions` and cannot. They are separate containers
  because a single "events" list would make the distinction a matter of reading
  a field carefully.

:func:`validate_canonical_state` lives here rather than in the artefact reader
for the same reason: the shape a reader must hold an untrusted ``final_state`` to
is exactly the shape :meth:`MaintenanceState.canonical` writes, and the two go
out of step the moment they are described in two files. The reader dispatches
here through an operation-type registry, so the root artefact module never learns
what a quote is.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NoReturn

from operatebench.core.clock import parse_timestamp
from operatebench.core.errors import ArtifactError, MalformedTimestampError

#: The phases the operation moves through. Informative in the artefact and used
#: where a guard genuinely depends on it (the provisional window, above all).
PHASES: tuple[str, ...] = (
    "NEW",
    "ACTIVE",
    "WAITING_VISIT",
    "WAITING_APPROVAL",
    "WAITING_EXCEPTION",
    "PROVISIONALLY_CLOSED",
    "FINALIZED",
    "TRANSFERRED_TO_HUMAN",
)

CYCLE_DIAGNOSTIC = "DIAGNOSTIC"
CYCLE_APPROVED_WORK = "APPROVED_WORK"
CYCLE_WARRANTY_REVISIT = "WARRANTY_REVISIT"

VISIT_NONE = "NONE"
VISIT_REQUESTED = "REQUESTED"
VISIT_SCHEDULED = "SCHEDULED"
VISIT_DECLINED = "DECLINED"
VISIT_ATTENDED = "ATTENDED"
VISIT_WORK_REPORTED = "WORK_REPORTED"
VISIT_WORK_VERIFIED = "WORK_VERIFIED"

QUOTE_RECEIVED = "RECEIVED"
QUOTE_APPROVED = "APPROVED"
QUOTE_REJECTED = "REJECTED"
QUOTE_EXPIRED = "EXPIRED"

APPROVAL_OPEN = "OPEN"
APPROVAL_APPROVED = "RESOLVED_APPROVED"
APPROVAL_REJECTED = "RESOLVED_REJECTED"
APPROVAL_EXPIRED = "EXPIRED"

INVOICE_UNVALIDATED = "RECEIVED_UNVALIDATED"
INVOICE_VALIDATION_REQUESTED = "VALIDATION_REQUESTED"
INVOICE_VALIDATED = "VALIDATED"
INVOICE_REJECTED = "REJECTED"

PAYMENT_NOT_REQUESTED = "NOT_REQUESTED"
PAYMENT_REQUESTED = "REQUESTED"
PAYMENT_SETTLED = "SETTLED"

OBLIGATION_OPEN = "OPEN"
OBLIGATION_DISCHARGED = "DISCHARGED"
OBLIGATION_CANCELLED = "CANCELLED"
OBLIGATION_BREACHED = "BREACHED"

RESOLUTION_UNCONFIRMED = "UNCONFIRMED"
RESOLUTION_RESOLVED = "RESOLVED"
RESOLUTION_PERSISTS = "PERSISTS"

#: Exception classes an agent may claim, and the state condition each one needs.
EXCEPTION_TYPES: tuple[str, ...] = (
    "APPROVAL_REJECTED",
    "APPROVAL_EXPIRED",
    "EVIDENCE_CONFLICT",
    "REPEATED_VISIT_FAILURE",
)

#: The business terminals this state machine can actually enter, and therefore
#: the only ones a scenario may declare as its expectation. The list is derived
#: from the two transitions that write :attr:`MaintenanceState.terminal` —
#: finalization of a provisional close, and hand-off to an operator who accepted
#: ownership. A class nothing here can produce would be an authored expectation
#: no episode could ever meet, which is why the public methodology documents the
#: deferred classes rather than declaring them here.
PRODUCIBLE_TERMINALS: tuple[str, ...] = (
    "completed_successfully",
    "transferred_to_human_ownership",
)

RESOLUTION_STATUSES: tuple[str, ...] = (
    RESOLUTION_UNCONFIRMED,
    RESOLUTION_RESOLVED,
    RESOLUTION_PERSISTS,
)

CYCLE_KINDS: tuple[str, ...] = (
    CYCLE_DIAGNOSTIC,
    CYCLE_APPROVED_WORK,
    CYCLE_WARRANTY_REVISIT,
)

VISIT_STATUSES: tuple[str, ...] = (
    VISIT_NONE,
    VISIT_REQUESTED,
    VISIT_SCHEDULED,
    VISIT_DECLINED,
    VISIT_ATTENDED,
    VISIT_WORK_REPORTED,
    VISIT_WORK_VERIFIED,
)

QUOTE_STATUSES: tuple[str, ...] = (
    QUOTE_RECEIVED,
    QUOTE_APPROVED,
    QUOTE_REJECTED,
    QUOTE_EXPIRED,
)

APPROVAL_STATUSES: tuple[str, ...] = (
    APPROVAL_OPEN,
    APPROVAL_APPROVED,
    APPROVAL_REJECTED,
    APPROVAL_EXPIRED,
)

EXCEPTION_STATUSES: tuple[str, ...] = ("OPEN", "RESOLVED")

INVOICE_STATUSES: tuple[str, ...] = (
    INVOICE_UNVALIDATED,
    INVOICE_VALIDATION_REQUESTED,
    INVOICE_VALIDATED,
    INVOICE_REJECTED,
)

OBLIGATION_STATUSES: tuple[str, ...] = (
    OBLIGATION_OPEN,
    OBLIGATION_DISCHARGED,
    OBLIGATION_CANCELLED,
    OBLIGATION_BREACHED,
)

PAYMENT_STATUSES: tuple[str, ...] = (
    PAYMENT_NOT_REQUESTED,
    PAYMENT_REQUESTED,
    PAYMENT_SETTLED,
)


@dataclass
class Cycle:
    """One unit of work: what was asked for, who is doing it, whether it is done."""

    cycle_id: str
    kind: str
    parent_cycle_id: str | None = None
    scope_digest: str | None = None
    visit_id: str | None = None
    visit_status: str = VISIT_NONE
    reported_outcome: str | None = None
    work_evidence_id: str | None = None
    requires_payment: bool = False
    completion_notice_sent: bool = False

    @property
    def work_verified(self) -> bool:
        return self.visit_status == VISIT_WORK_VERIFIED

    def as_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "kind": self.kind,
            "parent_cycle_id": self.parent_cycle_id,
            "scope_digest": self.scope_digest,
            "visit_id": self.visit_id,
            "visit_status": self.visit_status,
            "reported_outcome": self.reported_outcome,
            "work_evidence_id": self.work_evidence_id,
            "requires_payment": self.requires_payment,
            "completion_notice_sent": self.completion_notice_sent,
        }


@dataclass
class Quote:
    """A priced proposal, identified by ``(quote_id, version)`` and bound to a cycle."""

    quote_id: str
    version: int
    cycle_id: str
    amount_minor: int
    currency: str
    scope_digest: str
    status: str = QUOTE_RECEIVED
    valid_until: str | None = None

    @property
    def key(self) -> str:
        return quote_key(self.quote_id, self.version)

    def as_dict(self) -> dict[str, Any]:
        return {
            "quote_id": self.quote_id,
            "version": self.version,
            "cycle_id": self.cycle_id,
            "amount_minor": self.amount_minor,
            "currency": self.currency,
            "scope_digest": self.scope_digest,
            "status": self.status,
            "valid_until": self.valid_until,
        }


def quote_key(quote_id: str, version: int) -> str:
    """Quote identity is the pair, never the id alone: a revision is a new quote."""
    return f"{quote_id}:v{version}"


@dataclass
class Approval:
    """A business approval checkpoint, bound to exactly one quote version."""

    checkpoint_id: str
    quote_id: str
    quote_version: int
    cycle_id: str
    amount_minor: int
    currency: str
    scope_digest: str
    opened_at: str
    deadline_at: str
    status: str = APPROVAL_OPEN
    resolved_at: str | None = None
    reminder_fired: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "quote_id": self.quote_id,
            "quote_version": self.quote_version,
            "cycle_id": self.cycle_id,
            "amount_minor": self.amount_minor,
            "currency": self.currency,
            "scope_digest": self.scope_digest,
            "opened_at": self.opened_at,
            "deadline_at": self.deadline_at,
            "status": self.status,
            "resolved_at": self.resolved_at,
            "reminder_fired": self.reminder_fired,
        }


@dataclass
class ExceptionCheckpoint:
    """A typed hand-off to a human operator, with the context it was opened with."""

    checkpoint_id: str
    exception_type: str
    cycle_id: str | None
    opened_at: str
    deadline_at: str
    evidence_refs: tuple[str, ...]
    status: str = "OPEN"
    resolution: str | None = None
    resolved_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "exception_type": self.exception_type,
            "cycle_id": self.cycle_id,
            "opened_at": self.opened_at,
            "deadline_at": self.deadline_at,
            "evidence_refs": list(self.evidence_refs),
            "status": self.status,
            "resolution": self.resolution,
            "resolved_at": self.resolved_at,
        }


@dataclass
class Invoice:
    """A supplier's bill. Arriving early is allowed; becoming payable is not."""

    invoice_id: str
    cycle_id: str
    quote_id: str
    quote_version: int
    amount_minor: int
    currency: str
    status: str = INVOICE_UNVALIDATED
    validation_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "invoice_id": self.invoice_id,
            "cycle_id": self.cycle_id,
            "quote_id": self.quote_id,
            "quote_version": self.quote_version,
            "amount_minor": self.amount_minor,
            "currency": self.currency,
            "status": self.status,
            "validation_id": self.validation_id,
        }


@dataclass
class Obligation:
    """Something that must happen by a stated instant, and whether it did."""

    obligation_id: str
    kind: str
    cycle_id: str | None
    due_at: str
    status: str = OBLIGATION_OPEN
    closed_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "kind": self.kind,
            "cycle_id": self.cycle_id,
            "due_at": self.due_at,
            "status": self.status,
            "closed_at": self.closed_at,
        }


@dataclass
class MaintenanceState:
    """The whole canonical operation record."""

    phase: str = "NEW"
    issue_id: str | None = None
    classification: str | None = None
    customer_resolution: str = RESOLUTION_UNCONFIRMED
    current_cycle_id: str | None = None
    cycles: dict[str, Cycle] = field(default_factory=dict)
    quotes: dict[str, Quote] = field(default_factory=dict)
    approvals: dict[str, Approval] = field(default_factory=dict)
    exceptions: dict[str, ExceptionCheckpoint] = field(default_factory=dict)
    invoices: dict[str, Invoice] = field(default_factory=dict)
    obligations: dict[str, Obligation] = field(default_factory=dict)
    payment: dict[str, Any] = field(
        default_factory=lambda: {
            "request_id": None,
            "invoice_id": None,
            "cycle_id": None,
            "status": PAYMENT_NOT_REQUESTED,
            "settlement_id": None,
        }
    )
    assertions: list[dict[str, Any]] = field(default_factory=list)
    authoritative_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    communications: list[dict[str, Any]] = field(default_factory=list)
    ownership_transferred_to: str | None = None
    transfer_notice_sent: bool = False
    provisional_close: dict[str, Any] = field(
        default_factory=lambda: {
            "cycle_id": None,
            "accepted_at": None,
            "reopen_until": None,
            "timer_event_id": None,
        }
    )
    reopen_count: int = 0
    terminal: dict[str, Any] | None = None
    replay_final: bool = False
    hidden: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- helpers

    @property
    def current_cycle(self) -> Cycle | None:
        if self.current_cycle_id is None:
            return None
        return self.cycles.get(self.current_cycle_id)

    def open_approvals(self) -> list[Approval]:
        return [
            approval
            for approval in self.approvals.values()
            if approval.status == APPROVAL_OPEN
        ]

    def open_exceptions(self) -> list[ExceptionCheckpoint]:
        return [
            checkpoint
            for checkpoint in self.exceptions.values()
            if checkpoint.status == "OPEN"
        ]

    def open_obligations(self) -> list[Obligation]:
        return [
            obligation
            for obligation in self.obligations.values()
            if obligation.status == OBLIGATION_OPEN
        ]

    def quote(self, quote_id: str, version: int) -> Quote | None:
        return self.quotes.get(quote_key(quote_id, version))

    # --------------------------------------------------------- projections

    def canonical(self) -> dict[str, Any]:
        """The full record, including environment truth the agent never sees."""
        payload = self._roots()
        payload["hidden"] = dict(self.hidden)
        return payload

    def project(self, roots: Sequence[str]) -> dict[str, Any]:
        """Exactly the named roots, each built once, in the order asked for.

        The one place a root becomes JSON. Phase 2 takes the normalized
        projection off the agent path entirely, and the read broker projects the
        roots one published tool serves — so a *whole-record* method is no longer
        something the agent path is allowed to reach for, and going through this
        instead of through :meth:`observable` is what makes that checkable: poison
        :meth:`observable` and a correct run is unaffected.
        """
        built = self._roots()
        missing = sorted(set(roots) - set(built))
        if missing:
            raise KeyError(f"no observable root(s) {missing} in this operation record")
        return {root: built[root] for root in roots}

    def observable(self) -> dict[str, Any]:
        """The whole normalized projection. Not on the agent path since Phase 2.

        Kept because the canonical record and the suite are entitled to the whole
        picture; an agent is not, and nothing that builds an observation, plans a
        read or answers one calls this.
        """
        return self._roots()

    def _roots(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "issue_id": self.issue_id,
            "classification": self.classification,
            "customer_resolution": self.customer_resolution,
            "current_cycle_id": self.current_cycle_id,
            "cycles": {
                cycle_id: cycle.as_dict()
                for cycle_id, cycle in sorted(self.cycles.items())
            },
            "quotes": {
                key: quote.as_dict() for key, quote in sorted(self.quotes.items())
            },
            "approvals": {
                key: approval.as_dict()
                for key, approval in sorted(self.approvals.items())
            },
            "exceptions": {
                key: checkpoint.as_dict()
                for key, checkpoint in sorted(self.exceptions.items())
            },
            "invoices": {
                key: invoice.as_dict() for key, invoice in sorted(self.invoices.items())
            },
            "obligations": {
                key: obligation.as_dict()
                for key, obligation in sorted(self.obligations.items())
            },
            "payment": dict(self.payment),
            "assertions": [dict(item) for item in self.assertions],
            "authoritative_records": {
                key: dict(value)
                for key, value in sorted(self.authoritative_records.items())
            },
            "communications": [dict(item) for item in self.communications],
            "ownership_transferred_to": self.ownership_transferred_to,
            "transfer_notice_sent": self.transfer_notice_sent,
            "provisional_close": dict(self.provisional_close),
            "reopen_count": self.reopen_count,
            "terminal": dict(self.terminal) if self.terminal is not None else None,
            "replay_final": self.replay_final,
        }


# ------------------------------------------------- the canonical state as a shape


def _fail(where: str, message: str) -> NoReturn:
    raise ArtifactError(f"{where}: {message}")


def _obj(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(where, f"must be a JSON object, got {type(value).__name__}")
    return value


def _seq(value: Any, where: str) -> Sequence[Any]:
    if not isinstance(value, list):
        _fail(where, f"must be a JSON array, got {type(value).__name__}")
    return value


def _exact(mapping: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        _fail(
            where,
            f"unknown field(s) {unknown}; the canonical state is written by this "
            f"build and carries exactly {sorted(allowed)}",
        )
    missing = sorted(set(allowed) - set(mapping))
    if missing:
        _fail(where, f"missing required field(s) {missing}")


def _txt(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(where, "must be a non-empty string")
    return value


def _opt_txt(value: Any, where: str) -> str | None:
    if value is None:
        return None
    return _txt(value, where)


def _num(value: Any, where: str, *, minimum: int | None = None) -> int:
    # ``bool`` is an ``int`` in Python and a yes/no answer is not a quantity.
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(where, f"must be an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        _fail(where, f"must not be below {minimum}, got {value}")
    return value


def _flag(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        _fail(where, "must be true or false")
    return value


def _one_of(value: Any, allowed: Sequence[str], where: str) -> str:
    if value not in allowed:
        _fail(where, f"{value!r} is not one of {list(allowed)}")
    return str(value)


def _moment(value: Any, where: str) -> str:
    try:
        parse_timestamp(value, where)
    except MalformedTimestampError as exc:
        raise ArtifactError(str(exc)) from exc
    return str(value)


def _opt_moment(value: Any, where: str) -> str | None:
    if value is None:
        return None
    return _moment(value, where)


def _keyed(value: Any, where: str) -> Mapping[str, Any]:
    body = _obj(value, where)
    for key in body:
        if not isinstance(key, str) or not key:
            _fail(where, f"record key {key!r} must be a non-empty string")
    return body


def _json_leaf(value: Any, where: str) -> None:
    """Environment truth is authored freely, but it is still JSON and still typed."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(where, f"object key {key!r} must be a string")
            _json_leaf(item, f"{where}.{key}")
        return
    if isinstance(value, list):
        for position, item in enumerate(value):
            _json_leaf(item, f"{where}[{position}]")
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    _fail(where, f"must be a JSON scalar or container, got {type(value).__name__}")


_CYCLE_FIELDS = (
    "cycle_id",
    "kind",
    "parent_cycle_id",
    "scope_digest",
    "visit_id",
    "visit_status",
    "reported_outcome",
    "work_evidence_id",
    "requires_payment",
    "completion_notice_sent",
)

_QUOTE_FIELDS = (
    "quote_id",
    "version",
    "cycle_id",
    "amount_minor",
    "currency",
    "scope_digest",
    "status",
    "valid_until",
)

_APPROVAL_FIELDS = (
    "checkpoint_id",
    "quote_id",
    "quote_version",
    "cycle_id",
    "amount_minor",
    "currency",
    "scope_digest",
    "opened_at",
    "deadline_at",
    "status",
    "resolved_at",
    "reminder_fired",
)

_EXCEPTION_FIELDS = (
    "checkpoint_id",
    "exception_type",
    "cycle_id",
    "opened_at",
    "deadline_at",
    "evidence_refs",
    "status",
    "resolution",
    "resolved_at",
)

_INVOICE_FIELDS = (
    "invoice_id",
    "cycle_id",
    "quote_id",
    "quote_version",
    "amount_minor",
    "currency",
    "status",
    "validation_id",
)

_OBLIGATION_FIELDS = (
    "obligation_id",
    "kind",
    "cycle_id",
    "due_at",
    "status",
    "closed_at",
)

_PAYMENT_FIELDS = ("request_id", "invoice_id", "cycle_id", "status", "settlement_id")

_ASSERTION_FIELDS = ("assertion_id", "actor_id", "assertion", "cycle_id", "at")

_COMMUNICATION_FIELDS = (
    "recipient_actor_id",
    "message_fixture_id",
    "correlation_id",
    "at",
)

_PROVISIONAL_FIELDS = ("cycle_id", "accepted_at", "reopen_until", "timer_event_id")

_TERMINAL_FIELDS = ("outcome", "at", "cycle_id")

#: The authoritative record variants, keyed by the ``kind`` each one declares.
#: A record whose kind this build does not write is refused rather than read as
#: whichever variant happens to share its fields.
_AUTHORITATIVE_VARIANTS: Mapping[str, tuple[str, ...]] = {
    "visit_attendance": ("kind", "cycle_id", "visit_id", "at"),
    "work_evidence": ("kind", "cycle_id", "visit_id", "scope_digest", "outcome", "at"),
    "invoice_validation": (
        "kind",
        "invoice_id",
        "cycle_id",
        "valid",
        "reason_code",
        "at",
    ),
    "payment_settlement": (
        "kind",
        "payment_request_id",
        "invoice_id",
        "cycle_id",
        "amount_minor",
        "currency",
        "at",
    ),
}

CANONICAL_STATE_FIELDS: tuple[str, ...] = (
    "phase",
    "issue_id",
    "classification",
    "customer_resolution",
    "current_cycle_id",
    "cycles",
    "quotes",
    "approvals",
    "exceptions",
    "invoices",
    "obligations",
    "payment",
    "assertions",
    "authoritative_records",
    "communications",
    "ownership_transferred_to",
    "transfer_notice_sent",
    "provisional_close",
    "reopen_count",
    "terminal",
    "replay_final",
    "hidden",
)


def _validate_cycle(body: Any, where: str, key: str) -> None:
    entry = _obj(body, where)
    _exact(entry, _CYCLE_FIELDS, where)
    cycle_id = _txt(entry["cycle_id"], f"{where}.cycle_id")
    if cycle_id != key:
        _fail(
            where,
            f"is filed under {key!r} but calls itself {cycle_id!r}; a record and the "
            "identity it is stored against are the same fact",
        )
    _one_of(entry["kind"], CYCLE_KINDS, f"{where}.kind")
    _opt_txt(entry["parent_cycle_id"], f"{where}.parent_cycle_id")
    _opt_txt(entry["scope_digest"], f"{where}.scope_digest")
    _opt_txt(entry["visit_id"], f"{where}.visit_id")
    _one_of(entry["visit_status"], VISIT_STATUSES, f"{where}.visit_status")
    _opt_txt(entry["reported_outcome"], f"{where}.reported_outcome")
    _opt_txt(entry["work_evidence_id"], f"{where}.work_evidence_id")
    _flag(entry["requires_payment"], f"{where}.requires_payment")
    _flag(entry["completion_notice_sent"], f"{where}.completion_notice_sent")


def _validate_quote(body: Any, where: str, key: str) -> None:
    entry = _obj(body, where)
    _exact(entry, _QUOTE_FIELDS, where)
    quote_id = _txt(entry["quote_id"], f"{where}.quote_id")
    version = _num(entry["version"], f"{where}.version", minimum=1)
    if quote_key(quote_id, version) != key:
        _fail(
            where,
            f"is filed under {key!r} but identifies itself as "
            f"{quote_key(quote_id, version)!r}; a revision is a different quote",
        )
    _txt(entry["cycle_id"], f"{where}.cycle_id")
    _num(entry["amount_minor"], f"{where}.amount_minor", minimum=0)
    _txt(entry["currency"], f"{where}.currency")
    _txt(entry["scope_digest"], f"{where}.scope_digest")
    _one_of(entry["status"], QUOTE_STATUSES, f"{where}.status")
    _opt_moment(entry["valid_until"], f"{where}.valid_until")


def _validate_approval(body: Any, where: str, key: str) -> None:
    entry = _obj(body, where)
    _exact(entry, _APPROVAL_FIELDS, where)
    if _txt(entry["checkpoint_id"], f"{where}.checkpoint_id") != key:
        _fail(where, f"is filed under {key!r} but names another checkpoint")
    _txt(entry["quote_id"], f"{where}.quote_id")
    _num(entry["quote_version"], f"{where}.quote_version", minimum=1)
    _txt(entry["cycle_id"], f"{where}.cycle_id")
    _num(entry["amount_minor"], f"{where}.amount_minor", minimum=0)
    _txt(entry["currency"], f"{where}.currency")
    _txt(entry["scope_digest"], f"{where}.scope_digest")
    _moment(entry["opened_at"], f"{where}.opened_at")
    _moment(entry["deadline_at"], f"{where}.deadline_at")
    _one_of(entry["status"], APPROVAL_STATUSES, f"{where}.status")
    _opt_moment(entry["resolved_at"], f"{where}.resolved_at")
    _flag(entry["reminder_fired"], f"{where}.reminder_fired")


def _validate_exception(body: Any, where: str, key: str) -> None:
    entry = _obj(body, where)
    _exact(entry, _EXCEPTION_FIELDS, where)
    if _txt(entry["checkpoint_id"], f"{where}.checkpoint_id") != key:
        _fail(where, f"is filed under {key!r} but names another checkpoint")
    _one_of(entry["exception_type"], EXCEPTION_TYPES, f"{where}.exception_type")
    _opt_txt(entry["cycle_id"], f"{where}.cycle_id")
    _moment(entry["opened_at"], f"{where}.opened_at")
    _moment(entry["deadline_at"], f"{where}.deadline_at")
    for position, ref in enumerate(
        _seq(entry["evidence_refs"], f"{where}.evidence_refs")
    ):
        _txt(ref, f"{where}.evidence_refs[{position}]")
    _one_of(entry["status"], EXCEPTION_STATUSES, f"{where}.status")
    _opt_txt(entry["resolution"], f"{where}.resolution")
    _opt_moment(entry["resolved_at"], f"{where}.resolved_at")


def _validate_invoice(body: Any, where: str, key: str) -> None:
    entry = _obj(body, where)
    _exact(entry, _INVOICE_FIELDS, where)
    if _txt(entry["invoice_id"], f"{where}.invoice_id") != key:
        _fail(where, f"is filed under {key!r} but names another invoice")
    _txt(entry["cycle_id"], f"{where}.cycle_id")
    _txt(entry["quote_id"], f"{where}.quote_id")
    _num(entry["quote_version"], f"{where}.quote_version", minimum=1)
    _num(entry["amount_minor"], f"{where}.amount_minor", minimum=0)
    _txt(entry["currency"], f"{where}.currency")
    _one_of(entry["status"], INVOICE_STATUSES, f"{where}.status")
    _opt_txt(entry["validation_id"], f"{where}.validation_id")


def _validate_obligation(body: Any, where: str, key: str) -> None:
    entry = _obj(body, where)
    _exact(entry, _OBLIGATION_FIELDS, where)
    if _txt(entry["obligation_id"], f"{where}.obligation_id") != key:
        _fail(where, f"is filed under {key!r} but names another obligation")
    _txt(entry["kind"], f"{where}.kind")
    _opt_txt(entry["cycle_id"], f"{where}.cycle_id")
    _moment(entry["due_at"], f"{where}.due_at")
    _one_of(entry["status"], OBLIGATION_STATUSES, f"{where}.status")
    _opt_moment(entry["closed_at"], f"{where}.closed_at")


def _validate_authoritative(body: Any, where: str) -> None:
    entry = _obj(body, where)
    kind = entry.get("kind")
    fields = _AUTHORITATIVE_VARIANTS.get(str(kind))
    if fields is None:
        _fail(
            f"{where}.kind",
            f"{kind!r} is not an authoritative record kind this build writes; known "
            f"kinds are {sorted(_AUTHORITATIVE_VARIANTS)}",
        )
    _exact(entry, fields, where)
    for name in fields:
        place = f"{where}.{name}"
        if name == "at":
            _moment(entry[name], place)
        elif name == "valid":
            _flag(entry[name], place)
        elif name == "amount_minor":
            _num(entry[name], place, minimum=0)
        elif name == "reason_code":
            if not isinstance(entry[name], str):
                _fail(place, "must be a string")
        else:
            _txt(entry[name], place)


def validate_canonical_state(payload: Any, where: str) -> None:
    """Hold an untrusted ``final_state`` to the exact shape this operation writes.

    Recursive by necessity. Checking that the outer value is a mapping proves
    nothing about ``cycles``, and a reader that walks into a list where a record
    registry belongs is one ``AttributeError`` away from a traceback an operator
    cannot act on. Every container, every record variant, every enumerated status,
    every instant and every counter is named here, and an unknown or missing
    nested field is an :class:`~operatebench.core.errors.ArtifactError` rather
    than something a later comparison silently disagrees about.
    """
    body = _obj(payload, where)
    _exact(body, CANONICAL_STATE_FIELDS, where)

    _one_of(body["phase"], PHASES, f"{where}.phase")
    _opt_txt(body["issue_id"], f"{where}.issue_id")
    _opt_txt(body["classification"], f"{where}.classification")
    _one_of(
        body["customer_resolution"], RESOLUTION_STATUSES, f"{where}.customer_resolution"
    )
    _opt_txt(body["current_cycle_id"], f"{where}.current_cycle_id")

    for name, validator in (
        ("cycles", _validate_cycle),
        ("quotes", _validate_quote),
        ("approvals", _validate_approval),
        ("exceptions", _validate_exception),
        ("invoices", _validate_invoice),
        ("obligations", _validate_obligation),
    ):
        registry = _keyed(body[name], f"{where}.{name}")
        for key, entry in registry.items():
            validator(entry, f"{where}.{name}[{key!r}]", key)

    payment = _obj(body["payment"], f"{where}.payment")
    _exact(payment, _PAYMENT_FIELDS, f"{where}.payment")
    _opt_txt(payment["request_id"], f"{where}.payment.request_id")
    _opt_txt(payment["invoice_id"], f"{where}.payment.invoice_id")
    _opt_txt(payment["cycle_id"], f"{where}.payment.cycle_id")
    _one_of(payment["status"], PAYMENT_STATUSES, f"{where}.payment.status")
    _opt_txt(payment["settlement_id"], f"{where}.payment.settlement_id")

    for position, item in enumerate(_seq(body["assertions"], f"{where}.assertions")):
        place = f"{where}.assertions[{position}]"
        entry = _obj(item, place)
        _exact(entry, _ASSERTION_FIELDS, place)
        _txt(entry["assertion_id"], f"{place}.assertion_id")
        _txt(entry["actor_id"], f"{place}.actor_id")
        _txt(entry["assertion"], f"{place}.assertion")
        _opt_txt(entry["cycle_id"], f"{place}.cycle_id")
        _moment(entry["at"], f"{place}.at")

    records = _keyed(body["authoritative_records"], f"{where}.authoritative_records")
    for key, item in records.items():
        _validate_authoritative(item, f"{where}.authoritative_records[{key!r}]")

    for position, item in enumerate(
        _seq(body["communications"], f"{where}.communications")
    ):
        place = f"{where}.communications[{position}]"
        entry = _obj(item, place)
        _exact(entry, _COMMUNICATION_FIELDS, place)
        _txt(entry["recipient_actor_id"], f"{place}.recipient_actor_id")
        _txt(entry["message_fixture_id"], f"{place}.message_fixture_id")
        _opt_txt(entry["correlation_id"], f"{place}.correlation_id")
        _moment(entry["at"], f"{place}.at")

    _opt_txt(body["ownership_transferred_to"], f"{where}.ownership_transferred_to")
    _flag(body["transfer_notice_sent"], f"{where}.transfer_notice_sent")

    provisional = _obj(body["provisional_close"], f"{where}.provisional_close")
    _exact(provisional, _PROVISIONAL_FIELDS, f"{where}.provisional_close")
    _opt_txt(provisional["cycle_id"], f"{where}.provisional_close.cycle_id")
    _opt_moment(provisional["accepted_at"], f"{where}.provisional_close.accepted_at")
    _opt_moment(provisional["reopen_until"], f"{where}.provisional_close.reopen_until")
    _opt_txt(provisional["timer_event_id"], f"{where}.provisional_close.timer_event_id")

    _num(body["reopen_count"], f"{where}.reopen_count", minimum=0)

    if body["terminal"] is not None:
        terminal = _obj(body["terminal"], f"{where}.terminal")
        _exact(terminal, _TERMINAL_FIELDS, f"{where}.terminal")
        _one_of(terminal["outcome"], PRODUCIBLE_TERMINALS, f"{where}.terminal.outcome")
        _moment(terminal["at"], f"{where}.terminal.at")
        _opt_txt(terminal["cycle_id"], f"{where}.terminal.cycle_id")

    _flag(body["replay_final"], f"{where}.replay_final")
    _json_leaf(_obj(body["hidden"], f"{where}.hidden"), f"{where}.hidden")


__all__ = [
    "APPROVAL_APPROVED",
    "APPROVAL_EXPIRED",
    "APPROVAL_OPEN",
    "APPROVAL_REJECTED",
    "APPROVAL_STATUSES",
    "CANONICAL_STATE_FIELDS",
    "CYCLE_APPROVED_WORK",
    "CYCLE_DIAGNOSTIC",
    "CYCLE_KINDS",
    "CYCLE_WARRANTY_REVISIT",
    "EXCEPTION_STATUSES",
    "EXCEPTION_TYPES",
    "INVOICE_REJECTED",
    "INVOICE_STATUSES",
    "INVOICE_UNVALIDATED",
    "INVOICE_VALIDATED",
    "INVOICE_VALIDATION_REQUESTED",
    "OBLIGATION_BREACHED",
    "OBLIGATION_CANCELLED",
    "OBLIGATION_DISCHARGED",
    "OBLIGATION_OPEN",
    "OBLIGATION_STATUSES",
    "PAYMENT_NOT_REQUESTED",
    "PAYMENT_REQUESTED",
    "PAYMENT_SETTLED",
    "PAYMENT_STATUSES",
    "PHASES",
    "PRODUCIBLE_TERMINALS",
    "QUOTE_APPROVED",
    "QUOTE_EXPIRED",
    "QUOTE_RECEIVED",
    "QUOTE_REJECTED",
    "QUOTE_STATUSES",
    "RESOLUTION_PERSISTS",
    "RESOLUTION_RESOLVED",
    "RESOLUTION_STATUSES",
    "RESOLUTION_UNCONFIRMED",
    "VISIT_ATTENDED",
    "VISIT_DECLINED",
    "VISIT_NONE",
    "VISIT_REQUESTED",
    "VISIT_SCHEDULED",
    "VISIT_STATUSES",
    "VISIT_WORK_REPORTED",
    "VISIT_WORK_VERIFIED",
    "Approval",
    "Cycle",
    "ExceptionCheckpoint",
    "Invoice",
    "MaintenanceState",
    "Obligation",
    "Quote",
    "quote_key",
    "validate_canonical_state",
]
