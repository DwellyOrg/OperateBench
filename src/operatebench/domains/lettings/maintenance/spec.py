"""The Maintenance OperationSpec: strict loading, authority binding, identity.

Three things this module refuses to do.

It does not guess. Every key is enumerated; an unknown field is an
:class:`~operatebench.core.errors.UnknownFieldError` rather than something a
later reader silently ignores. A spec that a future field would change the
meaning of must fail on the build that cannot honour it.

It does not let an actor assert its own authority. An event type names the
authority it requires, the actor registry says who holds it, and an authored
event whose actor lacks it is refused when the spec loads — so "the supplier
verified its own work" is impossible to author, not merely impolite.

It does not interpret prose. Amounts are integer minor units, instants are
canonical UTC, scope is an opaque digest, and message wording is a fixture id.
There is nothing here for a language model to be persuasive about.

Identity is a SHA-256 over the canonical JSON of the whole authored mapping, so
reordering keys or reflowing YAML leaves it alone and changing one authored
amount does not.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

from boundarybench.jsonsafe import JsonSafetyError, canonical_json_bytes
from boundarybench.loader import read_yaml_mapping
from boundarybench.schema import SchemaError as BoundarySchemaError
from operatebench.core.clock import parse_timestamp
from operatebench.core.errors import (
    AuthorityError,
    DuplicateIdentityError,
    SpecFormatError,
    SpecIdentityError,
    SpecSchemaError,
    UnknownFieldError,
    UnknownTypeError,
)
from operatebench.core.read_contract import (
    COMPLETE_OUTCOME_KEY,
    READ_REQUIRED,
    ActionEvidenceContract,
    PayloadRegistryKey,
    QuoteForSelectedApproval,
    WorkEvidenceForInvoice,
    action_evidence_contract_problem,
)
from operatebench.domains.lettings.maintenance.retrieval import (
    MAINTENANCE_RETRIEVAL_TOOLS,
    maintenance_retrieval_catalogue,
)
from operatebench.domains.lettings.maintenance.state import PRODUCIBLE_TERMINALS

#: Every event type the Maintenance vertical knows, mapped to the authority an
#: actor must hold to emit it. This mapping *is* the provenance contract.
#:
#: It is also a completeness claim: every type in here has an executable reducer
#: in :mod:`operatebench.domains.lettings.maintenance.operation`, asserted by the
#: test suite. A type this build accepts but cannot execute would be a schema
#: that promises more than the engine honours, which is why
#: ``supplier_visit_status`` was removed rather than left declared — no shipped
#: scenario exercises it and nothing reduces it. The public methodology records
#: that deferred surface rather than leaving it in the executable vocabulary.
MAINTENANCE_EVENT_TYPES: Mapping[str, str] = {
    "customer_issue_reported": "report_issue",
    "customer_resolution_reported": "confirm_resolution",
    "supplier_visit_response": "respond_to_visit",
    "supplier_work_reported": "report_work",
    "supplier_quote_received": "submit_quote",
    "supplier_invoice_received": "submit_invoice",
    "supplier_assertion_received": "assert_status",
    "approver_decision_received": "approve_repair_quote",
    "operator_exception_resolved": "resolve_exception",
    "visit_attendance_verified": "verify_attendance",
    "work_evidence_verified": "verify_work",
    "invoice_validation_completed": "validate_invoice",
    "payment_settlement_confirmed": "confirm_settlement",
    "approval_reminder_due": "emit_timer",
    "approval_deadline_due": "emit_timer",
    "visit_followup_due": "emit_timer",
    "provisional_close_expired": "emit_timer",
}

#: Every action the agent may propose. ``complete`` is the pseudo-action the
#: engine records when a terminal proposal is accepted, so a scenario can hang a
#: conditional event off a closure the same way it hangs one off a visit request.
MAINTENANCE_ACTION_TYPES: tuple[str, ...] = (
    "request_supplier_visit",
    "request_approval",
    "send_message",
    "authorise_supplier_work",
    "request_invoice_validation",
    "request_payment",
    "request_exception_resolution",
    "complete",
)

#: The terminal classes a scenario may declare as its authored expectation.
#:
#: Taken from the state machine rather than restated here, so "this build accepts
#: it" and "this build can produce it" are one fact. A scenario that declares an
#: unsupported terminal is refused rather than running toward an expectation
#: nothing in
#: the operation could ever have met. They remain deferred architecture classes,
#: documented in the public methodology, not executable scenario expectations.
TERMINAL_OUTCOMES: tuple[str, ...] = PRODUCIBLE_TERMINALS

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "operation_id",
        "operation_type",
        "operation_version",
        "semantic_scenario_id",
        "privacy_status",
        "disclaimer",
        "data_provenance",
        "actors",
        "policy",
        "message_fixtures",
        "hidden_state",
        "scenarios",
    }
)
_ACTOR_FIELDS = frozenset({"role", "authority"})

#: Every policy field, in one place, used for three jobs that must never
#: disagree: what the loader accepts, what it requires, and what the operation's
#: content identity is taken over. Written out rather than derived because this
#: is also the wire schema — the loader refuses a key that is not here, and a
#: schema that reads itself out of the code it validates cannot refuse anything.
#: :data:`POLICY_FIELDS` is proved equal to :class:`PolicySpec` at import time,
#: immediately after that class exists, so the written list cannot drift from
#: the executable one: a field added to the dataclass and left out here fails
#: the import rather than quietly leaving the digest blind to it.
POLICY_FIELDS: tuple[str, ...] = (
    "currency",
    "issue_classification",
    "approval_threshold_minor",
    "approval_reminder_after_minutes",
    "approval_deadline_after_minutes",
    "approval_validity_minutes",
    "provisional_close_minutes",
    "visit_followup_after_minutes",
    "completion_notice_within_minutes",
    "max_turns_per_invocation",
    "horizon_minutes",
    "max_retrieval_batches_per_invocation",
)
_POLICY_FIELDS = frozenset(POLICY_FIELDS)
_SCENARIO_FIELDS = frozenset(
    {
        "label",
        "starts_at",
        "expected_terminal",
        "human_checkpoint_budget",
        "required_checkpoint_types",
        "dispatch_failures",
        "expected_event_rejections",
        "events",
    }
)
_EXPECTED_REJECTION_FIELDS = frozenset({"event_id", "code", "reason"})
_EVENT_FIELDS = frozenset(
    {
        "event_id",
        "type",
        "actor",
        "at",
        "on_action",
        "on_event",
        "delay_minutes",
        "triggers_agent",
        "payload",
    }
)
_TRIGGER_FIELDS = frozenset({"type", "cycle_id"})

#: The primitive kinds a payload field may hold. Deliberately few: an operation
#: whose facts are integer minor units, opaque identifiers and yes/no
#: verifications has nothing to gain from floats or nested containers, and a
#: payload that cannot nest cannot smuggle a structure past a reducer that
#: indexes it as a scalar. ``optional_string`` exists for the one shape an agent
#: legitimately produces — a correlation or scope that is genuinely not known yet.
_PAYLOAD_KINDS: Mapping[str, tuple[type, ...]] = {
    "string": (str,),
    "optional_string": (str,),
    "integer": (int,),
    "boolean": (bool,),
}

#: The exact payload contract of every event type: what a reducer will index
#: into, and nothing else. Required fields are checked for presence and type;
#: optional fields are checked for type when present; anything not named here is
#: an unknown field and is refused.
#:
#: This is the schema whose absence let a spec drop ``cycle_id``, pass
#: ``validate`` and then raise ``KeyError`` from inside a reducer halfway
#: through an episode. Every entry below was read off the reducer that consumes
#: it, so "the spec loaded" and "the reducers can run it" are the same claim.
_EVENT_PAYLOAD_SCHEMAS: Mapping[str, tuple[Mapping[str, str], Mapping[str, str]]] = {
    "customer_issue_reported": (
        {"issue_id": "string", "classification": "string", "cycle_id": "string"},
        {"description_fixture_id": "string"},
    ),
    "customer_resolution_reported": (
        {"issue_id": "string", "status": "string"},
        {"related_visit_id": "string", "cycle_id": "string"},
    ),
    "supplier_visit_response": (
        {"visit_id": "string", "cycle_id": "string", "response": "string"},
        {},
    ),
    "supplier_work_reported": (
        {
            "visit_id": "string",
            "cycle_id": "string",
            "outcome": "string",
            "scope_digest": "string",
        },
        {},
    ),
    "supplier_quote_received": (
        {
            "quote_id": "string",
            "quote_version": "integer",
            "cycle_id": "string",
            "amount_minor": "integer",
            "currency": "string",
            "scope_digest": "string",
        },
        {"parent_cycle_id": "string"},
    ),
    "supplier_invoice_received": (
        {
            "invoice_id": "string",
            "cycle_id": "string",
            "quote_id": "string",
            "quote_version": "integer",
            "amount_minor": "integer",
            "currency": "string",
        },
        {},
    ),
    "supplier_assertion_received": (
        {"assertion_id": "string", "assertion": "string"},
        {"cycle_id": "string"},
    ),
    "approver_decision_received": (
        {
            "checkpoint_id": "string",
            "quote_id": "string",
            "quote_version": "integer",
            "amount_minor": "integer",
            "currency": "string",
            "scope_digest": "string",
            "decision": "string",
        },
        {"cycle_id": "string"},
    ),
    "operator_exception_resolved": (
        {"checkpoint_id": "string", "decision": "string"},
        {"cycle_id": "string"},
    ),
    "visit_attendance_verified": (
        {"visit_id": "string", "cycle_id": "string", "attended": "boolean"},
        {},
    ),
    "work_evidence_verified": (
        {
            "evidence_id": "string",
            "visit_id": "string",
            "cycle_id": "string",
            "scope_digest": "string",
            "outcome": "string",
        },
        {},
    ),
    "invoice_validation_completed": (
        {"validation_id": "string", "invoice_id": "string", "valid": "boolean"},
        {"reason_code": "string", "cycle_id": "string"},
    ),
    # ``cycle_id`` is required, not incidental: a settlement binds to one exact
    # payment request, invoice, cycle, amount and currency, and a field the
    # authored event may omit is a binding the reducer could not check.
    "payment_settlement_confirmed": (
        {
            "settlement_id": "string",
            "payment_request_id": "string",
            "invoice_id": "string",
            "cycle_id": "string",
            "amount_minor": "integer",
            "currency": "string",
        },
        {},
    ),
    "approval_reminder_due": (
        {"checkpoint_id": "string", "cycle_id": "string"},
        {},
    ),
    "approval_deadline_due": (
        {"checkpoint_id": "string", "cycle_id": "string"},
        {},
    ),
    "visit_followup_due": ({"cycle_id": "string"}, {}),
    "provisional_close_expired": ({"cycle_id": "string"}, {}),
}


#: The identities the *environment* allocates, and which therefore appear in no
#: action payload contract below.
#:
#: A visit, a checkpoint and a payment request come into existence because the
#: operation accepted an action, and the authored world answers them by name. If
#: the agent chose those names, then two agents that run the identical operation
#: with different spellings get different worlds: the authored
#: ``supplier_visit_response`` still says ``visit_1``, the operation now holds
#: whatever the agent invented, and the event is refused as ``UNKNOWN_VISIT``.
#: That measures naming, not behaviour. So the agent does not get to name them;
#: it reads them back out of the record after the effect is accepted.
#:
#: Identifiers that genuinely originate with an authored actor — a quote id, an
#: invoice id, an evidence id, a validation id, a settlement id, an assertion id
#: — are the opposite case and stay in the contracts: the agent is *citing* what
#: the world already told it, and inventing one of those is a fabricated
#: reference the evidence guard is there to refuse.
ENVIRONMENT_ALLOCATED_IDENTITY_FIELDS: frozenset[str] = frozenset(
    {"visit_id", "checkpoint_id", "payment_request_id"}
)

#: The exact payload contract of every action the agent may propose, read off
#: the handler that consumes it exactly as the event table above was read off the
#: reducers. Its absence is what let ``quote_version: []`` reach ``int()`` and
#: leave the public engine boundary as a raw ``TypeError``: missing fields were
#: caught by a ``KeyError`` net, wrong types were not caught at all.
#:
#: ``complete`` is absent deliberately — a terminal proposal carries no payload,
#: and its guard is :meth:`MaintenanceOperation.apply_terminal`.
MAINTENANCE_ACTION_PAYLOAD_SCHEMAS: Mapping[
    str, tuple[Mapping[str, str], Mapping[str, str]]
] = {
    "request_supplier_visit": (
        {"visit_type": "string", "cycle_id": "string"},
        # A diagnostic cycle has no scope until somebody reports one, so the
        # agent may carry the scope it can see and nothing more.
        {"scope_digest": "optional_string"},
    ),
    "request_approval": (
        {
            "quote_id": "string",
            "quote_version": "integer",
            "amount_minor": "integer",
            "currency": "string",
            "scope_digest": "string",
            "cycle_id": "string",
            "deadline_after_minutes": "integer",
        },
        {},
    ),
    "send_message": (
        {"recipient_actor_id": "string", "message_fixture_id": "string"},
        {"correlation_id": "optional_string"},
    ),
    "authorise_supplier_work": (
        {
            "quote_id": "string",
            "quote_version": "integer",
            "approval_checkpoint_id": "string",
            "cycle_id": "string",
        },
        {},
    ),
    "request_invoice_validation": ({"invoice_id": "string"}, {}),
    "request_payment": (
        {
            "invoice_id": "string",
            "amount_minor": "integer",
            "currency": "string",
        },
        {},
    ),
    "request_exception_resolution": (
        {
            "exception_type": "string",
            "approval_checkpoint_id": "string",
            "deadline_after_minutes": "integer",
        },
        {},
    ),
}

#: What each business outcome must have read before it can be proposed, as one
#: canonical contract object.
#:
#: Phase 1A stated this as a plain table here and let three consumers hold their
#: own view of it; Phase 2 makes it a single
#: :class:`~operatebench.core.read_contract.ReadRequirementContract` that the
#: runtime guard, the published action schemas and the evaluator all read. There
#: is no second copy to drift from, and the perturbation tests prove it by
#: changing one requirement and watching both the refusal and the grade move.
#:
#: Every entry is read off the handler that consumes the outcome, exactly as the
#: payload contract above was. ``request_approval`` binds a quote version to a
#: checkpoint and is refused if a checkpoint already holds that version, so it
#: rests on the case record, the quotes and the checkpoints — all three, and an
#: agent that had read two of them would be proposing on a partial view.
#: ``complete`` is a business outcome like any other: :meth:`apply_terminal`
#: refuses on open checkpoints, open obligations, unsettled billing and the case
#: record's own account of the cycle, so those four reads are what it rests on.
#: A WAIT rests on nothing — it changes nothing and commits nothing — so it has
#: no entry, which is how "this outcome requires no read" is stated.
MAINTENANCE_ACTION_EVIDENCE_CONTRACT = ActionEvidenceContract(
    {
        "request_supplier_visit": {
            "reads": {"get_case_record": READ_REQUIRED},
        },
        "request_approval": {
            "reads": {
                "get_case_record": READ_REQUIRED,
                "list_checkpoints": READ_REQUIRED,
                "list_quotes": READ_REQUIRED,
            }
        },
        "send_message": {
            "reads": {
                "get_case_record": READ_REQUIRED,
                "list_communications": READ_REQUIRED,
            }
        },
        "authorise_supplier_work": {
            "reads": {
                "get_case_record": READ_REQUIRED,
                "list_checkpoints": READ_REQUIRED,
                "list_quotes": READ_REQUIRED,
            },
            "evidence_refs": (
                PayloadRegistryKey(
                    "list_checkpoints", "approvals", "approval_checkpoint_id"
                ),
            ),
        },
        "request_invoice_validation": {
            "reads": {
                "get_case_record": READ_REQUIRED,
                "list_authoritative_records": READ_REQUIRED,
                "list_billing": READ_REQUIRED,
            },
            "evidence_refs": (
                WorkEvidenceForInvoice(
                    "list_billing",
                    "invoices",
                    "invoice_id",
                    "get_case_record",
                    "cycles",
                    "list_authoritative_records",
                    "authoritative_records",
                ),
            ),
        },
        "request_payment": {
            "reads": {
                "get_case_record": READ_REQUIRED,
                "list_authoritative_records": READ_REQUIRED,
                "list_billing": READ_REQUIRED,
            },
            "evidence_refs": (
                PayloadRegistryKey("list_billing", "invoices", "invoice_id"),
                WorkEvidenceForInvoice(
                    "list_billing",
                    "invoices",
                    "invoice_id",
                    "get_case_record",
                    "cycles",
                    "list_authoritative_records",
                    "authoritative_records",
                ),
            ),
        },
        "request_exception_resolution": {
            "reads": {
                "get_case_record": READ_REQUIRED,
                "list_checkpoints": READ_REQUIRED,
                "list_quotes": READ_REQUIRED,
            },
            "evidence_refs": (
                PayloadRegistryKey(
                    "list_checkpoints", "approvals", "approval_checkpoint_id"
                ),
                QuoteForSelectedApproval(
                    "list_checkpoints",
                    "approvals",
                    "approval_checkpoint_id",
                    "list_quotes",
                    "quotes",
                ),
            ),
        },
        COMPLETE_OUTCOME_KEY: {
            "reads": {
                "get_case_record": READ_REQUIRED,
                "list_billing": READ_REQUIRED,
                "list_checkpoints": READ_REQUIRED,
                "list_obligations": READ_REQUIRED,
            }
        },
    }
)

# Compatibility name for read-only consumers. It is the same object, not a
# second declaration.
MAINTENANCE_READ_CONTRACT = MAINTENANCE_ACTION_EVIDENCE_CONTRACT


def maintenance_action_evidence_contract() -> ActionEvidenceContract:
    """Return the live canonical declaration for all independent consumers."""
    return MAINTENANCE_ACTION_EVIDENCE_CONTRACT


#: Validated where it is declared, at import, against the published catalogue and
#: the outcomes this operation can actually produce. A requirement naming a read
#: the catalogue does not offer would refuse every proposal of that outcome for a
#: record nobody can fetch; a requirement for an outcome that does not exist is a
#: rule that never runs. Both are drift, and neither survives the import.
_READ_CONTRACT_PROBLEM = action_evidence_contract_problem(
    MAINTENANCE_ACTION_EVIDENCE_CONTRACT,
    catalogue=maintenance_retrieval_catalogue(),
    retrieval_tools=MAINTENANCE_RETRIEVAL_TOOLS,
    payload_schemas=MAINTENANCE_ACTION_PAYLOAD_SCHEMAS,
    outcome_keys=(*MAINTENANCE_ACTION_PAYLOAD_SCHEMAS, COMPLETE_OUTCOME_KEY),
)
if _READ_CONTRACT_PROBLEM is not None:  # pragma: no cover - import-time guard
    raise SpecSchemaError(
        f"the maintenance read contract cannot be executed: {_READ_CONTRACT_PROBLEM}"
    )

#: The authority class an actor's word carries, keyed by the role the actor
#: registry gives it. Derived from the registry rather than from a second table
#: of event types: an event is authoritative because of *who produced it*, and
#: keying it by type would let one actor's claim inherit another's standing by
#: being named similarly.
MAINTENANCE_ROLE_AUTHORITY: Mapping[str, str] = {
    "agent": "actor_claim",
    "approver": "actor_claim",
    "authoritative_external_system": "authoritative_verification",
    "customer": "actor_claim",
    "operator_human": "actor_claim",
    "supplier": "actor_claim",
    "system_scheduler": "system_event",
}

#: The event types that can wake an agent, as a sorted vocabulary.
#:
#: Derived from the executable event table, so it is the operation's whole
#: vocabulary rather than a second list that could drift from it. Deliberately
#: static: it tells an agent what a WAIT may legally name, and it says nothing
#: about what this episode has queued or when — a "reachable right now" set
#: would be the pending queue wearing a different name, and the pending queue is
#: the answer.
MAINTENANCE_WAKE_EVENT_TYPES: tuple[str, ...] = tuple(sorted(MAINTENANCE_EVENT_TYPES))


def action_schema_view() -> dict[str, dict[str, Any]]:
    """The action contracts as detached plain builtins, safe to hand an agent.

    ``reads`` is a schema *part*, a tool-name-to-requirement mapping beside
    ``required`` and ``optional`` rather than a bare list. Two reasons, and the
    second is the load-bearing one: the observation projects each part as a
    mapping, so a list would be projected as one and reach a provider as
    nonsense; and a mapping has room for a requirement other than ``"required"``
    when a later phase needs one, where a list would have to change shape.
    """
    published = {
        action_type: {
            "required": dict(required),
            "optional": dict(optional),
            "reads": maintenance_action_evidence_contract().schema_part(action_type),
            "evidence_refs": maintenance_action_evidence_contract().evidence_schema_part(
                action_type
            ),
        }
        for action_type, (
            required,
            optional,
        ) in MAINTENANCE_ACTION_PAYLOAD_SCHEMAS.items()
    }
    published["send_message"]["field_guidance"] = {
        "correlation_id": (
            "cycle_id (get_case_record/list_obligations): completion: "
            "match-only discharge; mismatch accepted; approval reminder: "
            "open/due approval; transfer: current/transferred; else optional. "
            "Recheck open obligations after each wake: a new completion duty "
            "requires a newly delivered, correctly correlated notice even if an "
            "identical notice was accepted earlier. Early notices are permitted; "
            "they do not discharge duties created later. Repeats without a new "
            "duty remain redundant."
        ),
        "recipient_actor_id": (
            "completion/transfer notices: "
            "get_case_record.issue_reporting_actor_id; "
            "approval reminder: approver_1; else any"
        ),
    }
    # The terminal is published beside the actions. It carries no payload — that
    # is why it is absent from the payload table — but its correctness depends on
    # the record exactly as an action's does, so its read requirement is
    # disclosed rather than discovered by being refused.
    published[COMPLETE_OUTCOME_KEY] = {
        "field_guidance": {
            "operation": (
                "A valid declared wait may be interrupted by unsolicited runtime "
                "events outside wake_on; this is informational, not a WAIT failure. "
                "Recheck current records and obligations on every wake. "
                "Open obligations must still be discharged before completion."
            ),
        },
        "required": {},
        "optional": {},
        "reads": maintenance_action_evidence_contract().schema_part(COMPLETE_OUTCOME_KEY),
        "evidence_refs": maintenance_action_evidence_contract().evidence_schema_part(
            COMPLETE_OUTCOME_KEY
        ),
    }
    return published


def _validate_payload(
    schema: tuple[Mapping[str, str], Mapping[str, str]],
    payload: Mapping[str, Any],
    where: str,
    what: str,
) -> None:
    required, optional = schema
    allowed = frozenset(required) | frozenset(optional)
    _reject_unknown(payload, allowed, f"{where}: payload")
    missing = sorted(set(required) - set(payload))
    if missing:
        raise SpecSchemaError(
            f"{where}: payload for {what!r} is missing required field(s) "
            f"{missing}; the handler indexes these, so a payload without them "
            "cannot be executed"
        )
    for name, kind in (*required.items(), *optional.items()):
        if name not in payload:
            continue
        value = payload[name]
        if kind == "optional_string" and value is None:
            continue
        expected = _PAYLOAD_KINDS[kind]
        # ``bool`` is a subclass of ``int``: payloads such as ``attended: 1`` and
        # ``quote_version: true`` are adversarial type confusion, so integer
        # validation must explicitly reject booleans.
        if kind == "integer" and isinstance(value, bool):
            raise SpecSchemaError(
                f"{where}: payload field {name!r} must be an integer, got a boolean; "
                "a yes/no answer is not a quantity"
            )
        if kind in {"string", "optional_string"} and isinstance(value, bool):
            raise SpecSchemaError(
                f"{where}: payload field {name!r} must be a string, got a boolean"
            )
        if not isinstance(value, expected):
            raise SpecSchemaError(
                f"{where}: payload field {name!r} must be a {kind}, got "
                f"{type(value).__name__}"
            )
        if kind in {"string", "optional_string"} and not value:
            raise SpecSchemaError(
                f"{where}: payload field {name!r} must be a non-empty string"
            )


def validate_event_payload(
    event_type: str, payload: Mapping[str, Any], where: str
) -> None:
    """Hold one event payload to the exact contract its type declares.

    Raised as :class:`~operatebench.core.errors.SpecSchemaError` (or the
    :class:`~operatebench.core.errors.UnknownFieldError` subclass for a field
    this build does not know), so a caller catching one named error catches
    every way a payload can be wrong.
    """
    schema = _EVENT_PAYLOAD_SCHEMAS.get(event_type)
    if schema is None:
        raise UnknownTypeError(
            f"{where}: {event_type!r} has no payload schema in this build; an event "
            "type this build accepts must state exactly what its payload carries"
        )
    _validate_payload(schema, payload, where, event_type)


def validate_action_payload(
    action_type: str, payload: Mapping[str, Any], where: str
) -> None:
    """Hold one proposed action payload to the exact contract its type declares.

    The action-side twin of :func:`validate_event_payload`, and for the same
    reason: a handler that indexes a field is entitled to a field of the right
    primitive type, and the place to say so is once, in a table, rather than in
    whichever ``int()`` call the handler happens to reach first.
    """
    schema = MAINTENANCE_ACTION_PAYLOAD_SCHEMAS.get(action_type)
    if schema is None:
        raise UnknownTypeError(
            f"{where}: {action_type!r} has no payload schema in this build; an action "
            "this build accepts must state exactly what its payload carries"
        )
    _validate_payload(schema, payload, where, action_type)


#: The only privacy status this build will execute. A spec that claims anything
#: else is refused rather than run with a disclaimer.
REQUIRED_PRIVACY_STATUS = "SYNTHETIC_ONLY"

SUPPORTED_SCHEMA_VERSION = 1


def _reject_unknown(
    mapping: Mapping[str, Any], allowed: frozenset[str], where: str
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise UnknownFieldError(
            f"{where}: unknown field(s) {unknown}; this build refuses a spec it "
            f"cannot fully honour. Known fields are {sorted(allowed)}"
        )


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpecSchemaError(f"{where}: expected a mapping, got {type(value).__name__}")
    return value


def _require_text(mapping: Mapping[str, Any], key: str, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise SpecSchemaError(f"{where}: {key!r} must be a non-empty string")
    return value


def _require_int(mapping: Mapping[str, Any], key: str, where: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecSchemaError(f"{where}: {key!r} must be an integer")
    return value


def _require_unique(values: Sequence[str], where: str) -> tuple[str, ...]:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise DuplicateIdentityError(f"{where}: {value!r} is declared twice")
        seen.add(value)
    return tuple(values)


@dataclass(frozen=True)
class ActorSpec:
    """One actor, its role, and exactly what it is allowed to bring about."""

    actor_id: str
    role: str
    authority: frozenset[str]

    def holds(self, authority: str) -> bool:
        return authority in self.authority


@dataclass(frozen=True)
class PolicySpec:
    """Invented, executable policy constants. Every number is a fixture."""

    currency: str
    issue_classification: str
    approval_threshold_minor: int
    approval_reminder_after_minutes: int
    approval_deadline_after_minutes: int
    approval_validity_minutes: int
    provisional_close_minutes: int
    visit_followup_after_minutes: int
    completion_notice_within_minutes: int
    max_turns_per_invocation: int
    horizon_minutes: int
    #: How many retrieval batches one invocation may have served. A separate
    #: bound from ``max_turns_per_invocation``, which counts business outcomes:
    #: one budget for reading and one for deciding, so a careful read never costs
    #: the operation a turn it needed to act.
    max_retrieval_batches_per_invocation: int


#: The executable policy, as the dataclass actually carries it. ``fields()``
#: rather than ``__dataclass_fields__``: the latter also reports ``ClassVar``
#: and ``InitVar`` pseudo-fields, which are not instance data and have no place
#: in a content identity — and an ``InitVar`` would hash its class default while
#: reading like a real field.
_EXECUTABLE_POLICY_FIELDS = frozenset(entry.name for entry in fields(PolicySpec))

if _EXECUTABLE_POLICY_FIELDS != _POLICY_FIELDS:
    # Import-time, not test-time: a policy field the schema does not name is a
    # field the loader cannot accept and the digest cannot see, and a build that
    # can compute a blind identity must not start. Naming both sides of the
    # difference means the message says which direction the drift went.
    raise RuntimeError(
        "maintenance policy schema and PolicySpec disagree: "
        f"missing from POLICY_FIELDS "
        f"{sorted(_EXECUTABLE_POLICY_FIELDS - _POLICY_FIELDS)}, "
        f"absent from PolicySpec "
        f"{sorted(_POLICY_FIELDS - _EXECUTABLE_POLICY_FIELDS)}"
    )


@dataclass(frozen=True)
class ConditionalTrigger:
    """What an authored event is waiting for before it can be scheduled."""

    kind: str  # "action" or "event"
    type_name: str
    cycle_id: str | None

    def matches(self, kind: str, type_name: str, cycle_id: str | None) -> bool:
        if self.kind != kind or self.type_name != type_name:
            return False
        return self.cycle_id is None or self.cycle_id == cycle_id


@dataclass(frozen=True)
class AuthoredEvent:
    """One authored external event: unconditional at an instant, or conditional.

    Exactly one timing form. An event that declares both an absolute instant and
    a cause has two different answers to "when", and a scenario that can be read
    two ways is not frozen.
    """

    event_id: str
    event_type: str
    actor_id: str
    sequence: int
    payload: Mapping[str, Any]
    triggers_agent: bool
    at: str | None
    trigger: ConditionalTrigger | None
    delay_minutes: int

    @property
    def cycle_id(self) -> str | None:
        value = self.payload.get("cycle_id")
        return value if isinstance(value, str) else None

    def payload_snapshot(self) -> dict[str, Any]:
        """A detached plain-builtin copy, safe to hand to Core and to mutate."""
        payload: dict[str, Any] = _plain(self.payload)
        return payload

    def semantic_payload(self) -> dict[str, Any]:
        """Everything about this event that decides what an episode does."""
        return {
            "event_id": self.event_id,
            "type": self.event_type,
            "actor": self.actor_id,
            "sequence": self.sequence,
            "payload": self.payload_snapshot(),
            "triggers_agent": self.triggers_agent,
            "at": self.at,
            "trigger": (
                None
                if self.trigger is None
                else {
                    "kind": self.trigger.kind,
                    "type": self.trigger.type_name,
                    "cycle_id": self.trigger.cycle_id,
                }
            ),
            "delay_minutes": self.delay_minutes,
        }


@dataclass(frozen=True)
class ExpectedEventRejection:
    """One authored delivery this scenario expects the operation to refuse.

    Some authored events are *supposed* to bounce: an approver answering after
    the deadline timer already expired the checkpoint, a supplier asserting
    something at an operation that is already replay-final. Those are the
    evidence the scenario exists to produce, and they must stay legal and
    non-mutating.

    Everything else that bounces is the environment coming apart, and the only
    way to tell the two cases apart is for the scenario to say, by event id and
    by refusal code, which ones it meant. Declaring the code and not just the
    event is what stops a declaration becoming a blanket amnesty: an event that
    is refused for a *different* reason than the one authored here is still
    desync.
    """

    event_id: str
    code: str
    reason: str

    def semantic_payload(self) -> dict[str, Any]:
        return {"event_id": self.event_id, "code": self.code, "reason": self.reason}


@dataclass(frozen=True)
class ScenarioSpec:
    """One frozen variant. Same schemas as its siblings; different world."""

    scenario_id: str
    semantic_scenario_id: str
    label: str
    starts_at: str
    expected_terminal: str
    human_checkpoint_budget: int
    required_checkpoint_types: tuple[str, ...]
    dispatch_failures: frozenset[str]
    expected_event_rejections: tuple[ExpectedEventRejection, ...]
    events: tuple[AuthoredEvent, ...]

    def expected_rejection_codes(self) -> dict[str, str]:
        """Event id to the one refusal code this scenario declares for it."""
        return {
            expected.event_id: expected.code
            for expected in self.expected_event_rejections
        }

    @property
    def unconditional_events(self) -> tuple[AuthoredEvent, ...]:
        return tuple(event for event in self.events if event.at is not None)

    @property
    def conditional_events(self) -> tuple[AuthoredEvent, ...]:
        return tuple(event for event in self.events if event.trigger is not None)

    def semantic_payload(self) -> dict[str, Any]:
        """Everything about this variant that decides what an episode does."""
        return {
            "label": self.label,
            "starts_at": self.starts_at,
            "expected_terminal": self.expected_terminal,
            "human_checkpoint_budget": self.human_checkpoint_budget,
            "required_checkpoint_types": list(self.required_checkpoint_types),
            "dispatch_failures": sorted(self.dispatch_failures),
            "expected_event_rejections": [
                expected.semantic_payload() for expected in self.expected_event_rejections
            ],
            "events": [event.semantic_payload() for event in self.events],
        }


@dataclass(frozen=True)
class OperationSpec:
    """A versioned, executable operation definition with a content identity."""

    operation_id: str
    operation_type: str
    operation_version: str
    semantic_scenario_id: str
    privacy_status: str
    disclaimer: str
    data_provenance: str
    actors: Mapping[str, ActorSpec]
    policy: PolicySpec
    message_fixtures: Mapping[str, str]
    hidden_state: Mapping[str, Any]
    scenarios: Mapping[str, ScenarioSpec]
    spec_digest_sha256: str
    source: str = field(default="<memory>", compare=False)

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.scenarios))

    def scenario(self, scenario_id: str) -> ScenarioSpec:
        try:
            return self.scenarios[scenario_id]
        except KeyError:
            raise UnknownTypeError(
                f"{self.source}: no scenario {scenario_id!r} in this operation; "
                f"available scenarios are {list(self.scenario_ids)}"
            ) from None

    def actor(self, actor_id: str) -> ActorSpec:
        try:
            return self.actors[actor_id]
        except KeyError:
            raise UnknownTypeError(
                f"{self.source}: no actor {actor_id!r} in the actor registry; "
                f"known actors are {sorted(self.actors)}"
            ) from None

    def identity_payload(self) -> dict[str, Any]:
        """What the CLI prints and what a run artefact records as spec identity.

        A freshly built plain dict every call: a caller that mutates what it was
        handed changes its own copy and nothing about the spec.
        """
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "operation_version": self.operation_version,
            "semantic_scenario_id": self.semantic_scenario_id,
            "privacy_status": self.privacy_status,
            "spec_digest_sha256": self.spec_digest_sha256,
            "scenario_ids": list(self.scenario_ids),
        }

    def actor_snapshot(self) -> dict[str, dict[str, Any]]:
        """A detached plain projection of the actor registry."""
        return {
            actor_id: {"role": actor.role, "authority": sorted(actor.authority)}
            for actor_id, actor in sorted(self.actors.items())
        }

    def policy_snapshot(self) -> dict[str, Any]:
        """Every policy field, over the one list the loader also reads.

        The projection used to be written out a second time, here, and the two
        enumerations drifted: ``max_retrieval_batches_per_invocation`` was
        accepted by the loader, read by the runtime, and absent from this
        payload, so two operations that execute differently hashed the same and
        one published digest named both. Taking the schema list instead means
        the accepted-key set and the hashed set cannot disagree, and
        :data:`POLICY_FIELDS` is in turn proved equal to :class:`PolicySpec` at
        import time — so a field can be added to the policy in exactly one way:
        all three at once, or not at all.
        """
        return {name: getattr(self.policy, name) for name in POLICY_FIELDS}

    def hidden_state_snapshot(self) -> dict[str, Any]:
        """A detached plain copy of the environment truth the agent never sees."""
        payload: dict[str, Any] = _plain(self.hidden_state)
        return payload

    def message_fixture_snapshot(self) -> dict[str, str]:
        """A detached plain copy of the message fixture table."""
        return dict(self.message_fixtures)

    def semantic_payload(self) -> dict[str, Any]:
        """Every authored fact that decides what this operation does.

        Derived from the parsed structures rather than from the mapping that was
        read, which is what makes :meth:`verify_identity` an *independent*
        recomputation: it hashes what would execute now, not what a loader saw
        once.
        """
        return {
            "schema_version": SUPPORTED_SCHEMA_VERSION,
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "operation_version": self.operation_version,
            "semantic_scenario_id": self.semantic_scenario_id,
            "privacy_status": self.privacy_status,
            "disclaimer": self.disclaimer,
            "data_provenance": self.data_provenance,
            "actors": self.actor_snapshot(),
            "policy": self.policy_snapshot(),
            "message_fixtures": self.message_fixture_snapshot(),
            "hidden_state": self.hidden_state_snapshot(),
            "scenarios": {
                scenario_id: self.scenarios[scenario_id].semantic_payload()
                for scenario_id in self.scenario_ids
            },
        }

    def compute_digest(self) -> str:
        """SHA-256 over the canonical JSON of everything that decides behaviour."""
        try:
            return hashlib.sha256(
                canonical_json_bytes(
                    self.semantic_payload(), f"{self.source}: operation spec"
                )
            ).hexdigest()
        except JsonSafetyError as exc:
            raise SpecSchemaError(
                f"{self.source}: this spec cannot be given a content identity: {exc}"
            ) from exc

    def verify_identity(self) -> str:
        """Re-derive the identity and refuse a spec that no longer matches it.

        Called before an episode executes. Loading validated a file; this
        validates the object that is about to run, so semantics replaced between
        the two — a swapped actor registry, a rewritten hidden state — cannot
        execute under the digest the artefact will record as their provenance.
        """
        recomputed = self.compute_digest()
        if recomputed != self.spec_digest_sha256:
            raise SpecIdentityError(
                f"{self.source}: this operation claims spec digest "
                f"{self.spec_digest_sha256!r} but what it now holds hashes to "
                f"{recomputed!r}; a run refuses rather than executing changed "
                "semantics under a stale identity"
            )
        return recomputed

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, source: str = "<memory>"
    ) -> OperationSpec:
        """Validate an authored mapping into a spec, refusing anything unknown."""
        _reject_unknown(raw, _TOP_LEVEL_FIELDS, source)
        missing = sorted(_TOP_LEVEL_FIELDS - set(raw))
        if missing:
            raise SpecSchemaError(f"{source}: missing required field(s) {missing}")

        version = _require_int(raw, "schema_version", source)
        if version != SUPPORTED_SCHEMA_VERSION:
            raise SpecSchemaError(
                f"{source}: schema_version {version} is not supported by this "
                f"build (expects {SUPPORTED_SCHEMA_VERSION})"
            )

        privacy = _require_text(raw, "privacy_status", source)
        if privacy != REQUIRED_PRIVACY_STATUS:
            raise SpecSchemaError(
                f"{source}: privacy_status must be {REQUIRED_PRIVACY_STATUS!r}; this "
                f"build executes synthetic fixtures only, and {privacy!r} claims "
                "otherwise"
            )

        actors = _parse_actors(raw["actors"], source)
        policy = _parse_policy(raw["policy"], source)
        fixtures = _parse_message_fixtures(raw["message_fixtures"], source)
        hidden = _freeze(_require_mapping(raw["hidden_state"], f"{source}: hidden_state"))
        semantic_scenario_id = _require_text(raw, "semantic_scenario_id", source)
        scenarios = _parse_scenarios(
            raw["scenarios"], actors, fixtures, semantic_scenario_id, source
        )

        # Built once with a placeholder identity, then given the identity it
        # computes for itself. The digest is over the *semantic projection*, so
        # the very same code path answers "what is this spec's identity?" when it
        # loads and "is it still that?" before every episode.
        spec = cls(
            operation_id=_require_text(raw, "operation_id", source),
            operation_type=_require_text(raw, "operation_type", source),
            operation_version=_require_text(raw, "operation_version", source),
            semantic_scenario_id=semantic_scenario_id,
            privacy_status=privacy,
            disclaimer=_require_text(raw, "disclaimer", source),
            data_provenance=_require_text(raw, "data_provenance", source),
            actors=MappingProxyType(actors),
            policy=policy,
            message_fixtures=MappingProxyType(fixtures),
            hidden_state=hidden,
            scenarios=MappingProxyType(scenarios),
            spec_digest_sha256="",
            source=source,
        )
        object.__setattr__(spec, "spec_digest_sha256", spec.compute_digest())
        return spec


def _plain(value: Any) -> Any:
    """A detached plain-builtin projection, so identity never depends on types."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    """The same structure with every mutable container replaced by a frozen one.

    Recursive on purpose. Freezing only the outer mapping leaves every nested
    authored payload writable, and a payload rewritten after loading changes
    what executes while ``spec_digest_sha256`` goes on describing what was read.
    ``MappingProxyType`` and ``tuple`` both raise :class:`TypeError` on a write,
    so the refusal arrives at the assignment rather than at the consequence.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _parse_actors(raw: Any, source: str) -> dict[str, ActorSpec]:
    mapping = _require_mapping(raw, f"{source}: actors")
    if not mapping:
        raise SpecSchemaError(f"{source}: actors must declare at least one actor")
    actors: dict[str, ActorSpec] = {}
    for actor_id, body in mapping.items():
        where = f"{source}: actor {actor_id}"
        entry = _require_mapping(body, where)
        _reject_unknown(entry, _ACTOR_FIELDS, where)
        authority = entry.get("authority")
        if not isinstance(authority, list) or not all(
            isinstance(item, str) for item in authority
        ):
            raise SpecSchemaError(f"{where}: authority must be a list of strings")
        actors[str(actor_id)] = ActorSpec(
            actor_id=str(actor_id),
            role=_require_text(entry, "role", where),
            authority=frozenset(_require_unique(authority, f"{where}: authority")),
        )
    return actors


def _parse_policy(raw: Any, source: str) -> PolicySpec:
    where = f"{source}: policy"
    entry = _require_mapping(raw, where)
    _reject_unknown(entry, _POLICY_FIELDS, where)
    missing = sorted(_POLICY_FIELDS - set(entry))
    if missing:
        raise SpecSchemaError(f"{where}: missing required policy field(s) {missing}")
    numeric = {
        key: _require_int(entry, key, where)
        for key in _POLICY_FIELDS
        if key not in {"currency", "issue_classification"}
    }
    for key, value in numeric.items():
        if value <= 0:
            raise SpecSchemaError(f"{where}: {key} must be positive, got {value}")
    return PolicySpec(
        currency=_require_text(entry, "currency", where),
        issue_classification=_require_text(entry, "issue_classification", where),
        **numeric,
    )


def _parse_message_fixtures(raw: Any, source: str) -> dict[str, str]:
    where = f"{source}: message_fixtures"
    entry = _require_mapping(raw, where)
    if not entry:
        raise SpecSchemaError(f"{where}: at least one message fixture is required")
    fixtures: dict[str, str] = {}
    for fixture_id, text in entry.items():
        if not isinstance(text, str) or not text:
            raise SpecSchemaError(f"{where}: {fixture_id!r} must map to non-empty text")
        fixtures[str(fixture_id)] = text
    return fixtures


def _parse_scenarios(
    raw: Any,
    actors: Mapping[str, ActorSpec],
    fixtures: Mapping[str, str],
    semantic_scenario_id: str,
    source: str,
) -> dict[str, ScenarioSpec]:
    mapping = _require_mapping(raw, f"{source}: scenarios")
    if not mapping:
        raise SpecSchemaError(f"{source}: at least one scenario variant is required")
    scenarios: dict[str, ScenarioSpec] = {}
    for scenario_id, body in mapping.items():
        where = f"{source}: scenario {scenario_id}"
        entry = _require_mapping(body, where)
        _reject_unknown(entry, _SCENARIO_FIELDS, where)
        missing = sorted(_SCENARIO_FIELDS - set(entry))
        if missing:
            raise SpecSchemaError(f"{where}: missing required field(s) {missing}")

        expected_terminal = _require_text(entry, "expected_terminal", where)
        if expected_terminal not in TERMINAL_OUTCOMES:
            raise UnknownTypeError(
                f"{where}: expected_terminal {expected_terminal!r} is not a terminal "
                f"class; known classes are {list(TERMINAL_OUTCOMES)}"
            )

        starts_at = _require_text(entry, "starts_at", where)
        parse_timestamp(starts_at, f"{where}: starts_at")

        budget = _require_int(entry, "human_checkpoint_budget", where)
        if budget < 0:
            raise SpecSchemaError(f"{where}: human_checkpoint_budget cannot be negative")

        required = entry.get("required_checkpoint_types")
        if not isinstance(required, list) or not all(
            isinstance(item, str) for item in required
        ):
            raise SpecSchemaError(
                f"{where}: required_checkpoint_types must be a list of strings"
            )

        failures = entry.get("dispatch_failures")
        if not isinstance(failures, list) or not all(
            isinstance(item, str) for item in failures
        ):
            raise SpecSchemaError(
                f"{where}: dispatch_failures must be a list of message fixture ids"
            )
        for fixture_id in failures:
            if fixture_id not in fixtures:
                raise UnknownTypeError(
                    f"{where}: dispatch_failures names {fixture_id!r}, which is not a "
                    f"declared message fixture ({sorted(fixtures)})"
                )

        events = _parse_events(entry["events"], actors, where)
        scenarios[str(scenario_id)] = ScenarioSpec(
            scenario_id=str(scenario_id),
            semantic_scenario_id=semantic_scenario_id,
            label=_require_text(entry, "label", where),
            starts_at=starts_at,
            expected_terminal=expected_terminal,
            human_checkpoint_budget=budget,
            required_checkpoint_types=tuple(required),
            dispatch_failures=frozenset(failures),
            expected_event_rejections=_parse_expected_rejections(
                entry["expected_event_rejections"], events, where
            ),
            events=events,
        )
    return scenarios


def _parse_expected_rejections(
    raw: Any, events: tuple[AuthoredEvent, ...], where: str
) -> tuple[ExpectedEventRejection, ...]:
    """Validate the scenario's declared non-mutating deliveries.

    Bound to the authored events by id, so an expectation for an event this
    scenario does not contain is refused rather than sitting in the file
    licensing nothing. One expectation per event: two would give the desync
    check two answers.
    """
    place = f"{where}: expected_event_rejections"
    if not isinstance(raw, list):
        raise SpecSchemaError(f"{place}: must be a list (use [] for none)")
    known = {event.event_id for event in events}
    expected: list[ExpectedEventRejection] = []
    seen: set[str] = set()
    for index, body in enumerate(raw):
        position = f"{place}[{index}]"
        item = _require_mapping(body, position)
        _reject_unknown(item, _EXPECTED_REJECTION_FIELDS, position)
        missing = sorted(_EXPECTED_REJECTION_FIELDS - set(item))
        if missing:
            raise SpecSchemaError(f"{position}: missing required field(s) {missing}")
        event_id = _require_text(item, "event_id", position)
        if event_id not in known:
            raise UnknownTypeError(
                f"{position}: names event {event_id!r}, which this scenario does not "
                f"author; known events are {sorted(known)}"
            )
        if event_id in seen:
            raise DuplicateIdentityError(
                f"{position}: event {event_id!r} already has a declared refusal; one "
                "expectation per event, or the desync check has two answers"
            )
        seen.add(event_id)
        expected.append(
            ExpectedEventRejection(
                event_id=event_id,
                code=_require_text(item, "code", position),
                reason=_require_text(item, "reason", position),
            )
        )
    return tuple(expected)


def _parse_events(
    raw: Any, actors: Mapping[str, ActorSpec], where: str
) -> tuple[AuthoredEvent, ...]:
    if not isinstance(raw, list) or not raw:
        raise SpecSchemaError(f"{where}: events must be a non-empty list")
    events: list[AuthoredEvent] = []
    seen: set[str] = set()
    for index, body in enumerate(raw):
        position = f"{where}: event #{index}"
        entry = _require_mapping(body, position)
        _reject_unknown(entry, _EVENT_FIELDS, position)
        event_id = _require_text(entry, "event_id", position)
        if event_id in seen:
            raise DuplicateIdentityError(
                f"{where}: event id {event_id!r} is declared twice; an event identity "
                "is used once so a duplicate cannot mutate twice"
            )
        seen.add(event_id)

        event_type = _require_text(entry, "type", position)
        if event_type not in MAINTENANCE_EVENT_TYPES:
            raise UnknownTypeError(
                f"{position}: unknown event type {event_type!r}; known types are "
                f"{sorted(MAINTENANCE_EVENT_TYPES)}"
            )
        actor_id = _require_text(entry, "actor", position)
        if actor_id not in actors:
            raise UnknownTypeError(
                f"{position}: unknown actor {actor_id!r}; known actors are "
                f"{sorted(actors)}"
            )
        required_authority = MAINTENANCE_EVENT_TYPES[event_type]
        if not actors[actor_id].holds(required_authority):
            raise AuthorityError(
                f"{position}: actor {actor_id!r} does not hold {required_authority!r}, "
                f"so it cannot emit {event_type!r}; provenance is derived from the "
                "actor registry and cannot be self-asserted"
            )

        at = entry.get("at")
        trigger = _parse_trigger(entry, position)
        if at is not None and trigger is not None:
            raise SpecSchemaError(
                f"{position}: declares both an absolute instant and a cause; an "
                "authored event has exactly one answer to 'when'"
            )
        if at is None and trigger is None:
            raise SpecSchemaError(
                f"{position}: declares neither 'at' nor 'on_action'/'on_event', so "
                "nothing can ever schedule it"
            )
        if at is not None:
            parse_timestamp(at, position)

        delay = entry.get("delay_minutes", 0)
        if isinstance(delay, bool) or not isinstance(delay, int) or delay < 0:
            raise SpecSchemaError(
                f"{position}: delay_minutes must be a non-negative integer, got {delay!r}"
            )

        triggers_agent = entry.get("triggers_agent", True)
        if not isinstance(triggers_agent, bool):
            raise SpecSchemaError(f"{position}: triggers_agent must be true or false")

        payload = _require_mapping(entry.get("payload", {}), f"{position}: payload")
        validate_event_payload(event_type, payload, position)
        events.append(
            AuthoredEvent(
                event_id=event_id,
                event_type=event_type,
                actor_id=actor_id,
                sequence=index,
                payload=_freeze(payload),
                triggers_agent=triggers_agent,
                at=at if isinstance(at, str) else None,
                trigger=trigger,
                delay_minutes=delay,
            )
        )
    return tuple(events)


def _parse_trigger(entry: Mapping[str, Any], position: str) -> ConditionalTrigger | None:
    on_action = entry.get("on_action")
    on_event = entry.get("on_event")
    if on_action is not None and on_event is not None:
        raise SpecSchemaError(
            f"{position}: declares both on_action and on_event; an authored event has "
            "one cause"
        )
    if on_action is None and on_event is None:
        return None
    kind = "action" if on_action is not None else "event"
    body = _require_mapping(
        on_action if on_action is not None else on_event, f"{position}: on_{kind}"
    )
    _reject_unknown(body, _TRIGGER_FIELDS, f"{position}: on_{kind}")
    type_name = _require_text(body, "type", f"{position}: on_{kind}")
    known = MAINTENANCE_ACTION_TYPES if kind == "action" else MAINTENANCE_EVENT_TYPES
    if type_name not in known:
        raise UnknownTypeError(
            f"{position}: on_{kind} names {type_name!r}, which is not a known "
            f"{kind} type; known types are {sorted(known)}"
        )
    cycle_id = body.get("cycle_id")
    if cycle_id is not None and not isinstance(cycle_id, str):
        raise SpecSchemaError(f"{position}: on_{kind} cycle_id must be a string")
    return ConditionalTrigger(kind=kind, type_name=type_name, cycle_id=cycle_id)


def load_spec(path: str | Path) -> OperationSpec:
    """Read and validate an operation spec from a YAML file.

    Duplicate mapping keys, non-UTF-8 bytes, cycles and over-deep nesting are all
    refused by the shared strict reader; they arrive here as one named
    :class:`~operatebench.core.errors.SpecFormatError` rather than as three
    different tracebacks.
    """
    path = Path(path)
    try:
        raw = read_yaml_mapping(path, what="operation spec")
    except BoundarySchemaError as exc:
        raise SpecFormatError(str(exc)) from exc
    except OSError as exc:
        raise SpecFormatError(f"cannot read operation spec {path}: {exc}") from exc
    return OperationSpec.from_mapping(raw, source=str(path))


__all__ = [
    "ENVIRONMENT_ALLOCATED_IDENTITY_FIELDS",
    "MAINTENANCE_ACTION_PAYLOAD_SCHEMAS",
    "MAINTENANCE_ACTION_TYPES",
    "MAINTENANCE_EVENT_TYPES",
    "MAINTENANCE_WAKE_EVENT_TYPES",
    "POLICY_FIELDS",
    "REQUIRED_PRIVACY_STATUS",
    "SUPPORTED_SCHEMA_VERSION",
    "TERMINAL_OUTCOMES",
    "ActorSpec",
    "AuthoredEvent",
    "ConditionalTrigger",
    "ExpectedEventRejection",
    "OperationSpec",
    "PolicySpec",
    "ScenarioSpec",
    "action_schema_view",
    "load_spec",
    "validate_action_payload",
    "validate_event_payload",
]
