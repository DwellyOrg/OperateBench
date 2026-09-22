"""Strict synthetic specification for the commerce return/refund pilot."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
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
    action_evidence_contract_problem,
)
from operatebench.domains.commerce.return_refund.retrieval import (
    RETURN_REFUND_RETRIEVAL_TOOLS,
    return_refund_retrieval_catalogue,
)
from operatebench.domains.commerce.return_refund.state import PRODUCIBLE_TERMINALS

PACK_OPERATION_TYPE = "commerce.return_refund.synthetic.v1"
REQUIRED_PRIVACY_STATUS = "SYNTHETIC_ONLY"
SUPPORTED_SCHEMA_VERSION = 1

RETURN_REFUND_EVENT_TYPES: Mapping[str, str] = {
    "customer_return_requested": "request_return",
    "customer_handover_claimed": "claim_handover",
    "carrier_handover_verified": "verify_handover",
    "warehouse_inspection_completed": "inspect_return",
    "refund_approval_received": "approve_refund",
    "refund_settlement_confirmed": "settle_refund",
    "refund_settlement_reversed": "reverse_refund",
    "handover_reminder_due": "emit_timer",
    "handover_deadline_due": "emit_timer",
    "approval_deadline_due": "emit_timer",
    "resolution_cooldown_expired": "emit_timer",
}
RETURN_REFUND_EVENT_TYPE_NAMES: tuple[str, ...] = tuple(sorted(RETURN_REFUND_EVENT_TYPES))
RETURN_REFUND_WAKE_EVENT_TYPES = RETURN_REFUND_EVENT_TYPE_NAMES
RETURN_REFUND_ACTION_TYPES: tuple[str, ...] = (
    "issue_return_authorisation",
    "request_refund_approval",
    "request_refund",
    "send_message",
    "complete",
)
RETURN_REFUND_CHECKPOINT_TYPES: tuple[str, ...] = ("high_value_refund_approval",)

RETURN_REFUND_ROLE_AUTHORITY: Mapping[str, str] = {
    "agent": "actor_claim",
    "customer": "actor_claim",
    "returns_approver": "actor_claim",
    "authoritative_external_system": "authoritative_verification",
    "system_scheduler": "system_event",
}

_ROLE_ALLOWED_AUTHORITIES: Mapping[str, frozenset[str]] = {
    "agent": frozenset(RETURN_REFUND_ACTION_TYPES),
    "customer": frozenset({"request_return", "claim_handover"}),
    "returns_approver": frozenset({"approve_refund"}),
    "authoritative_external_system": frozenset(
        {"verify_handover", "inspect_return", "settle_refund", "reverse_refund"}
    ),
    "system_scheduler": frozenset({"emit_timer"}),
}
_REQUIRED_ACTOR_ROLES: Mapping[str, str] = {
    "agent": "agent",
    "customer_1": "customer",
    "returns_approver_1": "returns_approver",
    "carrier_system": "authoritative_external_system",
    "warehouse_system": "authoritative_external_system",
    "payment_system": "authoritative_external_system",
    "scheduler": "system_scheduler",
}
_REQUIRED_ACTOR_AUTHORITIES: Mapping[str, frozenset[str]] = {
    "agent": frozenset(RETURN_REFUND_ACTION_TYPES),
    "customer_1": frozenset({"request_return", "claim_handover"}),
    "returns_approver_1": frozenset({"approve_refund"}),
    "carrier_system": frozenset({"verify_handover"}),
    "warehouse_system": frozenset({"inspect_return"}),
    "payment_system": frozenset({"settle_refund", "reverse_refund"}),
    "scheduler": frozenset({"emit_timer"}),
}
_REQUIRED_MESSAGE_FIXTURES = frozenset(
    {
        "msg_handover_reminder",
        "msg_return_expired",
        "msg_refund_complete",
        "msg_refund_denied",
    }
)

_EVENT_PAYLOAD_SCHEMAS: Mapping[str, tuple[Mapping[str, str], Mapping[str, str]]] = {
    "customer_return_requested": (
        {
            "return_id": "string",
            "order_id": "string",
            "cycle_id": "string",
            "amount_minor": "positive_integer",
            "currency": "string",
        },
        {},
    ),
    "customer_handover_claimed": (
        {
            "claim_id": "string",
            "return_id": "string",
            "authorisation_id": "string",
            "cycle_id": "string",
        },
        {},
    ),
    "carrier_handover_verified": (
        {
            "carrier_record_id": "string",
            "return_id": "string",
            "authorisation_id": "string",
            "cycle_id": "string",
            "handed_over": "boolean",
        },
        {},
    ),
    "warehouse_inspection_completed": (
        {
            "inspection_id": "string",
            "return_id": "string",
            "cycle_id": "string",
            "disposition": "string",
            "amount_minor": "positive_integer",
            "currency": "string",
        },
        {},
    ),
    "refund_approval_received": (
        {
            "checkpoint_id": "string",
            "inspection_id": "string",
            "cycle_id": "string",
            "amount_minor": "positive_integer",
            "currency": "string",
            "decision": "string",
        },
        {},
    ),
    "refund_settlement_confirmed": (
        {
            "settlement_id": "string",
            "refund_request_id": "string",
            "inspection_id": "string",
            "cycle_id": "string",
            "amount_minor": "positive_integer",
            "currency": "string",
        },
        {},
    ),
    "refund_settlement_reversed": (
        {
            "reversal_id": "string",
            "refund_request_id": "string",
            "cycle_id": "string",
            "reason_code": "string",
        },
        {},
    ),
    "handover_reminder_due": (
        {"authorisation_id": "string", "cycle_id": "string"},
        {},
    ),
    "handover_deadline_due": (
        {"authorisation_id": "string", "cycle_id": "string"},
        {},
    ),
    "approval_deadline_due": (
        {"checkpoint_id": "string", "cycle_id": "string"},
        {},
    ),
    "resolution_cooldown_expired": (
        {"refund_request_id": "string", "cycle_id": "string"},
        {},
    ),
}

RETURN_REFUND_ACTION_PAYLOAD_SCHEMAS: Mapping[
    str, tuple[Mapping[str, str], Mapping[str, str]]
] = {
    "issue_return_authorisation": (
        {"return_id": "string", "cycle_id": "string"},
        {},
    ),
    "request_refund_approval": (
        {"inspection_id": "string", "cycle_id": "string"},
        {},
    ),
    "request_refund": (
        {"inspection_id": "string", "cycle_id": "string"},
        {"approval_checkpoint_id": "optional_string"},
    ),
    "send_message": (
        {"recipient_actor_id": "string", "message_fixture_id": "string"},
        {"cycle_id": "optional_string"},
    ),
}

RETURN_REFUND_ACTION_EVIDENCE_CONTRACT = ActionEvidenceContract(
    {
        "issue_return_authorisation": {
            "reads": {"get_return_case": READ_REQUIRED},
        },
        "request_refund_approval": {
            "reads": {
                "get_return_case": READ_REQUIRED,
                "list_inspections": READ_REQUIRED,
                "list_checkpoints": READ_REQUIRED,
            },
            "evidence_refs": (
                PayloadRegistryKey("list_inspections", "inspections", "inspection_id"),
            ),
        },
        "request_refund": {
            "reads": {
                "get_return_case": READ_REQUIRED,
                "list_carrier_records": READ_REQUIRED,
                "list_inspections": READ_REQUIRED,
                "list_checkpoints": READ_REQUIRED,
                "list_refunds": READ_REQUIRED,
            },
            "evidence_refs": (
                PayloadRegistryKey("list_inspections", "inspections", "inspection_id"),
            ),
        },
        "send_message": {
            "reads": {
                "get_return_case": READ_REQUIRED,
                "list_communications": READ_REQUIRED,
                "list_obligations": READ_REQUIRED,
            },
        },
        COMPLETE_OUTCOME_KEY: {
            "reads": {
                "get_return_case": READ_REQUIRED,
                "list_checkpoints": READ_REQUIRED,
                "list_refunds": READ_REQUIRED,
                "list_communications": READ_REQUIRED,
                "list_obligations": READ_REQUIRED,
            },
        },
    }
)
RETURN_REFUND_READ_CONTRACT = RETURN_REFUND_ACTION_EVIDENCE_CONTRACT


def return_refund_action_evidence_contract() -> ActionEvidenceContract:
    return RETURN_REFUND_ACTION_EVIDENCE_CONTRACT


def action_schema_view() -> dict[str, dict[str, Any]]:
    contract = return_refund_action_evidence_contract()
    result = {
        action_type: {
            "required": dict(required),
            "optional": dict(optional),
            "reads": contract.schema_part(action_type),
            "evidence_refs": contract.evidence_schema_part(action_type),
        }
        for action_type, (required, optional) in (
            RETURN_REFUND_ACTION_PAYLOAD_SCHEMAS.items()
        )
    }
    result[COMPLETE_OUTCOME_KEY] = {
        "required": {},
        "optional": {},
        "reads": contract.schema_part(COMPLETE_OUTCOME_KEY),
        "evidence_refs": contract.evidence_schema_part(COMPLETE_OUTCOME_KEY),
    }
    return result


_EVIDENCE_CONTRACT_PROBLEM = action_evidence_contract_problem(
    RETURN_REFUND_ACTION_EVIDENCE_CONTRACT,
    catalogue=return_refund_retrieval_catalogue(),
    retrieval_tools=RETURN_REFUND_RETRIEVAL_TOOLS,
    payload_schemas=RETURN_REFUND_ACTION_PAYLOAD_SCHEMAS,
    outcome_keys=(*RETURN_REFUND_ACTION_PAYLOAD_SCHEMAS, COMPLETE_OUTCOME_KEY),
)
if _EVIDENCE_CONTRACT_PROBLEM is not None:  # pragma: no cover - import guard
    raise SpecSchemaError(
        "the return/refund evidence contract cannot be executed: "
        f"{_EVIDENCE_CONTRACT_PROBLEM}"
    )


def _validate_payload(
    schema: tuple[Mapping[str, str], Mapping[str, str]],
    payload: Mapping[str, Any],
    where: str,
    what: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise SpecSchemaError(f"{where}: payload must be a mapping")
    required, optional = schema
    allowed = frozenset(required) | frozenset(optional)
    _reject_unknown(payload, allowed, f"{where}: payload")
    missing = sorted(set(required) - set(payload))
    if missing:
        raise SpecSchemaError(
            f"{where}: payload for {what!r} is missing required field(s) {missing}"
        )
    for name, kind in (*required.items(), *optional.items()):
        if name not in payload:
            continue
        value = payload[name]
        if kind == "optional_string" and value is None:
            continue
        if kind in {"string", "optional_string"}:
            valid = type(value) is str and bool(value.strip())
        elif kind == "integer":
            valid = type(value) is int
        elif kind == "positive_integer":
            valid = type(value) is int and value > 0
        elif kind == "boolean":
            valid = type(value) is bool
        else:  # pragma: no cover - declaration is closed in this module
            valid = False
        if not valid:
            raise SpecSchemaError(
                f"{where}: payload field {name!r} for {what!r} must be {kind}"
            )


def validate_event_payload(
    event_type: str, payload: Mapping[str, Any], where: str
) -> None:
    schema = _EVENT_PAYLOAD_SCHEMAS.get(event_type)
    if schema is None:
        raise UnknownTypeError(f"{where}: unknown event type {event_type!r}")
    _validate_payload(schema, payload, where, event_type)


def validate_action_payload(
    action_type: str, payload: Mapping[str, Any], where: str
) -> None:
    schema = RETURN_REFUND_ACTION_PAYLOAD_SCHEMAS.get(action_type)
    if schema is None:
        raise UnknownTypeError(f"{where}: unknown action type {action_type!r}")
    _validate_payload(schema, payload, where, action_type)


@dataclass(frozen=True)
class ActorSpec:
    actor_id: str
    role: str
    authority: frozenset[str]

    def holds(self, authority: str) -> bool:
        return authority in self.authority


@dataclass(frozen=True)
class PolicySpec:
    currency: str
    refund_amount_minor: int
    approval_threshold_minor: int
    handover_reminder_after_minutes: int
    handover_deadline_after_minutes: int
    approval_deadline_after_minutes: int
    provisional_close_minutes: int
    completion_notice_within_minutes: int
    max_turns_per_invocation: int
    horizon_minutes: int
    max_retrieval_batches_per_invocation: int


@dataclass(frozen=True)
class ConditionalTrigger:
    kind: str
    type_name: str
    cycle_id: str | None = None


@dataclass(frozen=True)
class AuthoredEvent:
    event_id: str
    event_type: str
    actor_id: str
    sequence: int
    payload: Mapping[str, Any]
    triggers_agent: bool = True
    at: str | None = None
    trigger: ConditionalTrigger | None = None
    delay_minutes: int = 0

    @property
    def cycle_id(self) -> str | None:
        value = self.payload.get("cycle_id")
        return value if isinstance(value, str) else None

    def payload_snapshot(self) -> dict[str, Any]:
        return _plain_mapping(self.payload)

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "type": self.event_type,
            "actor": self.actor_id,
            "sequence": self.sequence,
            "payload": self.payload_snapshot(),
            "triggers_agent": self.triggers_agent,
            "at": self.at,
            "trigger": (
                {
                    "kind": self.trigger.kind,
                    "type": self.trigger.type_name,
                    "cycle_id": self.trigger.cycle_id,
                }
                if self.trigger is not None
                else None
            ),
            "delay_minutes": self.delay_minutes,
        }


@dataclass(frozen=True)
class ExpectedEventRejection:
    event_id: str
    code: str
    reason: str

    def semantic_payload(self) -> dict[str, str]:
        return {"event_id": self.event_id, "code": self.code, "reason": self.reason}


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    semantic_scenario_id: str
    label: str
    starts_at: str
    expected_terminal: str
    human_checkpoint_budget: int
    required_checkpoint_types: tuple[str, ...]
    approval_threshold_minor: int
    dispatch_failures: frozenset[str]
    expected_event_rejections: tuple[ExpectedEventRejection, ...]
    events: tuple[AuthoredEvent, ...]

    def expected_rejection_codes(self) -> dict[str, str]:
        return {item.event_id: item.code for item in self.expected_event_rejections}

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "semantic_scenario_id": self.semantic_scenario_id,
            "label": self.label,
            "starts_at": self.starts_at,
            "expected_terminal": self.expected_terminal,
            "human_checkpoint_budget": self.human_checkpoint_budget,
            "required_checkpoint_types": list(self.required_checkpoint_types),
            "dispatch_failures": sorted(self.dispatch_failures),
            "expected_event_rejections": [
                item.semantic_payload() for item in self.expected_event_rejections
            ],
            "events": [event.semantic_payload() for event in self.events],
        }


@dataclass(frozen=True)
class ReturnRefundSpec:
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
    source: str

    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.scenarios))

    def scenario(self, scenario_id: str) -> ScenarioSpec:
        if scenario_id not in self.scenarios:
            raise KeyError(scenario_id)
        return self.scenarios[scenario_id]

    def hidden_state_snapshot(self) -> dict[str, Any]:
        return _plain_mapping(self.hidden_state)

    def message_fixture_snapshot(self) -> dict[str, str]:
        return dict(self.message_fixtures)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "operation_version": self.operation_version,
            "semantic_scenario_id": self.semantic_scenario_id,
            "privacy_status": self.privacy_status,
            "spec_digest_sha256": self.spec_digest_sha256,
            "scenario_ids": list(self.scenario_ids()),
        }

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SUPPORTED_SCHEMA_VERSION,
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "operation_version": self.operation_version,
            "semantic_scenario_id": self.semantic_scenario_id,
            "privacy_status": self.privacy_status,
            "disclaimer": self.disclaimer,
            "data_provenance": self.data_provenance,
            "actors": {
                actor_id: {
                    "role": actor.role,
                    "authority": sorted(actor.authority),
                }
                for actor_id, actor in sorted(self.actors.items())
            },
            "policy": {
                name: getattr(self.policy, name)
                for name in self.policy.__dataclass_fields__
            },
            "message_fixtures": dict(sorted(self.message_fixtures.items())),
            "hidden_state": _plain(self.hidden_state),
            "scenarios": {
                scenario_id: self.scenarios[scenario_id].semantic_payload()
                for scenario_id in self.scenario_ids()
            },
        }

    def compute_digest(self) -> str:
        try:
            body = canonical_json_bytes(
                self.semantic_payload(), "commerce return/refund operation spec"
            )
        except JsonSafetyError as exc:
            raise SpecIdentityError(
                f"{self.source}: cannot compute operation spec identity: {exc}"
            ) from exc
        return hashlib.sha256(body).hexdigest()

    def verify_identity(self) -> str:
        digest = self.compute_digest()
        if digest != self.spec_digest_sha256:
            raise SpecIdentityError(
                f"{self.source}: spec identity changed after validation"
            )
        return digest

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, source: str = "<memory>"
    ) -> ReturnRefundSpec:
        _reject_unknown(raw, _TOP_LEVEL_FIELDS, source)
        missing = sorted(_TOP_LEVEL_FIELDS - set(raw))
        if missing:
            raise SpecSchemaError(f"{source}: missing required field(s) {missing}")
        version = _require_int(raw, "schema_version", source)
        if version != SUPPORTED_SCHEMA_VERSION:
            raise SpecSchemaError(
                f"{source}: unsupported schema_version {version}; expected 1"
            )
        privacy = _require_text(raw, "privacy_status", source)
        if privacy != REQUIRED_PRIVACY_STATUS:
            raise SpecSchemaError(
                f"{source}: privacy_status must be {REQUIRED_PRIVACY_STATUS!r}"
            )
        # Loading proves that this is a well-formed return/refund specification.
        # The SDK pack adapter separately binds it to PACK_OPERATION_TYPE; keeping
        # those two checks separate lets a caller report a pack mismatch by name.
        operation_type = _require_text(raw, "operation_type", source)
        actors = _parse_actors(raw["actors"], source)
        fixtures = _parse_message_fixtures(raw["message_fixtures"], source)
        policy = _parse_policy(raw["policy"], source)
        semantic_id = _require_text(raw, "semantic_scenario_id", source)
        spec = cls(
            operation_id=_require_text(raw, "operation_id", source),
            operation_type=operation_type,
            operation_version=_require_text(raw, "operation_version", source),
            semantic_scenario_id=semantic_id,
            privacy_status=privacy,
            disclaimer=_require_text(raw, "disclaimer", source),
            data_provenance=_require_text(raw, "data_provenance", source),
            actors=MappingProxyType(actors),
            policy=policy,
            message_fixtures=MappingProxyType(fixtures),
            hidden_state=_freeze(
                _require_mapping(raw["hidden_state"], f"{source}: hidden_state")
            ),
            scenarios=MappingProxyType(
                _parse_scenarios(
                    raw["scenarios"],
                    actors,
                    fixtures,
                    semantic_id,
                    policy,
                    source,
                )
            ),
            spec_digest_sha256="",
            source=source,
        )
        object.__setattr__(spec, "spec_digest_sha256", spec.compute_digest())
        return spec


# Every authored level is closed.  An unknown field is never silently ignored.
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
_POLICY_FIELDS = frozenset(PolicySpec.__dataclass_fields__)
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
_EXPECTED_REJECTION_FIELDS = frozenset({"event_id", "code", "reason"})


def _reject_unknown(
    mapping: Mapping[str, Any], allowed: frozenset[str], where: str
) -> None:
    non_text = sorted(repr(name) for name in mapping if type(name) is not str)
    if non_text:
        raise UnknownFieldError(f"{where}: field names must be strings, got {non_text}")
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise UnknownFieldError(
            f"{where}: unknown field(s) {unknown}; known fields are {sorted(allowed)}"
        )


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpecSchemaError(f"{where}: expected mapping, got {type(value).__name__}")
    return value


def _require_text(mapping: Mapping[str, Any], key: str, where: str) -> str:
    value = mapping.get(key)
    if type(value) is not str or not value.strip():
        raise SpecSchemaError(f"{where}: {key!r} must be a non-empty string")
    return value


def _require_int(mapping: Mapping[str, Any], key: str, where: str) -> int:
    value = mapping.get(key)
    if type(value) is not int:
        raise SpecSchemaError(f"{where}: {key!r} must be an integer")
    return value


def _unique_strings(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        type(item) is str and item.strip() for item in value
    ):
        raise SpecSchemaError(f"{where}: must be a list of non-empty strings")
    if len(set(value)) != len(value):
        raise DuplicateIdentityError(f"{where}: values must be unique")
    return tuple(value)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    return value


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _plain(item) for key, item in value.items()}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _parse_actors(raw: Any, source: str) -> dict[str, ActorSpec]:
    mapping = _require_mapping(raw, f"{source}: actors")
    actors: dict[str, ActorSpec] = {}
    for raw_actor_id, body in mapping.items():
        if type(raw_actor_id) is not str or not raw_actor_id.strip():
            raise SpecSchemaError(f"{source}: actor ids must be non-empty strings")
        where = f"{source}: actor {raw_actor_id}"
        entry = _require_mapping(body, where)
        _reject_unknown(entry, _ACTOR_FIELDS, where)
        role = _require_text(entry, "role", where)
        allowed_authorities = _ROLE_ALLOWED_AUTHORITIES.get(role)
        if allowed_authorities is None:
            raise UnknownTypeError(
                f"{where}: unknown role {role!r}; expected one of "
                f"{sorted(_ROLE_ALLOWED_AUTHORITIES)}"
            )
        authorities = frozenset(
            _unique_strings(entry.get("authority"), f"{where}: authority")
        )
        excess = sorted(authorities - allowed_authorities)
        if excess:
            raise AuthorityError(f"{where}: role {role!r} cannot hold authority {excess}")
        actors[raw_actor_id] = ActorSpec(
            actor_id=raw_actor_id,
            role=role,
            authority=authorities,
        )
    for actor_id, expected_role in _REQUIRED_ACTOR_ROLES.items():
        actual = actors.get(actor_id)
        if actual is None or actual.role != expected_role:
            raise SpecSchemaError(
                f"{source}: actor {actor_id!r} must declare role {expected_role!r}"
            )
        missing = sorted(_REQUIRED_ACTOR_AUTHORITIES[actor_id] - actual.authority)
        if missing:
            raise AuthorityError(
                f"{source}: actor {actor_id!r} is missing authority {missing}"
            )
    return actors


def _parse_policy(raw: Any, source: str) -> PolicySpec:
    where = f"{source}: policy"
    entry = _require_mapping(raw, where)
    _reject_unknown(entry, _POLICY_FIELDS, where)
    missing = sorted(_POLICY_FIELDS - set(entry))
    if missing:
        raise SpecSchemaError(f"{where}: missing field(s) {missing}")
    currency = _require_text(entry, "currency", where)
    numeric = {
        key: _require_int(entry, key, where)
        for key in _POLICY_FIELDS
        if key != "currency"
    }
    if any(value <= 0 for value in numeric.values()):
        raise SpecSchemaError(f"{where}: every numeric policy value must be positive")
    if numeric["refund_amount_minor"] <= numeric["approval_threshold_minor"]:
        raise SpecSchemaError(
            f"{where}: refund_amount_minor must exceed approval_threshold_minor"
        )
    if (
        numeric["handover_reminder_after_minutes"]
        >= numeric["handover_deadline_after_minutes"]
    ):
        raise SpecSchemaError(f"{where}: handover reminder must precede deadline")
    if numeric["max_turns_per_invocation"] < 2:
        raise SpecSchemaError(f"{where}: max_turns_per_invocation must be at least 2")
    if numeric["max_retrieval_batches_per_invocation"] < 4:
        raise SpecSchemaError(
            f"{where}: max_retrieval_batches_per_invocation must be at least 4"
        )
    bounded_intervals = (
        "handover_deadline_after_minutes",
        "approval_deadline_after_minutes",
        "provisional_close_minutes",
        "completion_notice_within_minutes",
    )
    too_long = [
        name for name in bounded_intervals if numeric[name] > numeric["horizon_minutes"]
    ]
    if too_long:
        raise SpecSchemaError(
            f"{where}: intervals cannot exceed horizon_minutes: {too_long}"
        )
    return PolicySpec(currency=currency, **numeric)


def _parse_message_fixtures(raw: Any, source: str) -> dict[str, str]:
    mapping = _require_mapping(raw, f"{source}: message_fixtures")
    result: dict[str, str] = {}
    for fixture_id, text in mapping.items():
        if (
            type(fixture_id) is not str
            or not fixture_id.strip()
            or type(text) is not str
            or not text.strip()
        ):
            raise SpecSchemaError(
                f"{source}: message fixtures require non-empty string ids and text"
            )
        result[fixture_id] = text
    missing = sorted(_REQUIRED_MESSAGE_FIXTURES - set(result))
    if missing:
        raise SpecSchemaError(
            f"{source}: message_fixtures is missing required fixture(s) {missing}"
        )
    return result


def _parse_scenarios(
    raw: Any,
    actors: Mapping[str, ActorSpec],
    fixtures: Mapping[str, str],
    semantic_scenario_id: str,
    policy: PolicySpec,
    source: str,
) -> dict[str, ScenarioSpec]:
    mapping = _require_mapping(raw, f"{source}: scenarios")
    scenarios: dict[str, ScenarioSpec] = {}
    for raw_id, body in mapping.items():
        if type(raw_id) is not str or not raw_id.strip():
            raise SpecSchemaError(f"{source}: scenario ids must be non-empty strings")
        where = f"{source}: scenario {raw_id}"
        entry = _require_mapping(body, where)
        _reject_unknown(entry, _SCENARIO_FIELDS, where)
        missing = sorted(_SCENARIO_FIELDS - set(entry))
        if missing:
            raise SpecSchemaError(f"{where}: missing field(s) {missing}")
        terminal = _require_text(entry, "expected_terminal", where)
        if terminal not in PRODUCIBLE_TERMINALS:
            raise UnknownTypeError(f"{where}: unsupported expected terminal {terminal!r}")
        starts_at = _require_text(entry, "starts_at", where)
        starts_instant = parse_timestamp(starts_at, f"{where}: starts_at")
        budget = _require_int(entry, "human_checkpoint_budget", where)
        if budget < 0:
            raise SpecSchemaError(f"{where}: human_checkpoint_budget cannot be negative")
        required = _unique_strings(
            entry["required_checkpoint_types"],
            f"{where}: required_checkpoint_types",
        )
        unknown_checkpoint_types = sorted(
            set(required) - set(RETURN_REFUND_CHECKPOINT_TYPES)
        )
        if unknown_checkpoint_types:
            raise UnknownTypeError(
                f"{where}: unknown required checkpoint type(s) "
                f"{unknown_checkpoint_types}; expected "
                f"{list(RETURN_REFUND_CHECKPOINT_TYPES)}"
            )
        failures = _unique_strings(
            entry["dispatch_failures"], f"{where}: dispatch_failures"
        )
        unknown_failures = sorted(set(failures) - set(fixtures))
        if unknown_failures:
            raise UnknownTypeError(
                f"{where}: unknown dispatch failure fixture(s) {unknown_failures}"
            )
        events = _parse_events(entry["events"], actors, where)
        for event in events:
            if (
                "amount_minor" in event.payload
                and event.payload["amount_minor"] != policy.refund_amount_minor
            ):
                raise SpecSchemaError(
                    f"{where}: event {event.event_id!r} amount_minor must match "
                    "policy.refund_amount_minor"
                )
            if (
                "currency" in event.payload
                and event.payload["currency"] != policy.currency
            ):
                raise SpecSchemaError(
                    f"{where}: event {event.event_id!r} currency must match "
                    "policy.currency"
                )
            if event.trigger is not None and event.delay_minutes > policy.horizon_minutes:
                raise SpecSchemaError(
                    f"{where}: event {event.event_id!r} delay exceeds horizon_minutes"
                )
            if event.at is None:
                continue
            event_instant = parse_timestamp(event.at, f"{where}: event {event.event_id}")
            delta_seconds = event_instant - starts_instant
            if delta_seconds < 0:
                raise SpecSchemaError(
                    f"{where}: event {event.event_id!r} precedes starts_at"
                )
            if delta_seconds > policy.horizon_minutes * 60:
                raise SpecSchemaError(
                    f"{where}: event {event.event_id!r} is beyond horizon_minutes"
                )
        scenarios[raw_id] = ScenarioSpec(
            scenario_id=raw_id,
            semantic_scenario_id=semantic_scenario_id,
            label=_require_text(entry, "label", where),
            starts_at=starts_at,
            expected_terminal=terminal,
            human_checkpoint_budget=budget,
            required_checkpoint_types=required,
            approval_threshold_minor=policy.approval_threshold_minor,
            dispatch_failures=frozenset(failures),
            expected_event_rejections=_parse_expected_rejections(
                entry["expected_event_rejections"], events, where
            ),
            events=events,
        )
    if not scenarios:
        raise SpecSchemaError(f"{source}: at least one scenario is required")
    return scenarios


def _parse_events(
    raw: Any, actors: Mapping[str, ActorSpec], where: str
) -> tuple[AuthoredEvent, ...]:
    if not isinstance(raw, list) or not raw:
        raise SpecSchemaError(f"{where}: events must be a non-empty list")
    result: list[AuthoredEvent] = []
    seen: set[str] = set()
    for sequence, body in enumerate(raw):
        position = f"{where}: event #{sequence}"
        entry = _require_mapping(body, position)
        _reject_unknown(entry, _EVENT_FIELDS, position)
        event_id = _require_text(entry, "event_id", position)
        if event_id in seen:
            raise DuplicateIdentityError(f"{position}: duplicate event id {event_id!r}")
        seen.add(event_id)
        event_type = _require_text(entry, "type", position)
        if event_type not in RETURN_REFUND_EVENT_TYPES:
            raise UnknownTypeError(f"{position}: unknown event type {event_type!r}")
        actor_id = _require_text(entry, "actor", position)
        actor = actors.get(actor_id)
        if actor is None:
            raise UnknownTypeError(f"{position}: unknown actor {actor_id!r}")
        required_authority = RETURN_REFUND_EVENT_TYPES[event_type]
        if not actor.holds(required_authority):
            raise AuthorityError(
                f"{position}: actor {actor_id!r} lacks {required_authority!r}"
            )
        at = entry.get("at")
        trigger = _parse_trigger(entry, position)
        if (at is None) == (trigger is None):
            raise SpecSchemaError(
                f"{position}: declare exactly one of at or on_action/on_event"
            )
        if at is not None:
            if type(at) is not str:
                raise SpecSchemaError(f"{position}: at must be a timestamp string")
            parse_timestamp(at, f"{position}: at")
            if "delay_minutes" in entry:
                raise SpecSchemaError(
                    f"{position}: delay_minutes is only valid for conditional events"
                )
        delay = entry.get("delay_minutes", 0)
        if type(delay) is not int or delay < 0:
            raise SpecSchemaError(
                f"{position}: delay_minutes must be a non-negative integer"
            )
        triggers_agent = entry.get("triggers_agent", True)
        if type(triggers_agent) is not bool:
            raise SpecSchemaError(f"{position}: triggers_agent must be boolean")
        payload = _require_mapping(entry.get("payload", {}), f"{position}: payload")
        validate_event_payload(event_type, payload, position)
        result.append(
            AuthoredEvent(
                event_id=event_id,
                event_type=event_type,
                actor_id=actor_id,
                sequence=sequence,
                payload=_freeze(payload),
                triggers_agent=triggers_agent,
                at=at,
                trigger=trigger,
                delay_minutes=delay,
            )
        )
    return tuple(result)


def _parse_trigger(entry: Mapping[str, Any], position: str) -> ConditionalTrigger | None:
    action = entry.get("on_action")
    event = entry.get("on_event")
    if action is not None and event is not None:
        raise SpecSchemaError(f"{position}: an event cannot have two causes")
    if action is None and event is None:
        return None
    kind = "action" if action is not None else "event"
    body = _require_mapping(action if action is not None else event, f"{position}: cause")
    _reject_unknown(body, _TRIGGER_FIELDS, f"{position}: cause")
    type_name = _require_text(body, "type", f"{position}: cause")
    known = (
        RETURN_REFUND_ACTION_TYPES if kind == "action" else RETURN_REFUND_EVENT_TYPE_NAMES
    )
    if type_name not in known:
        raise UnknownTypeError(f"{position}: cause names unknown {kind} {type_name!r}")
    cycle_id = body.get("cycle_id")
    if cycle_id is not None and (type(cycle_id) is not str or not cycle_id.strip()):
        raise SpecSchemaError(f"{position}: cause cycle_id must be non-empty string")
    return ConditionalTrigger(kind, type_name, cycle_id)


def _parse_expected_rejections(
    raw: Any, events: tuple[AuthoredEvent, ...], where: str
) -> tuple[ExpectedEventRejection, ...]:
    if not isinstance(raw, list):
        raise SpecSchemaError(f"{where}: expected_event_rejections must be a list")
    known = {event.event_id for event in events}
    seen: set[str] = set()
    result: list[ExpectedEventRejection] = []
    for index, body in enumerate(raw):
        position = f"{where}: expected_event_rejections[{index}]"
        entry = _require_mapping(body, position)
        _reject_unknown(entry, _EXPECTED_REJECTION_FIELDS, position)
        event_id = _require_text(entry, "event_id", position)
        if event_id not in known:
            raise UnknownTypeError(f"{position}: unknown event id {event_id!r}")
        if event_id in seen:
            raise DuplicateIdentityError(f"{position}: duplicate expectation")
        seen.add(event_id)
        result.append(
            ExpectedEventRejection(
                event_id,
                _require_text(entry, "code", position),
                _require_text(entry, "reason", position),
            )
        )
    return tuple(result)


def load_spec(path: str | Path) -> ReturnRefundSpec:
    path = Path(path)
    try:
        raw = read_yaml_mapping(path, what="commerce return/refund operation spec")
    except BoundarySchemaError as exc:
        raise SpecFormatError(str(exc)) from exc
    except OSError as exc:
        raise SpecFormatError(f"cannot read operation spec {path}: {exc}") from exc
    return ReturnRefundSpec.from_mapping(raw, source=str(path))


# Compatibility with code that names the generic domain-owned spec type.
OperationSpec = ReturnRefundSpec


__all__ = [
    "ENVIRONMENT_ALLOCATED_IDENTITY_FIELDS",
    "PACK_OPERATION_TYPE",
    "REQUIRED_PRIVACY_STATUS",
    "RETURN_REFUND_ACTION_EVIDENCE_CONTRACT",
    "RETURN_REFUND_ACTION_PAYLOAD_SCHEMAS",
    "RETURN_REFUND_ACTION_TYPES",
    "RETURN_REFUND_CHECKPOINT_TYPES",
    "RETURN_REFUND_EVENT_TYPES",
    "RETURN_REFUND_EVENT_TYPE_NAMES",
    "RETURN_REFUND_READ_CONTRACT",
    "RETURN_REFUND_ROLE_AUTHORITY",
    "RETURN_REFUND_WAKE_EVENT_TYPES",
    "SUPPORTED_SCHEMA_VERSION",
    "ActorSpec",
    "AuthoredEvent",
    "ConditionalTrigger",
    "ExpectedEventRejection",
    "OperationSpec",
    "PolicySpec",
    "ReturnRefundSpec",
    "ScenarioSpec",
    "action_schema_view",
    "load_spec",
    "return_refund_action_evidence_contract",
    "validate_action_payload",
    "validate_event_payload",
]


ENVIRONMENT_ALLOCATED_IDENTITY_FIELDS = frozenset(
    {"authorisation_id", "checkpoint_id", "refund_request_id", "communication_id"}
)
