"""One synthetic month: claims, authoritative ledger, versioned human approval."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from typing import Any, cast

from operatebench.core.protocol import EpisodePlan, PlannedEvent, PlannedTrigger, Verdict
from operatebench.core.read_contract import ActionEvidenceContract
from operatebench.core.retrieval import ToolResult

from .spec import Spec, digest

START = "2026-09-01T09:00:00Z"
WAKE = ("imported", "ledger_arrived", "receipt_arrived", "approval", "quiet", "handoff")
SCHEMAS = {
    "request_receipt": {"transaction_id": "string", "document_id": "string"},
    "draft_report": {"source_version": "string"},
    "request_review": {"report_id": "string"},
    "publish_report": {"report_id": "string"},
    "request_handoff": {"transaction_id": "string"},
}
ISSUERS = {
    "imported": "connector",
    "ledger_arrived": "ledger",
    "receipt_arrived": "customer",
    "approval": "accountant",
    "quiet": "clock",
    "handoff": "accountant",
}


def source_version(state: Any) -> str:
    return digest([state["ledger"], state["receipt"]])


def matched(state: Any) -> bool:
    ledger, receipt = state["ledger"], state["receipt"]
    return bool(
        ledger
        and receipt
        and all(
            ledger[k] == receipt[k]
            for k in (
                "transaction_id",
                "document_id",
                "month",
                "amount_minor",
                "currency",
            )
        )
    )


class BookkeepingDomain:
    def __init__(self, spec: Spec, scenario: str) -> None:
        self.spec = spec
        self.scenario = spec.data["scenarios"][scenario]

    def record(self, version: int = 1) -> dict[str, Any]:
        return {
            k: self.spec.data[k]
            for k in (
                "transaction_id",
                "document_id",
                "month",
                "amount_minor",
                "currency",
            )
        } | {"version": version, "memo": "supplies" if version == 1 else "tools"}

    def build_plan(self) -> EpisodePlan:
        receipt = self.record()
        if self.scenario["receipt"] == "conflict":
            receipt["amount_minor"] = 11000
        events = [
            PlannedEvent(
                "import",
                "imported",
                "connector",
                1,
                at=START,
                payload={
                    "source_id": "connector-import-1",
                    "version": 1,
                    "status": self.scenario["connector"],
                    "count": 0 if self.scenario["connector"] == "failed" else 1,
                    "claimed_amount_minor": 12000,
                },
            ),
            PlannedEvent(
                "bank",
                "ledger_arrived",
                "ledger",
                2,
                at="2026-09-02T09:00:00Z",
                payload=self.record(),
            ),
        ]
        if self.scenario["receipt"] != "missing":
            events.append(
                PlannedEvent(
                    "receipt",
                    "receipt_arrived",
                    "customer",
                    3,
                    at="2026-09-02T09:05:00Z",
                    payload=receipt,
                )
            )
        if self.scenario["receipt"] == "conflict":
            events.append(
                PlannedEvent(
                    "clarification",
                    "receipt_arrived",
                    "customer",
                    4,
                    payload=self.record(2),
                    trigger=PlannedTrigger("action", "request_receipt"),
                    delay_minutes=15,
                )
            )
        if self.scenario["correction"]:
            events.append(
                PlannedEvent(
                    "correction",
                    "ledger_arrived",
                    "ledger",
                    5,
                    payload=self.record(2),
                    trigger=PlannedTrigger("action", "publish_report"),
                    delay_minutes=10,
                )
            )
        # An exogenous post-final event must be audited without resurrection.
        events.append(
            PlannedEvent(
                "too-late",
                "ledger_arrived",
                "ledger",
                6,
                at="2026-09-03T12:00:00Z",
                payload=self.record(3),
            )
        )
        return EpisodePlan(START, tuple(events), 4320, 24, max_invocations=80)

    def initial_state(self) -> dict[str, Any]:
        return {
            "binding": self.spec.content_digest,
            "company": self.spec.data["company"],
            "month": self.spec.data["month"],
            "transaction_id": "bank-line-1",
            "document_id": "receipt-1",
            "import": None,
            "ledger": None,
            "receipt": None,
            "history": [],
            "request_at": None,
            "report": None,
            "reports": {},
            "review_requested": None,
            "approved": None,
            "provisional": False,
            "provisional_at": None,
            "quiet": False,
            "handoff_requested": False,
            "handed_off": False,
            "final": False,
            "terminal": None,
        }

    def canonical_state(self, state: Any) -> dict[str, Any]:
        return deepcopy(state)

    def coarse_phase(self, state: Any) -> str:
        return (
            "FINAL"
            if state["final"]
            else "PROVISIONAL"
            if state["provisional"]
            else "OPEN"
        )

    def event_authority(self, actor_id: str, event_type: str) -> str:
        return (
            "system_of_record"
            if actor_id in ("ledger", "clock")
            else "authoritative_verification"
            if actor_id == "accountant"
            else "actor_claim"
        )

    def retrieval_catalogue(self) -> dict[str, Any]:
        return {
            "get_month": {
                "source": "month_case",
                "authority": "system_of_record",
                "schema_id": "synthetic.month.v1",
                "arguments": {},
            }
        }

    def retrieval_record_contract(self) -> list[str | int]:
        return ["month-2026-08", "synthetic month snapshot", "sha256-json-v1", 64]

    def read_requirements(self) -> ActionEvidenceContract:
        return ActionEvidenceContract(
            {k: {"get_month": "required"} for k in (*SCHEMAS, "complete")}
        )

    def serve_retrieval(self, state: Any, requests: Any, as_of: str) -> list[ToolResult]:
        records = {"record": deepcopy(state)}
        return [
            ToolResult(
                r.tool,
                "month_case",
                "system_of_record",
                "month-2026-08",
                digest(records),
                as_of,
                "synthetic.month.v1",
                records,
            )
            for r in requests
        ]

    def policy_view(self) -> dict[str, Any]:
        return {
            "scope": "Invented operational policy; SYN currency; no statutory rules.",
            "authority": "Connector status/count/upload success are claims. Only ledger "
            "Facts plus matching receipt support reconciliation. Only accountant "
            "approval of the exact report_id permits provisional publication.",
            "finality": "Wait 30 synthetic minutes after publication. Source correction "
            "Invalidates draft, approval and quiet timer. Missing receipt: "
            "scoped request. "
            "120 minute fallback then acknowledged unresolved handoff. "
            "Re-read after effects.",
        }

    def actor_roles(self) -> dict[str, str]:
        return {
            "connector": "import claimant",
            "ledger": "authoritative transaction system",
            "customer": "receipt issuer",
            "accountant": "human reviewer",
            "clock": "synthetic timer",
        }

    def message_fixture_ids(self) -> tuple[str, ...]:
        return ()

    def action_schemas(self) -> dict[str, Any]:
        return {k: {"required": fields, "optional": {}} for k, fields in SCHEMAS.items()}

    def wake_event_vocabulary(self) -> tuple[str, ...]:
        return WAKE

    def reduce_event(self, state: Any, event: Any, context: Any) -> Verdict:
        kind = event.event_type
        if not isinstance(event.payload, Mapping):
            return Verdict.refused(
                "IMPORT_PAYLOAD" if kind == "imported" else "EVENT_PAYLOAD",
                "event payload must be a mapping",
            )
        if kind == "imported":
            payload = event.payload
            if (
                not isinstance(payload, Mapping)
                or set(payload)
                != {"source_id", "version", "status", "count", "claimed_amount_minor"}
                or type(payload["source_id"]) is not str
                or not payload["source_id"].strip()
                or type(payload["status"]) is not str
                or payload["status"] not in ("ok", "failed")
                or any(
                    type(payload[key]) is not int or payload[key] < minimum
                    for key, minimum in (
                        ("version", 1),
                        ("count", 0),
                        ("claimed_amount_minor", 0),
                    )
                )
            ):
                return Verdict.refused("IMPORT_PAYLOAD", "invalid connector claim")
        p = dict(event.payload)
        if ISSUERS.get(kind) != event.actor_id:
            return Verdict.refused("AUTHORITY", "wrong source actor")
        if state["final"]:
            return Verdict.refused("FINALITY", "already final")
        if kind in ("ledger_arrived", "receipt_arrived"):
            key = "ledger" if kind == "ledger_arrived" else "receipt"
            if (
                set(p) != set(self.record())
                or any(
                    p[k] != self.spec.data[k]
                    for k in ("transaction_id", "document_id", "month", "currency")
                )
                or type(p["version"]) is not int
                or p["version"] < 1
                or type(p["amount_minor"]) is not int
                or p["amount_minor"] < 0
                or type(p["memo"]) is not str
            ):
                return Verdict.refused("SOURCE_SCOPE", "invalid record")
            previous = state[key]
            if previous and p["version"] <= previous["version"]:
                if previous == p:
                    return Verdict.ok()
                return Verdict.refused("SOURCE_VERSION", "reused or older version")
            state[key] = p
            state["history"].append({"source": event.actor_id, "record": p})
            state["report"] = state["review_requested"] = state["approved"] = None
            state["provisional"] = state["quiet"] = False
            state["provisional_at"] = None
        elif kind == "imported":
            state["import"] = p
        elif kind == "approval":
            if (
                set(p) != {"report_id", "decision"}
                or not state["report"]
                or state["review_requested"] != p.get("report_id")
                or p.get("decision") != "approved"
            ):
                return Verdict.refused(
                    "APPROVAL_VERSION", "approval not for active report"
                )
            state["approved"] = p["report_id"]
        elif kind == "quiet":
            if (
                set(p) != {"report_id"}
                or type(p["report_id"]) is not str
                or not p["report_id"].strip()
            ):
                return Verdict.refused("QUIET_PAYLOAD", "invalid quiet timer")
            if (
                state["provisional"]
                and p.get("report_id") == state["report"]["report_id"]
                and state["provisional_at"]
                and (
                    datetime.fromisoformat(context.now)
                    - datetime.fromisoformat(state["provisional_at"])
                ).total_seconds()
                >= 1800
            ):
                state["quiet"] = True
        elif kind == "handoff":
            if not state["handoff_requested"] or p != {"unresolved": True}:
                return Verdict.refused("HANDOFF", "no acknowledged unresolved handoff")
            state["handed_off"] = True
        context.record(
            "bookkeeping_event", {"kind": kind, "source": event.actor_id, "data": p}
        )
        return Verdict.ok()

    def apply_action(
        self, state: Any, action_type: str, payload: Any, evidence_refs: Any, context: Any
    ) -> Verdict:
        def refuse(code: str) -> Verdict:
            return Verdict.refused(code, "synthetic bookkeeping gate refused")

        if state["final"]:
            return refuse("FINALITY")
        if action_type not in SCHEMAS or set(payload) != set(SCHEMAS[action_type]):
            return refuse("PAYLOAD")
        if any(type(v) is not str or not v for v in payload.values()):
            return refuse("PAYLOAD")
        if action_type == "request_receipt":
            if (
                payload
                != {
                    "transaction_id": state["transaction_id"],
                    "document_id": state["document_id"],
                }
                or not state["ledger"]
                or matched(state)
                or state["request_at"]
            ):
                return refuse("REQUEST_SCOPE")
            state["request_at"] = context.now
        elif action_type == "draft_report":
            if not matched(state):
                code = (
                    "IMPORT_NOT_LEDGER"
                    if not state["ledger"]
                    and state["import"]
                    and state["import"]["status"] == "failed"
                    else "UNVERIFIED_SOURCES"
                )
                return refuse(code)
            version = source_version(state)
            if payload["source_version"] != version or version not in evidence_refs:
                return refuse("SOURCE_VERSION")
            report = {
                "report_id": "report-" + version,
                "source_version": version,
                "ledger_version": state["ledger"]["version"],
                "receipt_version": state["receipt"]["version"],
                "amount_minor": state["ledger"]["amount_minor"],
                "month": state["month"],
            }
            if state["report"]:
                return refuse("DUPLICATE")
            state["report"] = report
            state["reports"][report["report_id"]] = deepcopy(report)
        elif action_type in ("request_review", "publish_report"):
            report = state["report"]
            if not report:
                return refuse("UNVERIFIED_SOURCES")
            if (
                payload["report_id"] != report["report_id"]
                or report["report_id"] not in evidence_refs
            ):
                return refuse("REPORT_VERSION")
            if action_type == "request_review":
                if state["review_requested"]:
                    return refuse("DUPLICATE")
                state["review_requested"] = report["report_id"]
                context.schedule_timer(
                    "review-" + report["report_id"],
                    "approval",
                    "accountant",
                    15,
                    {"report_id": report["report_id"], "decision": "approved"},
                )
            else:
                if state["approved"] != report["report_id"]:
                    return refuse("APPROVAL_REQUIRED")
                if state["provisional"]:
                    return refuse("DUPLICATE")
                state["provisional"] = True
                state["provisional_at"] = context.now
                context.schedule_timer(
                    "quiet-" + report["report_id"],
                    "quiet",
                    "clock",
                    30,
                    {"report_id": report["report_id"]},
                )
        elif action_type == "request_handoff":
            requested = state["request_at"]
            if (
                payload["transaction_id"] != state["transaction_id"]
                or matched(state)
                or not requested
                or state["handoff_requested"]
                or (
                    datetime.fromisoformat(context.now)
                    - datetime.fromisoformat(requested)
                ).total_seconds()
                < 7200
            ):
                return refuse("HANDOFF")
            state["handoff_requested"] = True
            context.schedule_timer(
                "handoff", "handoff", "accountant", 15, {"unresolved": True}
            )
        context.record(
            "bookkeeping_action", {"action": action_type, "data": dict(payload)}
        )
        return Verdict.ok()

    def apply_terminal(self, state: Any, evidence_refs: Any, context: Any) -> Verdict:
        if state["handed_off"] and not state["provisional"]:
            outcome = "UNRESOLVED_HANDOFF"
        elif (
            matched(state)
            and state["report"]
            and state["provisional"]
            and state["quiet"]
            and state["approved"] == state["report"]["report_id"]
            and state["report"]["source_version"] == source_version(state)
        ):
            outcome = "REVIEWED_RELEASE"
        else:
            return Verdict.refused(
                "FINALITY", "unknown or unreviewed state cannot complete"
            )
        state["final"], state["terminal"] = True, outcome
        return Verdict.ok()

    def is_replay_final(self, state: Any) -> bool:
        return cast(bool, state["final"])

    def terminal_outcome(self, state: Any) -> str | None:
        return cast(str | None, state["terminal"])

    def current_cycle_id(self, state: Any) -> str:
        return source_version(state)
