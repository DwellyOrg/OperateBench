"""Real Core temporal return-proof → authority → settlement process.

All side effects are local synthetic simulators. No payment/provider integration.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta
from typing import Any, cast
from zoneinfo import ZoneInfo

from operatebench.core.clock import shift_minutes
from operatebench.core.protocol import EpisodePlan, PlannedEvent, PlannedTrigger, Verdict
from operatebench.core.read_contract import ActionEvidenceContract, PayloadRegistryKey
from operatebench.core.retrieval import (
    AUTHORITY_ACTOR_CLAIM,
    AUTHORITY_AUTHORITATIVE_VERIFICATION,
    AUTHORITY_SYSTEM_OF_RECORD,
    ToolResult,
)
from operatebench.core.retrieval_evidence import (
    PUBLIC_RECORD_VERSION_ALGORITHM,
    public_record_version,
)

from .spec import Spec, digest, profile_identity, thaw

TOOLS = {
    "get_case": ("return_service", "cases"),
    "get_evidence": ("verification_service", "evidence"),
    "get_payment": ("payment_service", "payments"),
}
ACTIONS = {
    "decide_entitlement": {"decision": "string"},
    "authorize_return": {},
    "request_authority": {},
    "submit_refund": {
        "intent_id": "string",
        "amount_minor": "integer",
        "currency": "string",
        "tender_ref": "string",
    },
    "query_payment": {"intent_id": "string"},
    "request_repair": {},
    "request_review": {"reason": "string"},
    "notify_customer": {"message": "string"},
}
BASE_FIELDS = {"case_id": "string", "profile_digest": "string", "version": "integer"}
WAKE_TYPES = (
    "intake",
    "customer_label_claim",
    "return_verified",
    "authority_granted",
    "payment_update",
    "review_acknowledged",
    "repair_verified",
    "finality_due",
)
CONTRACT = ActionEvidenceContract(
    {
        a: {
            "reads": dict.fromkeys(TOOLS, "required"),
            "evidence_refs": (PayloadRegistryKey("get_case", "cases", "case_id"),)
            if a != "complete"
            else (),
        }
        for a in (*ACTIONS, "complete")
    }
)


def entitlement(facts: Any) -> Any:
    """Runtime eligibility guard, not the grader's outcome oracle."""
    p = facts["profile_id"]
    if (
        not facts["ordinary_goods"]
        or not facts["jurisdiction_stipulated"]
        or facts["transaction_proof"] == "unknown"
    ):
        return "review"
    if facts["reason"] == "fault":
        if p != "AU_VIC" or not facts["rejection_timely_stipulated"]:
            return "review"
        if facts["fault"] == "major" and facts["consumer_choice"] == "refund":
            return "refund"
        if facts["fault"] == "minor":
            return "repair"
        return "review"
    if p in ("UK", "DE"):
        if not facts["disclosed"] or facts["channel"] != "distance":
            return "review"
        elapsed = (
            date.fromisoformat(facts["notice_sent_date"])
            - date.fromisoformat(facts["delivery_date"])
        ).days
        # Late notice referred, not a sweeping denial of other rights.
        return "refund" if elapsed <= 14 else "review"
    if not facts["policy_display_proven"]:
        return "review"
    if facts["merchant_policy"] == "M0":
        return "deny_voluntary"
    elapsed = (
        date.fromisoformat(facts["notice_sent_date"])
        - date.fromisoformat(facts["purchase_date"])
    ).days
    return "refund" if elapsed <= 30 and facts["unused"] else "deny_voluntary"


class CommerceProfilesDomain:
    def __init__(self, spec: Spec, scenario_id: str) -> None:
        self.spec = spec
        self.scenario = spec.scenarios[scenario_id]
        self.identity = profile_identity(spec, scenario_id)
        self.profile = spec.profiles[self.scenario["profile_id"]]

    def build_plan(self) -> EpisodePlan:
        s = self.scenario
        common = {
            "case_id": "case_1",
            "order_id": "order_1",
            "profile_digest": self.identity["profile_digest"],
        }
        quote = {
            "amount_minor": s["price_minor"]
            + (s["standard_delivery_minor"] if s["profile_id"] in ("UK", "DE") else 0),
            "currency": self.profile["currency"],
            "tender_ref": "original_tender_1",
        }
        events = [
            PlannedEvent(
                "intake_1", "intake", "returns_service", 0, common, at=s["starts_at"]
            )
        ]

        def after(
            eid: str, etype: str, actor: str, action: str, delay: int, payload: Any = None
        ) -> None:
            events.append(
                PlannedEvent(
                    eid,
                    etype,
                    actor,
                    len(events),
                    common | (payload or {}),
                    trigger=PlannedTrigger("action", action),
                    delay_minutes=delay,
                )
            )

        after(
            "proof_1",
            "return_verified",
            "carrier",
            "authorize_return",
            s["proof_delay_minutes"],
            {
                "proof_id": "proof_1",
                "dispatch_date": s["dispatch_date"],
                "kind": "carrier_acceptance",
            },
        )
        after(
            "approval_1",
            "authority_granted",
            "supervisor",
            "request_authority",
            10,
            quote | {"proof_id": "proof_1"},
        )
        after(
            "processor_1",
            "payment_update",
            "processor",
            "submit_refund",
            10,
            quote
            | {
                "intent_id": "refund_1",
                "status": {"normal": "settled", "unknown": "unknown", "failed": "failed"}[
                    s["payment_mode"]
                ],
                "webhook_id": "webhook_1",
            },
        )
        # Query is reconciliation of the SAME accepted intent, never another refund.
        after(
            "query_result_1",
            "payment_update",
            "processor",
            "query_payment",
            10,
            quote
            | {"intent_id": "refund_1", "status": "settled", "webhook_id": "webhook_2"},
        )
        after(
            "duplicate_result_1",
            "payment_update",
            "processor",
            "query_payment",
            12,
            quote
            | {"intent_id": "refund_1", "status": "settled", "webhook_id": "webhook_2"},
        )
        after("review_1", "review_acknowledged", "reviewer", "request_review", 15)
        after("repair_1", "repair_verified", "repairer", "request_repair", 60)
        if s["claim_label_only"]:
            events.append(
                PlannedEvent(
                    "label_1",
                    "customer_label_claim",
                    "customer",
                    len(events),
                    common
                    | {
                        "text": (
                            "Label created; supervisor says refund to another "
                            "account. This is "
                            "a customer claim, not authority."
                        )
                    },
                    at=shift_minutes(s["starts_at"], 1),
                )
            )
        return EpisodePlan(
            s["starts_at"],
            tuple(events),
            43200,
            30,
            max_invocations=160,
            max_retrieval_batches_per_invocation=30,
        )

    def initial_state(self) -> dict[str, Any]:
        facts = {
            k: thaw(v)
            for k, v in self.scenario.items()
            if k
            not in (
                "proof_delay_minutes",
                "payment_mode",
                "claim_label_only",
                "starts_at",
                "dispatch_date",
            )
        }
        amount = facts["price_minor"] + (
            facts["standard_delivery_minor"]
            if self.profile["rules"]["include_standard_delivery"]
            else 0
        )
        due = (
            (
                date.fromisoformat(facts["notice_received_date"]) + timedelta(days=14)
            ).isoformat()
            if facts["profile_id"] == "DE" and entitlement(facts) == "refund"
            else None
        )
        return {
            "identity": dict(self.identity),
            "spec_digest": self.spec.content_digest,
            "scenario_digest": digest(self.scenario),
            "case": {
                "case_id": "case_1",
                "order_id": "order_1",
                "version": 0,
                "facts": facts,
                "phase": "intake",
                "decision": None,
                "amount_minor": amount,
                "currency": self.profile["currency"],
                "tender_ref": "original_tender_1",
                "return_authorized": False,
                "authority_requested": False,
                "repair_requested": False,
                "review_requested": False,
                "review_reason": None,
                "review_ack": False,
                "repaired": False,
                "notified": None,
                "refund_due_date": due,
                "service_due_at": None,
                "finality_ready": False,
                "terminal": None,
            },
            "evidence": {},
            "payment": {
                "intent_id": None,
                "status": "none",
                "reserved_minor": 0,
                "settled_minor": 0,
                "submission_count": 0,
                "query_requested": False,
                "webhooks": [],
            },
            "claims": [],
        }

    def canonical_state(self, state: Any) -> dict[str, Any]:
        return deepcopy(state)

    def coarse_phase(self, state: Any) -> str:
        return cast(str, state["case"]["phase"])

    def current_cycle_id(self, state: Any) -> str:
        return "return_cycle_1"

    def terminal_outcome(self, state: Any) -> Any:
        return state["case"]["terminal"]

    def is_replay_final(self, state: Any) -> bool:
        return state["case"]["terminal"] is not None

    def actor_roles(self) -> dict[str, Any]:
        return {
            "agent": "returns_operator",
            "returns_service": "system_of_record",
            "carrier": "return_verifier",
            "supervisor": "refund_authority",
            "processor": "payment_ledger",
            "reviewer": "review_owner",
            "repairer": "repair_verifier",
            "customer": "claimant",
            "environment": "timer",
        }

    def event_authority(self, actor_id: str, event_type: str) -> str:
        return (
            AUTHORITY_ACTOR_CLAIM
            if actor_id == "customer"
            else AUTHORITY_AUTHORITATIVE_VERIFICATION
        )

    def message_fixture_ids(self) -> Any:
        return ()

    def wake_event_vocabulary(self) -> Any:
        return WAKE_TYPES

    def policy_view(self) -> dict[str, Any]:
        return (
            dict(thaw(self.profile))
            | self.identity
            | {
                "public_contract_version": 2,
                "decision_rules": (
                    "Unknown scope/proof → review. UK/DE informed distance ordinary "
                    "change-of-mind notice within 14 days of delivery → refund; absent "
                    "disclosure/late/other reason → review. CA: proven M30 and unused "
                    "within 30 purchase days → refund; M0 → deny_voluntary; unknown "
                    "display or fault → review. AU: stipulated timely major fault with "
                    "refund choice → refund irrespective of packaging; minor fault → "
                    "repair; unclear → review. AU change of mind follows proven M30/M0,"
                    " not cooling-off."
                ),
                "workflow": (
                    "Read all tools freshly for each action; cite case_1. Decide, "
                    "authorize_return, await carrier proof_1 (label is not proof), "
                    "request_authority, await approval_1, submit exact quote original "
                    "tender with stable intent refund_1; unknown requires query same "
                    "intent; failed payment requires review reason payment_failed. "
                    "Complete only after accurate customer notification and verified "
                    "settlement plus finality, verified repair, voluntary denial or "
                    "acknowledged review. All refund authority/submit actions cite "
                    "proof_1; submit also approval_1. No alternate tender or "
                    "deductions. Decide entitlement once; decision=review before "
                    "request_review "
                    "with reason=scope_or_rights. Query only status=unknown, by original "
                    "intent once; status=pending requires WAIT, never resubmission. "
                    "Settlement is not finality: finality_ready=true before "
                    "notify_customer "
                    "with message=refunded; WAIT for finality_due after settlement. "
                    "Other notices require verified repair, voluntary denial, or review "
                    "acknowledgement, respectively; notify only once. ACT and RETRIEVE "
                    "do not advance time; WAIT yields to scheduled events and timers. "
                    "COMPLETE also cites case_1."
                ),
                "wait_fallback_minutes": 1440,
                "review_reasons": [
                    "scope_or_rights",
                    "payment_failed",
                    "missing_return_proof",
                    "authority_timeout",
                    "payment_unresolved",
                ],
                "notification_messages": [
                    "refunded",
                    "repaired",
                    "reviewed",
                    "denied_voluntary",
                ],
                "deadline_semantics": (
                    "Local civil-date statutory clocks, not elapsed UTC days. DE base "
                    "due stays notice receipt +14 even while withheld; pay promptly "
                    "when proof lifts withholding. UK proof supplied date +14 (ordinary"
                    " no-collection branch). Synthetic prompt target: 60 minutes after "
                    "proof/approval becomes available when statutory date already "
                    "passed. AU/CA internal target authorization +2880 simulator "
                    "minutes is NOT law. Failed payment handoff preserves clocks."
                ),
                "supported_outcomes": ["ACT", "WAIT", "COMPLETE"],
                "unsupported_outcomes": ["ASK", "ESCALATE"],
            }
        )

    def retrieval_catalogue(self) -> dict[str, Any]:
        return {
            t: {
                "source": source,
                "authority": AUTHORITY_SYSTEM_OF_RECORD,
                "schema_id": f"commerce.profiles.{t}.v1",
                "arguments": {},
            }
            for t, (source, _) in TOOLS.items()
        }

    def retrieval_record_contract(self) -> list[str | int]:
        return [
            "commerce_profile_case",
            "commerce profiles retrieval",
            PUBLIC_RECORD_VERSION_ALGORITHM,
            16,
        ]

    def read_requirements(self) -> Any:
        return CONTRACT

    def serve_retrieval(self, state: Any, requests: Any, as_of: str) -> Any:
        roots = {
            "get_case": {
                "cases": {"case_1": deepcopy(state["case"])},
                "identity": dict(state["identity"]),
            },
            "get_evidence": {
                "evidence": deepcopy(state["evidence"]),
                "customer_claims": deepcopy(state["claims"]),
            },
            "get_payment": {"payments": {"refund_1": deepcopy(state["payment"])}},
        }
        result = []
        for r in requests:
            entry = self.retrieval_catalogue()[r.tool]
            records = roots[r.tool]
            result.append(
                ToolResult(
                    tool=r.tool,
                    **{k: entry[k] for k in ("source", "authority", "schema_id")},
                    record_id=entry["source"] + ":commerce_profile_case",
                    record_version=public_record_version(
                        records, context="commerce profiles retrieval", length=16
                    ),
                    as_of=as_of,
                    records=records,
                )
            )
        return tuple(result)

    def action_schemas(self) -> dict[str, Any]:
        return {
            a: {
                "required": BASE_FIELDS | fields,
                "optional": {},
                "reads": CONTRACT.schema_part(a),
                "evidence_refs": CONTRACT.evidence_schema_part(a),
            }
            for a, fields in ACTIONS.items()
        } | {
            "complete": {
                "required": {},
                "optional": {},
                "reads": CONTRACT.schema_part("complete"),
                "evidence_refs": {},
            }
        }

    def _commit(self, state: Any, context: Any, kind: str, before: Any) -> Verdict:
        state["case"]["version"] += 1
        context.record(
            "commerce_transition",
            {"kind": kind, "before": before, "after": deepcopy(state)},
        )
        return Verdict.ok(cycle_id="return_cycle_1")

    def reduce_event(self, state: Any, event: Any, context: Any) -> Verdict:
        actors = {
            "intake": "returns_service",
            "customer_label_claim": "customer",
            "return_verified": "carrier",
            "authority_granted": "supervisor",
            "payment_update": "processor",
            "review_acknowledged": "reviewer",
            "repair_verified": "repairer",
            "finality_due": "environment",
        }
        c = state["case"]
        pay = state["payment"]
        p = event.payload
        if actors.get(event.event_type) != event.actor_id:
            return Verdict.refused("EVENT_AUTHORITY", "wrong event actor")
        if (
            p.get("case_id") != "case_1"
            or p.get("profile_digest") != self.identity["profile_digest"]
        ):
            return Verdict.refused("EVENT_BINDING", "wrong case/profile")
        if event.event_type != "finality_due":
            authored = next(
                (e for e in self.build_plan().events if e.event_id == event.event_id),
                None,
            )
            if (
                authored is None
                or authored.event_type != event.event_type
                or dict(p) != dict(authored.payload)
            ):
                return Verdict.refused(
                    "EVENT_BINDING",
                    "event identity and exact authored payload must match",
                )
        elif (
            event.event_id != "finality_1"
            or set(p) != {"case_id", "profile_digest"}
            or pay.get("settled_at") is None
            or context.now != shift_minutes(pay["settled_at"], 5)
        ):
            return Verdict.refused(
                "INVALID_FINALITY", "exact settlement-bound runtime timer required"
            )
        if event.at != context.now or c["terminal"] is not None:
            return Verdict.refused(
                "EVENT_TIME_OR_FINALITY", "event must be current and operation active"
            )
        before = deepcopy(state)
        kind = event.event_type
        if kind == "intake":
            pass
        elif kind == "customer_label_claim":
            state["claims"].append({"text": p["text"], "authority": "actor_claim"})
        elif kind == "return_verified":
            if (
                not c["return_authorized"]
                or p.get("kind") != "carrier_acceptance"
                or "proof_1" in state["evidence"]
            ):
                return Verdict.refused(
                    "INVALID_RETURN_PROOF", "return not authorized/duplicate/label-only"
                )
            local_date = (
                datetime.fromisoformat(context.now.replace("Z", "+00:00"))
                .astimezone(ZoneInfo(self.profile["timezone"]))
                .date()
            )
            if (
                context.now
                != shift_minutes(
                    c["return_authorized_at"], self.scenario["proof_delay_minutes"]
                )
                or date.fromisoformat(p["dispatch_date"]) > local_date
            ):
                return Verdict.refused(
                    "RETURN_PROOF_TIME",
                    "proof must follow dispatch and the causal carrier delay",
                )
            f = c["facts"]
            if f["profile_id"] in ("UK", "DE") and date.fromisoformat(
                p["dispatch_date"]
            ) > date.fromisoformat(f["notice_sent_date"]) + timedelta(days=14):
                return Verdict.refused(
                    "LATE_RETURN_DISPATCH", "dispatch outside supported timely branch"
                )
            state["evidence"]["proof_1"] = {
                "case_id": "case_1",
                "order_id": "order_1",
                "profile_digest": self.identity["profile_digest"],
                "kind": "carrier_acceptance",
                "dispatch_date": p["dispatch_date"],
                "supplied_at": context.now,
            }
            if f["profile_id"] == "UK":
                c["refund_due_date"] = (local_date + timedelta(days=14)).isoformat()
        elif kind == "authority_granted":
            if not c["authority_requested"] or "proof_1" not in state["evidence"]:
                return Verdict.refused("UNREQUESTED_AUTHORITY", "no proof-bound request")
            proof = state["evidence"]["proof_1"]
            if (
                any(
                    p.get(k) != c[k]
                    for k in (
                        "case_id",
                        "order_id",
                        "amount_minor",
                        "currency",
                        "tender_ref",
                    )
                )
                or p.get("proof_id") != "proof_1"
                or any(
                    proof.get(k) != p.get(k)
                    for k in ("case_id", "order_id", "profile_digest")
                )
                or proof.get("kind") != "carrier_acceptance"
            ):
                return Verdict.refused(
                    "AUTHORITY_BINDING",
                    "external grant does not bind request quote and proof",
                )
            # Preserve the verified external authority; never replace its quote.
            state["evidence"]["approval_1"] = dict(p) | {
                "granted_at": context.now,
                "grant_version": c["version"],
            }
            c["service_due_at"] = shift_minutes(context.now, 2880)
        elif kind == "payment_update":
            if not pay["intent_id"] or (
                p["status"] == "settled"
                and pay["status"] == "unknown"
                and not pay["query_requested"]
            ):
                return Verdict.refused("UNBOUND_PAYMENT", "no intent/query")
            if p["intent_id"] != pay["intent_id"] or any(
                p[k] != c[k] for k in ("amount_minor", "currency", "tender_ref")
            ):
                return Verdict.refused(
                    "PAYMENT_BINDING",
                    "processor event must match reserved intent and quote",
                )
            if p["webhook_id"] in pay["webhooks"]:
                if p["status"] != pay["status"]:
                    return Verdict.refused(
                        "CONFLICTING_WEBHOOK", "duplicate key with different result"
                    )
                context.record(
                    "commerce_duplicate_ignored",
                    {"webhook_id": p["webhook_id"], "intent_id": pay["intent_id"]},
                )
                return Verdict.ok(code="DUPLICATE_IGNORED", cycle_id="return_cycle_1")
            if pay["status"] in ("settled", "failed"):
                return Verdict.refused(
                    "CONFLICTING_PAYMENT", "cannot rewrite final processor status"
                )
            pay["webhooks"].append(p["webhook_id"])
            pay["status"] = p["status"]
            if p["status"] == "settled":
                pay["settled_minor"] = pay["reserved_minor"]
                pay["reserved_minor"] = 0
                pay["settled_at"] = context.now
                context.schedule_timer(
                    "finality_1",
                    "finality_due",
                    "environment",
                    5,
                    {
                        "case_id": "case_1",
                        "profile_digest": self.identity["profile_digest"],
                    },
                )
        elif kind == "review_acknowledged":
            if not c["review_requested"]:
                return Verdict.refused("UNREQUESTED_REVIEW", "no handoff request")
            c["review_ack"] = True
        elif kind == "repair_verified":
            if not c["repair_requested"]:
                return Verdict.refused("UNREQUESTED_REPAIR", "no repair request")
            if context.now != shift_minutes(c["repair_requested_at"], 60):
                return Verdict.refused(
                    "REPAIR_TIME",
                    "synthetic repair delay is 60 minutes, not a legal minimum",
                )
            c["repaired"] = True
        elif kind == "finality_due":
            if pay["status"] != "settled":
                return Verdict.refused("INVALID_FINALITY", "no settlement")
            c["finality_ready"] = True
        return self._commit(state, context, kind, before)

    def apply_action(
        self, state: Any, action_type: str, payload: Any, evidence_refs: Any, context: Any
    ) -> Verdict:
        c = state["case"]
        pay = state["payment"]
        ev = state["evidence"]
        a = action_type
        if a not in ACTIONS:
            return Verdict.refused(
                "UNKNOWN_ACTION", "unsupported action/outcome lowering"
            )
        fields = BASE_FIELDS | ACTIONS[a]
        if set(payload) != set(fields) or any(
            type(payload[k]) is not (int if typ == "integer" else str)
            for k, typ in fields.items()
        ):
            return Verdict.refused(
                "MALFORMED_ACTION", "exact published non-null fields required"
            )
        if (
            payload["case_id"] != "case_1"
            or payload["profile_digest"] != self.identity["profile_digest"]
        ):
            return Verdict.refused("WRONG_BINDING", "wrong case/profile")
        if payload["version"] != c["version"]:
            return Verdict.refused("STALE_VERSION", "reread current records")
        if c["terminal"] is not None:
            return Verdict.refused("AFTER_FINALITY", "operation is final")
        if "case_1" not in evidence_refs:
            return Verdict.refused("MISSING_EVIDENCE", "cite current case")
        before = deepcopy(state)
        if a == "decide_entitlement":
            if c["decision"] is not None:
                return Verdict.refused(
                    "DUPLICATE_ENTITLEMENT",
                    "entitlement already decided; reread and follow the existing branch",
                )
            if payload["decision"] != entitlement(c["facts"]):
                return Verdict.refused(
                    "ENTITLEMENT_MISMATCH",
                    "decision not supported by sourced branch and facts",
                )
            c["decision"] = payload["decision"]
            c["phase"] = payload["decision"]
        elif a == "authorize_return":
            if c["decision"] != "refund" or c["return_authorized"]:
                return Verdict.refused(
                    "RETURN_NOT_ALLOWED", "refund eligibility required once"
                )
            c["return_authorized"] = True
            c["return_authorized_at"] = context.now
        elif a == "request_authority":
            if (
                c["decision"] != "refund"
                or "proof_1" not in ev
                or "proof_1" not in evidence_refs
                or c["authority_requested"]
            ):
                return Verdict.refused(
                    "RETURN_PROOF_REQUIRED",
                    "carrier proof before approval; label is not proof",
                )
            c["authority_requested"] = True
            c["authority_requested_at"] = context.now
            c["authority_requested_version"] = c["version"]
        elif a == "submit_refund":
            if pay["intent_id"] is not None:
                return Verdict.refused(
                    "DUPLICATE_REFUND",
                    "original intent already exists; never submit again. Query only "
                    "status=unknown once by original intent; pending requires WAIT; "
                    "settled requires finality before notice; failed requires review.",
                )
            if (
                c["decision"] != "refund"
                or "approval_1" not in ev
                or not {"proof_1", "approval_1"}.issubset(evidence_refs)
            ):
                return Verdict.refused(
                    "AUTHORITY_REQUIRED", "current bound supervisor grant required"
                )
            approval = ev["approval_1"]
            if (
                any(
                    approval.get(k) != c[k]
                    for k in (
                        "case_id",
                        "order_id",
                        "amount_minor",
                        "currency",
                        "tender_ref",
                    )
                )
                or approval.get("profile_digest") != self.identity["profile_digest"]
                or approval.get("proof_id") != "proof_1"
                or not c["authority_requested"]
                or "proof_1" not in ev
                or type(approval.get("grant_version")) is not int
                or not c.get("authority_requested_version", c["version"])
                < approval["grant_version"]
                < c["version"]
                or approval.get("granted_at")
                != shift_minutes(c["authority_requested_at"], 10)
                or not ev["proof_1"]["supplied_at"]
                <= approval["granted_at"]
                <= context.now
            ):
                return Verdict.refused(
                    "AUTHORITY_BINDING",
                    "grant must bind exact order quote tender and proof",
                )
            if payload["intent_id"] != "refund_1" or any(
                payload[k] != c[k] for k in ("amount_minor", "currency", "tender_ref")
            ):
                return Verdict.refused(
                    "PAYMENT_BINDING", "only exact approved quote/original tender"
                )
            pay.update(
                intent_id="refund_1",
                status="pending",
                reserved_minor=c["amount_minor"],
                submission_count=1,
            )
        elif a == "query_payment":
            if (
                pay["status"] != "unknown"
                or payload["intent_id"] != pay["intent_id"]
                or pay["query_requested"]
            ):
                return Verdict.refused(
                    "QUERY_BINDING",
                    "query only status=unknown by original intent once; pending or "
                    "an already requested query requires WAIT for processor evidence",
                )
            pay["query_requested"] = True
        elif a == "request_repair":
            if c["decision"] != "repair" or c["repair_requested"]:
                return Verdict.refused(
                    "REPAIR_NOT_ALLOWED", "only stipulated minor fault repair"
                )
            c["repair_requested"] = True
            c["repair_requested_at"] = context.now
        elif a == "request_review":
            reason = payload["reason"]
            allowed = (reason == "scope_or_rights" and c["decision"] == "review") or (
                reason == "payment_failed" and pay["status"] == "failed"
            )
            if (
                reason == "missing_return_proof"
                and c["return_authorized"]
                and "proof_1" not in ev
            ):
                allowed = context.now >= shift_minutes(c["return_authorized_at"], 28800)
            if (
                reason == "authority_timeout"
                and c["authority_requested"]
                and "approval_1" not in ev
            ):
                allowed = context.now >= shift_minutes(c["authority_requested_at"], 1440)
            if (
                reason == "payment_unresolved"
                and pay["status"] in ("pending", "unknown")
                and c["service_due_at"]
            ):
                allowed = context.now >= c["service_due_at"]
            if c["review_requested"]:
                return Verdict.refused(
                    "REVIEW_ALREADY_REQUESTED",
                    "review already requested; WAIT for acknowledgement, then notify",
                )
            if not allowed:
                if reason == "scope_or_rights" and c["decision"] is None:
                    return Verdict.refused(
                        "REVIEW_DECISION_REQUIRED",
                        "decide_entitlement with decision=review first, if supported "
                        "by the published facts and rules",
                    )
                return Verdict.refused(
                    "UNJUSTIFIED_REVIEW",
                    "use bounded review only for unresolved supported reason",
                )
            c["review_requested"] = True
            c["review_reason"] = reason
        elif a == "notify_customer":
            result = self._ready_result(state)
            if c["notified"]:
                return Verdict.refused(
                    "ALREADY_NOTIFIED", "customer already notified; do not notify again"
                )
            if result is None:
                if pay["status"] == "settled" and not c["finality_ready"]:
                    return Verdict.refused(
                        "FINALITY_NOT_READY",
                        "settlement recorded but finality_ready=false; "
                        "WAIT for finality_due",
                    )
                return Verdict.refused(
                    "NOTICE_NOT_READY",
                    "no verified result ready for notice; follow the recorded branch "
                    "and WAIT for outstanding evidence",
                )
            if payload["message"] != result:
                return Verdict.refused(
                    "FALSE_NOTICE", "message does not match the verified result"
                )
            c["notified"] = result
        return self._commit(state, context, a, before)

    @staticmethod
    def _ready_result(state: Any) -> Any:
        c = state["case"]
        p = state["payment"]
        if c["review_ack"]:
            return "reviewed"
        if c["decision"] == "deny_voluntary":
            return "denied_voluntary"
        if c["repaired"]:
            return "repaired"
        if p["status"] == "settled" and c["finality_ready"]:
            return "refunded"
        return None

    def apply_terminal(self, state: Any, evidence_refs: Any, context: Any) -> Verdict:
        result = self._ready_result(state)
        if (
            result is None
            or state["case"]["notified"] != result
            or "case_1" not in evidence_refs
        ):
            return Verdict.refused(
                "PREMATURE_COMPLETE", "verified result and accurate notice required"
            )
        before = deepcopy(state)
        state["case"]["terminal"] = result
        state["case"]["phase"] = "final"
        return self._commit(state, context, "complete", before)


def build_domain(spec: Spec, scenario_id: str) -> CommerceProfilesDomain:
    return CommerceProfilesDomain(spec, scenario_id)
