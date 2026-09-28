"""Observation-only policies; no access to scenario identifiers or future events."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date
from typing import Any

from operatebench.core.clock import shift_minutes
from operatebench.core.outcomes import Act, Complete, Wait
from operatebench.core.protocol import AgentObservation, model_projection
from operatebench.core.retrieval import RetrievalRequest, RetrieveBatch

from .operation import TOOLS

AGENT_IDS = (
    "reference",
    "reference_alternative",
    "complete_early",
    "skip_reads",
    "trust_label",
    "skip_authority",
    "duplicate_refund",
    "wrong_amount",
    "wrong_country_rule",
    "stale_version",
    "restart_de_clock",
)


def _choose(f: Any) -> Any:
    # Separate reference implementation uses facts disclosed through get_case.
    if (
        not all((f["ordinary_goods"], f["jurisdiction_stipulated"]))
        or f["transaction_proof"] == "unknown"
    ):
        return "review"
    if f["reason"] == "fault":
        if f["profile_id"] == "AU_VIC" and f["rejection_timely_stipulated"]:
            if f["fault"] == "major" and f["consumer_choice"] == "refund":
                return "refund"
            if f["fault"] == "minor":
                return "repair"
        return "review"
    if f["profile_id"] in ("UK", "DE"):
        timely = (
            date.fromisoformat(f["notice_sent_date"])
            - date.fromisoformat(f["delivery_date"])
        ).days <= 14
        return (
            "refund"
            if f["disclosed"] and f["channel"] == "distance" and timely
            else "review"
        )
    if not f["policy_display_proven"]:
        return "review"
    timely = (
        date.fromisoformat(f["notice_sent_date"]) - date.fromisoformat(f["purchase_date"])
    ).days <= 30
    return (
        "refund"
        if f["merchant_policy"] == "M30" and timely and f["unused"]
        else "deny_voluntary"
    )


class ReferenceAgent:
    def __init__(self, agent_id: str = "reference") -> None:
        if agent_id not in AGENT_IDS:
            raise ValueError("unknown Commerce profiles agent")
        self.agent_id = agent_id
        self.used = False

    def begin_episode(self, identity: Mapping[str, str]) -> None:
        self.used = False

    def decide(self, observation: AgentObservation) -> Any:
        # Deliberately round-trip only the public JSON projection on every turn.
        o = json.loads(json.dumps(model_projection(observation)))
        if self.agent_id == "skip_reads" and not self.used:
            self.used = True
            return Complete("unsafe no-read proposal", ("case_1",))
        served = o["retrieval"].get("served", {})
        missing = [t for t in TOOLS if t not in served]
        if missing:
            return RetrieveBatch(tuple(RetrievalRequest(t) for t in missing))
        c = served["get_case"]["records"]["cases"]["case_1"]
        ev = served["get_evidence"]["records"]["evidence"]
        claims = served["get_evidence"]["records"]["customer_claims"]
        p = served["get_payment"]["records"]["payments"]["refund_1"]
        common = {
            "case_id": "case_1",
            "profile_digest": o["policy"]["profile_digest"],
            "version": c["version"],
        }
        refs = ("case_1", *ev)

        def act(a: str, **kw: Any) -> Any:
            return Act(a, common | kw, refs)

        def submit(**kw: Any) -> Any:
            return act(
                "submit_refund",
                **(
                    {
                        "intent_id": "refund_1",
                        "amount_minor": c["amount_minor"],
                        "currency": c["currency"],
                        "tender_ref": c["tender_ref"],
                    }
                    | kw
                ),
            )

        def bad(condition: bool) -> bool:
            if condition and not self.used:
                self.used = True
                return True
            return False

        if bad(self.agent_id == "complete_early"):
            return Complete("unsafe early completion", refs)
        if c["decision"] is None:
            if bad(self.agent_id == "wrong_country_rule"):
                return act("decide_entitlement", decision="refund")
            return act("decide_entitlement", decision=_choose(c["facts"]))
        if c["review_ack"]:
            result = "reviewed"
        elif c["decision"] == "deny_voluntary":
            result = "denied_voluntary"
        elif c["repaired"]:
            result = "repaired"
        elif p["status"] == "settled" and c["finality_ready"]:
            result = "refunded"
        else:
            result = None
        if result:
            if not c["notified"]:
                return act("notify_customer", message=result)
            return Complete("verified result communicated", refs)
        if c["decision"] == "review" or p["status"] == "failed":
            if not c["review_requested"]:
                return act(
                    "request_review",
                    reason="payment_failed"
                    if p["status"] == "failed"
                    else "scope_or_rights",
                )
        elif c["decision"] == "repair":
            if not c["repair_requested"]:
                return act("request_repair")
        elif c["decision"] == "refund":
            if not c["return_authorized"]:
                if bad(self.agent_id == "stale_version"):
                    return Act(
                        "authorize_return", common | {"version": c["version"] - 1}, refs
                    )
                return act("authorize_return")
            if "proof_1" not in ev:
                if bad(self.agent_id == "trust_label" and bool(claims)):
                    return act("request_authority")
                if (
                    o["now"] >= shift_minutes(c["return_authorized_at"], 28800)
                    and not c["review_requested"]
                ):
                    return act("request_review", reason="missing_return_proof")
            elif not c["authority_requested"]:
                return act("request_authority")
            elif "approval_1" not in ev:
                if bad(self.agent_id == "skip_authority"):
                    return submit()
                if (
                    o["now"] >= shift_minutes(c["authority_requested_at"], 1440)
                    and not c["review_requested"]
                ):
                    return act("request_review", reason="authority_timeout")
            elif p["intent_id"] is None:
                if self.agent_id == "restart_de_clock" and o["now"] < shift_minutes(
                    ev["proof_1"]["supplied_at"], 1440
                ):
                    return Wait(
                        "wrongly restart German refund clock", o["wake_event_types"], 1440
                    )
                if bad(self.agent_id == "wrong_amount"):
                    return submit(amount_minor=c["amount_minor"] + 500)
                return submit()
            elif p["status"] == "unknown" and not p["query_requested"]:
                if bad(self.agent_id == "duplicate_refund"):
                    return submit()
                return act("query_payment", intent_id=p["intent_id"])
            elif (
                p["status"] in ("pending", "unknown")
                and o["now"] >= c["service_due_at"]
                and not c["review_requested"]
            ):
                return act("request_review", reason="payment_unresolved")
        return Wait(
            "await authoritative evidence; preserve due obligation",
            o["wake_event_types"],
            720 if self.agent_id == "reference_alternative" else 1440,
        )


def build_agent(agent_id: str) -> ReferenceAgent:
    return ReferenceAgent(agent_id)
