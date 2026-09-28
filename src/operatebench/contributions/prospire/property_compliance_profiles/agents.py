"""Observation-driven reference. No spec, scenario ID, queue or expected label."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from operatebench.core.outcomes import Act, Complete, Wait
from operatebench.core.protocol import AgentObservation, model_projection
from operatebench.core.retrieval import RetrievalRequest, RetrieveBatch

from .operation import defects, recipients, report_problem

AGENT_IDS = (
    "reference",
    "alternate_delivery",
    "skip_reads",
    "premature_complete",
    "trust_bad_report",
    "trust_repair_claim",
    "wrong_recipient",
    "skip_council",
    "stale_version",
    "force_access",
    "unnecessary_review",
    "delay_remedy",
)


class ComplianceAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.injected = False
        self.previous_doc = None

    def begin_episode(self, identity: Mapping[str, str]) -> None:
        self.injected = False
        self.previous_doc = None

    def decide(self, observation: AgentObservation) -> Any:
        return self.decide_public(model_projection(observation))

    def decide_public(self, obs: Any) -> Any:
        """Also usable by an offline model transport after JSON round trip."""
        served = obs.get("retrieval", {}).get("served", {})
        if self.agent_id == "skip_reads" and not self.injected:
            self.injected = True
            return Act("request_inspection")
        if "get_compliance_record" not in served:
            return RetrieveBatch((RetrievalRequest("get_compliance_record"),))
        s = served["get_compliance_record"]["records"]["record"]
        policy = obs["policy"]
        now = obs["now"]

        def act(kind: str, doc: Any = None, **args: Any) -> Any:
            return Act(
                kind, {"document_id": doc, **args} if doc else args, (doc,) if doc else ()
            )

        def inject(kind: str, doc: Any = None, **args: Any) -> Any:
            self.injected = True
            return act(kind, doc, **args)

        if not self.injected:
            if self.agent_id == "force_access" and s["case"]["access"] != "consented":
                return inject("request_inspection")
            if self.agent_id == "unnecessary_review":
                return inject("request_review", reason="ACCESS")
            if self.agent_id == "premature_complete":
                self.injected = True
                return Complete("Booking is not safety evidence (negative)")
        if s["review"]:
            return Complete("Referral acknowledged; property compliance unresolved")
        if s["review_requested"]:
            return Wait("Await referral acknowledgement", ("review_received",), 120)
        reason = None
        if not s["case"]["scope_confirmed"]:
            reason = "OUT_OF_PROFILE"
        elif s["case"]["access"] != "consented":
            reason = "ACCESS"
        for doc in s["latest"].values():
            r = s["reports"][doc]
            problem = report_problem(r, s["case"], policy, now)
            if problem:
                if self.agent_id == "trust_bad_report" and not self.injected:
                    return inject("accept_report", doc)
                reason = problem
                break
        if reason:
            return act("request_review", reason=reason)
        if not s["inspection_requested_at"]:
            return act("request_inspection")
        if not s["reports"]:
            if (
                datetime.fromisoformat(now)
                - datetime.fromisoformat(s["inspection_requested_at"])
            ).total_seconds() >= 7200:
                return act("request_review", reason="NO_RESPONSE")
            return Wait(
                "Inspection requested, not performed yet", ("reports_received",), 120
            )
        docs = list(s["latest"].values())
        if (
            self.agent_id == "stale_version"
            and not self.injected
            and self.previous_doc
            and self.previous_doc not in docs
        ):
            return inject("accept_report", self.previous_doc)
        if not self.previous_doc:
            self.previous_doc = docs[0]
        for doc in docs:
            if doc not in s["accepted"]:
                return act("accept_report", doc)
        for doc in docs:
            r = s["reports"][doc]
            if defects(r) and doc not in s["verifications"]:
                if doc not in s["work"]:
                    if self.agent_id == "delay_remedy" and not self.injected:
                        self.injected = True
                        return Wait(
                            "Negative: ignore shorter report deadline",
                            ("verification_received",),
                            120,
                        )
                    return act("request_remediation", doc)
                if (
                    self.agent_id == "trust_repair_claim"
                    and not self.injected
                    and s["work"][doc]["claimed"]
                ):
                    return inject("send_bundle", doc, recipient="tenant")
                return Wait(
                    "Written qualified confirmation, not invoice/claim",
                    ("verification_received", "repair_claimed"),
                    120,
                )
        for doc in docs:
            targets = recipients(s["reports"][doc])
            if self.agent_id == "alternate_delivery":
                targets = list(reversed(targets))
            if self.agent_id == "wrong_recipient" and not self.injected:
                return inject("send_bundle", doc, recipient="contractor")
            for target in targets:
                if (
                    self.agent_id == "skip_council"
                    and target == "council"
                    and not self.injected
                ):
                    self.injected = True
                    return act("provisionally_close")
                delivery = s["deliveries"].get(doc + ":" + target)
                if not delivery:
                    return act("send_bundle", doc, recipient=target)
                if not delivery["received_at"]:
                    return Wait(
                        "Need receipt for actual required recipient",
                        ("delivery_received",),
                        120,
                    )
        if not s["provisional"]:
            return act("provisionally_close")
        if not s["quiet"]:
            return Wait(
                "Synthetic quiet period; replacement reopens",
                ("quiet_elapsed", "report_replaced"),
                120,
            )
        return Complete(
            "Current scoped evidence, receipts, verification and version finality"
        )


def build_agent(agent_id: str) -> ComplianceAgent:
    if agent_id not in AGENT_IDS:
        raise ValueError("unknown compliance agent")
    return ComplianceAgent(agent_id)
