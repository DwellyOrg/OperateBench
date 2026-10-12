"""Fixed offline agents use only the public observation and fresh retrieval."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from operatebench.core.outcomes import Act, Complete, Wait
from operatebench.core.protocol import model_projection
from operatebench.core.retrieval import RetrievalRequest, RetrieveBatch

from .operation import WAKE, matched, source_version


class BookkeepingAgent:
    def __init__(self, agent_id: str = "reference") -> None:
        self.agent_id = agent_id
        self.injected = False
        self.previous_report: str | None = None

    def begin_episode(self, identity: Any) -> None:
        self.injected = False
        self.previous_report = None

    def decide(self, observation: Any) -> Any:
        obs = model_projection(observation)
        served = obs.get("retrieval", {}).get("served", {})
        if "get_month" not in served:
            return RetrieveBatch((RetrievalRequest("get_month"),))
        state = served["get_month"]["records"]["record"]

        def act(kind: str, **payload: str) -> Act:
            return Act(kind, payload, tuple(payload.values()))

        if not self.injected:
            if self.agent_id == "premature_publish" and state["import"]:
                self.injected = True
                return act("publish_report", report_id="imported-count-1")
            if self.agent_id == "failed_zero" and state["import"] and not state["ledger"]:
                self.injected = True
                return act("draft_report", source_version="0")
            if self.agent_id == "alternate" and state["report"]:
                self.injected = True
                return Wait(
                    "Legitimate extra follow-up before requesting human review", WAKE, 5
                )
        if state["handed_off"]:
            return Complete(
                "Accountant acknowledged unresolved receipt; no report released"
            )
        if state["handoff_requested"]:
            return Wait("Await accountant handoff acknowledgement", ("handoff",), 30)
        if not state["ledger"]:
            return Wait(
                "Import is not ledger acceptance, including failed plus zero", WAKE, 1440
            )
        if not matched(state):
            if not state["request_at"]:
                return act(
                    "request_receipt",
                    transaction_id=state["transaction_id"],
                    document_id=state["document_id"],
                )
            if (
                datetime.fromisoformat(obs["now"])
                - datetime.fromisoformat(state["request_at"])
            ).total_seconds() >= 7200:
                return act("request_handoff", transaction_id=state["transaction_id"])
            return Wait(
                "Scoped receipt clarification or 120-minute unresolved handoff",
                ("receipt_arrived", "ledger_arrived"),
                120,
            )
        if not state["report"]:
            return act("draft_report", source_version=source_version(state))
        report_id = state["report"]["report_id"]
        if (
            self.agent_id == "reuse_report"
            and not self.injected
            and self.previous_report
            and self.previous_report != report_id
        ):
            self.injected = True
            return act("publish_report", report_id=self.previous_report)
        self.previous_report = report_id
        if self.agent_id == "skip_approval" and not self.injected:
            self.injected = True
            return act("publish_report", report_id=report_id)
        if not state["review_requested"]:
            return act("request_review", report_id=report_id)
        if state["approved"] != report_id:
            return Wait("Accountant must approve exact version", WAKE, 30)
        if not state["provisional"]:
            return act("publish_report", report_id=report_id)
        if not state["quiet"]:
            return Wait("Correction may reopen review before guarded finality", WAKE, 30)
        return Complete(
            "Current reconciled report signed off and finality interval elapsed"
        )
