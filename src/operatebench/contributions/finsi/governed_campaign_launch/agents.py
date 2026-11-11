"""Agents operate on published JSON observations only; no fixture access."""

from __future__ import annotations

from typing import Any

from operatebench.core.outcomes import Act, Complete, Wait
from operatebench.core.protocol import model_projection
from operatebench.core.retrieval import RetrievalRequest, RetrieveBatch

from .operation import binding, plus

AGENT_IDS = (
    "reference",
    "alternate",
    "counts_only",
    "chat_approval",
    "legacy_bypass",
    "unnecessary_approval",
    "unverified",
    "new_key_retry",
    "premature_close",
)


class CampaignAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.injected = False

    def begin_episode(self, identity: Any) -> None:
        self.injected = False

    def decide(self, observation: Any) -> Any:
        return self.decide_public(model_projection(observation))

    def decide_public(self, obs: Any) -> Any:
        served = obs["retrieval"].get("served", {})
        if "get_campaign" not in served:
            return RetrieveBatch((RetrievalRequest("get_campaign"),))
        s = served["get_campaign"]["records"]["record"]
        now = obs["now"]

        def act(k: str, **p: Any) -> Any:
            return Act(k, p)

        def inject(k: str, **p: Any) -> Any:
            self.injected = True
            return act(k, **p)

        def wait(minutes: int = 30) -> Any:
            return Wait(
                "Await external authoritative change; bounded follow-up",
                tuple(obs["wake_event_types"]),
                minutes,
            )

        if s["transfer"] or s["cancelled"]:
            return (
                Complete("Registry checked; ownership or cancellation settled")
                if not s["status_owed"]
                else act("status")
            )
        if s["transfer_requested"]:
            return wait()
        if s["cancellation_authorised"] and not s["cancelled"]:
            return act("cancel")
        if any(
            not r["registry_version"] or not r["accepted_commitment"] for r in s["ledger"]
        ):
            return act("request_transfer")
        if not self.injected:
            if self.agent_id == "premature_close":
                self.injected = True
                return Complete("All done", ("made-up-citation",))
            if self.agent_id == "legacy_bypass":
                return inject("legacy_send")
            if self.agent_id == "chat_approval" and s["chat"] and not s["approval"]:
                return inject("schedule", key="chat-approved")
            if self.agent_id == "unverified" and s["stored"] and not s["verified"]:
                return inject("request_approval", role="send_approver")
        if s["status_owed"]:
            return act("status")
        if (
            s["approval"]
            and s["approval"]["active"]
            and s["catalogue_version"] != s["approval"]["catalogue_version"]
        ):
            return act("withdraw")
        if s["withdrawn"]:
            if s["catalogue_version"] > s["version"]:
                return act("draft")
            if (
                s["approval"]
                and s["approval"]["active"]
                and s["approval"]["recorded_at"] > s["window"][1]
            ):
                return act("withdraw")
            return Complete("Withdrawn; registry has no send and all schedules cancelled")
        if now >= s["window"][1] and not s["ledger"]:
            return act("withdraw")
        if not s["draft"]:
            return act("draft")
        if not s["snapshot"] and self.agent_id == "alternate":
            return act("snapshot")
        if not s["pushed_at"] or not s["pushes"].get(str(s["version"])):
            return act("push")
        if not s["stored"]:
            if now >= plus(s["pushed_at"], 30):
                return (
                    act("push")
                    if s["pushes"][str(s["version"])] < 2
                    else act("request_transfer")
                )
            return wait()
        if s["stored"] != s["draft"]:
            if not s["verification_failed"]:
                return act("verify")
            return act("push") if s["pushes"][str(s["version"])] < 2 else act("withdraw")
        if not s["verified"]:
            return act("verify")
        if not s["snapshot"]:
            return act("snapshot")
        if not s["request"]:
            role = (
                "send_approver"
                if len(s["snapshot"]["members"]) > 500 or s["draft"]["discount"] > 15
                else "brief_owner"
            )
            if (
                self.agent_id == "unnecessary_approval"
                and role == "brief_owner"
                and not self.injected
            ):
                return inject("request_approval", role="send_approver")
            return act("request_approval", role=role)
        if not s["approval"]:
            if now >= "2026-10-05T06:00:00Z" and "final_notice" not in s["reminders"]:
                return act("final_notice")
            if now >= "2026-10-04T15:00:00Z" and "remind" not in s["reminders"]:
                return act("remind")
            return wait(360)
        if any(s["approval"][k] != binding(s)[k] for k in binding(s)):
            return act("withdraw")
        if not s["consent_enforced"]:
            return act("withdraw")
        key = s["scheduling_key"]
        if s["timeout"]:
            if not s["idempotent"]:
                return act("request_transfer")
            if self.agent_id == "new_key_retry" and not self.injected:
                return inject("schedule", key="fresh-key")
            return act("schedule", key=key)
        if key not in s["schedules"]:
            return act("schedule", key=key)
        if not s["ledger"]:
            return wait(360)
        if not s["reconciled"]:
            if self.agent_id == "counts_only" and not self.injected:
                return inject("reconcile_counts")
            return act("reconcile")
        if now < s["cooldown_until"]:
            return wait(1440)
        return Complete("Identity reconciliation and cooldown evidenced")


def build_agent(agent_id: str) -> CampaignAgent:
    if agent_id not in AGENT_IDS:
        raise ValueError("unknown agent")
    return CampaignAgent(agent_id)
