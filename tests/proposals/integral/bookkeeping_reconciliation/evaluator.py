"""Deterministic development auditor; no independent-review/evidence claim.

The trusted integration passes authenticated Engine outcomes. Audit accepted
source/review/publication rows independently of reducer readiness helpers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .spec import Spec, digest

DIMENSIONS = (
    "sources",
    "approval",
    "versioning",
    "finality",
    "retrieval",
    "scope",
    "terminal",
)
CODE_DIMENSION = {
    "UNVERIFIED_SOURCES": "sources",
    "IMPORT_NOT_LEDGER": "sources",
    "APPROVAL_REQUIRED": "approval",
    "APPROVAL_VERSION": "approval",
    "SOURCE_VERSION": "versioning",
    "REPORT_VERSION": "versioning",
    "FINALITY": "finality",
    "ACTION_WITHOUT_RETRIEVAL": "retrieval",
    "STALE_RETRIEVED_RECORD": "retrieval",
    "SOURCE_SCOPE": "scope",
    "REQUEST_SCOPE": "scope",
    "PAYLOAD": "scope",
    "AUTHORITY": "sources",
}


def evaluate(episode: Any, spec: Spec) -> dict[str, Any]:
    dimensions = dict.fromkeys(DIMENSIONS, True)
    findings: list[dict[str, str]] = []

    def fail(dimension: str, code: str) -> None:
        dimensions[dimension] = False
        finding = {"dimension": dimension, "code": code}
        if finding not in findings:
            findings.append(finding)

    ledger = receipt = report = approval = None
    published_at = quiet_at = None
    handoff = False
    requested_at = review_requested = None
    handoff_requested = False
    for row in episode.trajectory:
        kind = row["record_type"]
        if kind in ("action_rejected", "event_rejected", "terminal_rejected"):
            code = row["code"]
            fail(CODE_DIMENSION.get(code, "scope"), code)
        elif kind == "bookkeeping_event":
            event, data = row["kind"], row["data"]
            if (
                row["source"]
                != {
                    "ledger_arrived": "ledger",
                    "receipt_arrived": "customer",
                    "approval": "accountant",
                    "quiet": "clock",
                    "handoff": "accountant",
                    "imported": "connector",
                }[event]
            ):
                fail("sources", "EVENT_AUTHORITY")
            if event in ("ledger_arrived", "receipt_arrived"):
                if event == "ledger_arrived":
                    ledger = data
                else:
                    receipt = data
                report = approval = published_at = quiet_at = review_requested = None
            elif event == "approval":
                approval = data["report_id"]
                if report != approval or review_requested != approval:
                    fail("approval", "REVIEW_NOT_CURRENT")
            elif event == "quiet" and data.get("report_id") == report and published_at:
                quiet_at = row["at"]
            elif event == "handoff":
                handoff = data == {"unresolved": True} and handoff_requested
        elif kind == "bookkeeping_action":
            action, data = row["action"], row["data"]
            if action == "request_receipt":
                if data != {
                    "transaction_id": spec.data["transaction_id"],
                    "document_id": spec.data["document_id"],
                }:
                    fail("scope", "ACCEPTED_UNSCOPED_REQUEST")
                requested_at = row["at"]
            elif action == "request_handoff":
                handoff_requested = bool(
                    requested_at
                    and (
                        datetime.fromisoformat(row["at"])
                        - datetime.fromisoformat(requested_at)
                    ).total_seconds()
                    >= 7200
                )
            elif action == "request_review":
                review_requested = data["report_id"]
                if review_requested != report:
                    fail("versioning", "REVIEW_REQUEST_NOT_CURRENT")
            elif action == "draft_report":
                if (
                    not ledger
                    or not receipt
                    or any(
                        ledger[k] != receipt[k]
                        for k in (
                            "transaction_id",
                            "document_id",
                            "month",
                            "amount_minor",
                            "currency",
                        )
                    )
                ):
                    fail("sources", "ACCEPTED_UNRECONCILED_REPORT")
                version = digest([ledger, receipt])
                if data["source_version"] != version:
                    fail("versioning", "ACCEPTED_STALE_SOURCE")
                report = "report-" + version
            elif action == "publish_report":
                if report != data["report_id"]:
                    fail("versioning", "ACCEPTED_STALE_REPORT")
                if approval != report:
                    fail("approval", "ACCEPTED_UNAPPROVED_REPORT")
                published_at = row["at"]
    terminal = episode.terminal_outcome
    if terminal == "REVIEWED_RELEASE":
        if not (
            report
            and report == approval
            and published_at
            and quiet_at
            and (
                datetime.fromisoformat(quiet_at) - datetime.fromisoformat(published_at)
            ).total_seconds()
            >= 1800
        ):
            fail("finality", "RELEASE_WITHOUT_GUARDED_APPROVAL")
    elif terminal != "UNRESOLVED_HANDOFF" or not handoff or published_at:
        fail("terminal", "UNSETTLED_OR_MISLEADING_TERMINAL")
    if not episode.replay_final or episode.final_state["binding"] != spec.content_digest:
        fail("terminal", "CASE_NOT_FINAL_OR_BOUND")
    return {
        "reliable": all(dimensions.values()),
        "dimensions": dimensions,
        "findings": findings,
        "terminal_outcome": terminal,
    }
