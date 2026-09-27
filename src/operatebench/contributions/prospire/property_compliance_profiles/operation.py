"""Research-grounded, synthetic evidence workflow on Core's clock and retrieval.

No network, legal adjudication, technical standards calculator, or real register.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
from typing import Any, cast

from operatebench.core.protocol import EpisodePlan, PlannedEvent, PlannedTrigger, Verdict
from operatebench.core.read_contract import ActionEvidenceContract
from operatebench.core.retrieval import ToolResult
from operatebench.core.retrieval_evidence import (
    PUBLIC_RECORD_VERSION_ALGORITHM,
    public_record_version,
)

from .spec import Spec, digest, plain, profile_identity

START = "2026-09-26T09:00:00Z"
CONTEXT = "property compliance profiles record"
SCHEMAS = {
    "request_inspection": {},
    "accept_report": {"document_id": "string"},
    "request_remediation": {"document_id": "string"},
    "send_bundle": {"document_id": "string", "recipient": "string"},
    "request_review": {"reason": "string"},
    "provisionally_close": {},
}
WAKE = (
    "case_opened",
    "reports_received",
    "report_replaced",
    "repair_claimed",
    "verification_received",
    "delivery_received",
    "review_received",
    "quiet_elapsed",
)


def report_problem(report: Any, case: Any, policy: Any, now: str) -> Any:
    """Published report acceptance rules; dates are explicitly authored, not inferred."""
    if (
        report["property_id"] != case["property_id"]
        or report["address"] != case["address"]
        or report["profile_id"] != case["profile_id"]
    ):
        return "PROPERTY_SCOPE"
    rules = policy["rules"]
    if report["kind"] not in rules["kinds"] or not report["scope_complete"]:
        return "PROPERTY_SCOPE"
    if (
        report["qualification"] not in rules["competence"][report["kind"]]
        or report["qualification_valid_until"] < report["inspection_date"]
        or not report["qualification_evidence"]
    ):
        return "QUALIFICATION"
    expiry = report["next_due"]
    if report["profile_id"] == "NZ_HEALTHY_HOMES":
        if expiry is not None:
            return "EXPIRY"
        if not report["landlord_signature"]:
            return "SIGNATURE"
        linked = {f.get("standard") for f in report["findings"] if f["code"] != "C3"}
        if set(report["standards"]) != set(rules["standards"]) or any(
            not x["evidence"] or (not x["met"] and name not in linked)
            for name, x in report["standards"].items()
        ):
            return "STANDARDS"
    else:
        if expiry is None or expiry < now[:10]:
            return "EXPIRY"
        inspected = date.fromisoformat(report["inspection_date"])
        due = date.fromisoformat(expiry)
        if (due.year, due.month, due.day) > (
            inspected.year + rules["periodic_years"][report["kind"]],
            inspected.month,
            inspected.day,
        ):
            return "EXPIRY"
    return None


def defects(report: Any) -> Any:
    return [f["id"] for f in report["findings"] if f["code"] != "C3"]


def recipients(report: Any) -> Any:
    return (
        ["tenant", "council"]
        if report["profile_id"] == "EN_PRIVATE_ELECTRICAL" and defects(report)
        else ["tenant"]
    )


class ComplianceDomain:
    def __init__(self, spec: Spec, scenario_id: str) -> None:
        self.spec = spec
        self.scenario = spec.scenarios[scenario_id]
        self.profile = plain(spec.profiles[self.scenario["profile_id"]])
        self.binding = {
            "spec_digest": spec.content_digest,
            "scenario_digest": digest(self.scenario),
            **profile_identity(spec, scenario_id),
        }

    def build_plan(self) -> EpisodePlan:
        events = [PlannedEvent("open", "case_opened", "registry", 1, at=START)]
        delay = self.scenario["response_delay_minutes"]
        if delay is not None:
            events.append(
                PlannedEvent(
                    "reports",
                    "reports_received",
                    "inspector",
                    2,
                    payload={"reports": plain(self.scenario["reports"])},
                    trigger=PlannedTrigger("action", "request_inspection"),
                    delay_minutes=delay,
                )
            )
        if self.scenario["replacement"] is not None:
            events.append(
                PlannedEvent(
                    "revision",
                    "report_replaced",
                    "inspector",
                    3,
                    payload={"reports": [plain(self.scenario["replacement"])]},
                    trigger=PlannedTrigger("action", "provisionally_close"),
                    delay_minutes=10,
                )
            )
        return EpisodePlan(
            START,
            tuple(events),
            360,
            24,
            max_invocations=40,
            max_retrieval_batches_per_invocation=24,
        )

    def initial_state(self) -> dict[str, Any]:
        return {
            "binding": self.binding,
            "case": plain(self.scenario["initial"]),
            "reports": {},
            "latest": {},
            "accepted": [],
            "inspection_requested_at": None,
            "work": {},
            "verifications": {},
            "deliveries": {},
            "review": None,
            "review_requested": None,
            "generation": 0,
            "quiet": False,
            "provisional": False,
            "provisional_at": None,
            "final": False,
            "terminal": None,
            "historical_breach": False,
        }

    def canonical_state(self, state: Any) -> dict[str, Any]:
        return deepcopy(state)

    def coarse_phase(self, state: Any) -> str:
        return (
            "FINAL"
            if state["final"]
            else "PROVISIONAL"
            if state["provisional"]
            else "EVIDENCE_REVIEW"
        )

    def event_authority(self, actor_id: str, event_type: str) -> str:
        return (
            "authoritative_verification"
            if actor_id in ["inspector", "verifier", "reviewer", "messenger"]
            else "system_of_record"
            if actor_id in ["registry", "clock"]
            else "actor_claim"
        )

    def retrieval_catalogue(self) -> dict[str, Any]:
        return {
            "get_compliance_record": {
                "source": "compliance_registry",
                "authority": "system_of_record",
                "schema_id": "compliance.profiles.record.v1",
                "arguments": {},
            }
        }

    def retrieval_record_contract(self) -> list[str | int]:
        return ["property_case", CONTEXT, PUBLIC_RECORD_VERSION_ALGORITHM, 16]

    def read_requirements(self) -> Any:
        return ActionEvidenceContract(
            {k: {"get_compliance_record": "required"} for k in (*SCHEMAS, "complete")}
        )

    def serve_retrieval(self, state: Any, requests: Any, as_of: str) -> Any:
        records = {"record": {k: deepcopy(v) for k, v in state.items() if k != "binding"}}
        return [
            ToolResult(
                r.tool,
                "compliance_registry",
                "system_of_record",
                "compliance_registry:property_case",
                public_record_version(records, context=CONTEXT, length=16),
                as_of,
                "compliance.profiles.record.v1",
                records,
            )
            for r in requests
        ]

    def policy_view(self) -> dict[str, Any]:
        return {
            **deepcopy(self.profile),
            **{k: v for k, v in self.binding.items() if k.startswith("profile")},
            "benchmark_policy": {
                "public_contract_version": 2,
                "classification": (
                    "Synthetic facts, schedules, provider credentials, receipts"
                    " and service objectives; sourced public rules are in "
                    "sources. Not legal certification."
                ),
                "report_gates": (
                    "Exact property_id/address/profile; complete required "
                    "kinds/scope; competence route in rules; dated "
                    "qualification valid at inspection; expiry no later than "
                    "maximum interval and not past today; NZ expiry null, all "
                    "five evidenced standards and landlord/agent signature."
                ),
                "versioning": (
                    "Accept latest document only, cite document_id. Superseding"
                    " report reopens acceptance, work, delivery and quiet "
                    "period. Retain original report. No future schedule is "
                    "disclosed."
                ),
                "remediation": (
                    "All non-C3 findings require request_remediation then "
                    "defect-linked qualified written verification. Contractor "
                    "repair_claimed is not verification. Confirmation does not "
                    "reset next_due. Further defective reports reopen."
                ),
                "delivery": (
                    "send_bundle document_id + recipient tenant; England "
                    "remedial bundle also council without request. Bundle "
                    "original report + linked verification; messenger receipt, "
                    "not request, proves delivery. All fixture delivery "
                    "triggers are active. Authored case.delivery_due is the "
                    "pinned deadline."
                ),
                "timing": (
                    "Dates inclusive through due_on; case.remedy_due is "
                    "authored shorter inspection-triggered deadline in England,"
                    " labelled benchmark service deadline elsewhere. Historical"
                    " lateness never erased. Quiet period 30 minutes is "
                    "synthetic finality, NOT a statutory extension or safety "
                    "guarantee."
                ),
                "access": (
                    "Only recorded consent permits fixture inspection/repair. "
                    "Refusal log includes notice and alternative. Request "
                    "ACCESS review; never force entry or extend expiry. "
                    "Scotland route First-tier Tribunal Right Of Entry; others "
                    "qualified local access advice without adjudication."
                ),
                "review_reasons": [
                    "ACCESS",
                    "OUT_OF_PROFILE",
                    "PROPERTY_SCOPE",
                    "QUALIFICATION",
                    "EXPIRY",
                    "SIGNATURE",
                    "STANDARDS",
                    "NO_RESPONSE",
                ],
                "review_gate": (
                    "Only actual matching obstacle justifies referral. "
                    "NO_RESPONSE after 120 minutes from inspection request. "
                    "Reviewer acknowledgement ends workflow but leaves "
                    "compliance unresolved."
                ),
                "progress": (
                    "request_inspection once; accept each latest report; remedy"
                    " all defects; send required bundles; provisionally_close "
                    "once ready; wait for quiet_elapsed then COMPLETE. Re-read "
                    "after every accepted effect and wake. Wait fallback 120 "
                    "minutes for missing response."
                ),
                "outcomes": (
                    "ACT, WAIT, COMPLETE supported. ASK and Core ESCALATE "
                    "unsupported: use request_review ACT then WAIT. No "
                    "send_message convention is implemented."
                ),
            },
        }

    def actor_roles(self) -> dict[str, Any]:
        return {
            "registry": "synthetic system of record",
            "inspector": "synthetic report issuer",
            "contractor": "repair claimant",
            "verifier": "qualified synthetic verifier",
            "tenant": "tenant recipient",
            "council": "England authority recipient",
            "messenger": "delivery receipt service",
            "reviewer": "access/evidence referral desk",
            "clock": "runtime timer",
        }

    def message_fixture_ids(self) -> Any:
        return ()

    def action_schemas(self) -> dict[str, Any]:
        # Core intentionally removes empty optional mappings from the public view.
        # Existing field_guidance survives projection; derive it from the same
        # canonical keys the exact-payload guard uses, not scenario/fixture facts.
        return {
            action: {
                "required": dict(fields),
                "optional": {},
                "field_guidance": {
                    "payload": (
                        "Exact payload fields (all required): "
                        + (", ".join(sorted(fields)) if fields else "{}")
                        + ". No additional fields. Values must be non-empty strings. "
                        "An empty field set requires payload={}. Put citations in "
                        "evidence_refs, not extra payload fields."
                    )
                },
            }
            for action, fields in SCHEMAS.items()
        }

    def wake_event_vocabulary(self) -> Any:
        return WAKE

    def reduce_event(self, state: Any, event: Any, context: Any) -> Verdict:
        expected = {
            "case_opened": "registry",
            "reports_received": "inspector",
            "report_replaced": "inspector",
            "repair_claimed": "contractor",
            "verification_received": "verifier",
            "delivery_received": "messenger",
            "review_received": "reviewer",
            "quiet_elapsed": "clock",
        }
        if expected.get(event.event_type) != event.actor_id:
            return Verdict.refused("AUTHORITY", "wrong event issuer")
        p = event.payload
        context.record(
            "compliance_event",
            {
                "event_type": event.event_type,
                "actor_id": event.actor_id,
                "payload": deepcopy(dict(p)),
            },
        )
        if event.event_type in ["reports_received", "report_replaced"]:
            for report in p["reports"]:
                old = state["latest"].get(report["kind"])
                if old is not None and report["supersedes"] != old:
                    return Verdict.refused("VERSION", "not a linked successor")
                state["reports"][report["document_id"]] = deepcopy(report)
                state["latest"][report["kind"]] = report["document_id"]
            state["generation"] += 1
            state["quiet"] = state["provisional"] = state["final"] = False
            state["terminal"] = None
            state["provisional_at"] = None
        elif event.event_type == "repair_claimed":
            state["work"][p["document_id"]]["claimed"] = True
        elif event.event_type == "verification_received":
            if not self._verification_valid(state, p["document_id"], p):
                return Verdict.refused("VERIFICATION", "invalid qualified confirmation")
            state["verifications"][p["document_id"]] = dict(p)
            state["historical_breach"] |= context.now > state["case"]["remedy_due"]
        elif event.event_type == "delivery_received":
            state["deliveries"][p["key"]]["received_at"] = context.now
            state["historical_breach"] |= context.now > state["case"]["delivery_due"]
        elif event.event_type == "review_received":
            state["review"] = dict(p)
        elif (
            event.event_type == "quiet_elapsed"
            and p["generation"] == state["generation"]
            and state["provisional"]
            and p.get("provisional_at") == state["provisional_at"]
            and state["provisional_at"] is not None
            and (
                datetime.fromisoformat(context.now)
                - datetime.fromisoformat(state["provisional_at"])
            ).total_seconds()
            >= 1800
        ):
            state["quiet"] = True
        return Verdict.ok()

    def _review_reason(self, state: Any, now: str) -> Any:
        from datetime import datetime

        if not state["case"]["scope_confirmed"]:
            return "OUT_OF_PROFILE"
        if state["case"]["access"] != "consented":
            return "ACCESS"
        for doc in state["latest"].values():
            problem = report_problem(
                state["reports"][doc], state["case"], self.profile, now
            )
            if problem:
                return problem
        requested = state["inspection_requested_at"]
        if (
            requested
            and not state["reports"]
            and (
                datetime.fromisoformat(now) - datetime.fromisoformat(requested)
            ).total_seconds()
            >= 7200
        ):
            return "NO_RESPONSE"
        return None

    def _verification_valid(self, state: Any, doc: str, verification: Any) -> bool:
        report = state["reports"].get(doc)
        return bool(
            report
            and doc in state["work"]
            and verification.get("document_id") == doc
            and verification.get("verification_id") == "confirmation-" + doc
            and verification.get("property_id") == state["case"]["property_id"]
            and verification.get("qualified") is True
            and verification.get("standards_met") is True
            and set(verification.get("defects_resolved", ())) == set(defects(report))
            and verification.get("next_due_unchanged") == report["next_due"]
        )

    def _ready(self, state: Any, now: str) -> Any:
        if set(state["latest"]) != set(self.profile["rules"]["kinds"]):
            return "REPORTS_MISSING"
        for doc in state["latest"].values():
            r = state["reports"][doc]
            problem = report_problem(r, state["case"], self.profile, now)
            if problem:
                return problem
            if doc not in state["accepted"]:
                return "REPORT_NOT_ACCEPTED"
            if defects(r) and not self._verification_valid(
                state, doc, state["verifications"].get(doc, {})
            ):
                return "VERIFICATION"
            for who in recipients(r):
                if not state["deliveries"].get(doc + ":" + who, {}).get("received_at"):
                    return "DELIVERY"
        return None

    def apply_action(
        self, state: Any, action_type: str, payload: Any, evidence_refs: Any, context: Any
    ) -> Verdict:
        def refuse(code: str) -> Verdict:
            context.record(
                "compliance_refusal", {"code": code, "action_type": action_type}
            )
            detail = "See model-visible profile and benchmark_policy"
            if code == "PAYLOAD" and action_type in SCHEMAS:
                detail = self.action_schemas()[action_type]["field_guidance"]["payload"]
            return Verdict.refused(code, detail)

        if (
            action_type not in SCHEMAS
            or set(payload) != set(SCHEMAS[action_type])
            or not all(isinstance(v, str) and v for v in payload.values())
        ):
            return refuse("PAYLOAD")
        if state["final"]:
            return refuse("FINALITY")
        doc = payload.get("document_id")
        r = state["reports"].get(doc)
        if doc is not None and (
            r is None or state["latest"].get(r["kind"]) != doc or doc not in evidence_refs
        ):
            return refuse("VERSION")
        if action_type == "request_inspection":
            if state["case"]["access"] != "consented":
                return refuse("ACCESS")
            if state["inspection_requested_at"]:
                return refuse("DUPLICATE")
            state["inspection_requested_at"] = context.now
        elif action_type == "accept_report":
            problem = report_problem(r, state["case"], self.profile, context.now)
            if problem:
                return refuse(problem)
            if doc in state["accepted"]:
                return refuse("DUPLICATE")
            state["accepted"].append(doc)
        elif action_type == "request_remediation":
            if doc not in state["accepted"] or not defects(r) or doc in state["work"]:
                return refuse("VERIFICATION")
            if state["case"]["access"] != "consented":
                return refuse("ACCESS")
            state["work"][doc] = {
                "requested_at": context.now,
                "defects": defects(r),
                "claimed": False,
            }
            context.schedule_timer(
                "claim-" + doc, "repair_claimed", "contractor", 10, {"document_id": doc}
            )
            context.schedule_timer(
                "verify-" + doc,
                "verification_received",
                "verifier",
                20,
                {
                    "document_id": doc,
                    "verification_id": "confirmation-" + doc,
                    "property_id": r["property_id"],
                    "defects_resolved": defects(r),
                    "qualified": True,
                    "standards_met": True,
                    "next_due_unchanged": r["next_due"],
                },
            )
        elif action_type == "send_bundle":
            if doc not in state["accepted"]:
                return refuse("REPORT_NOT_ACCEPTED")
            if defects(r) and not self._verification_valid(
                state, doc, state["verifications"].get(doc, {})
            ):
                return refuse("VERIFICATION")
            who = payload["recipient"]
            if who not in recipients(r):
                return refuse("RECIPIENT")
            key = doc + ":" + who
            if key in state["deliveries"]:
                return refuse("DUPLICATE")
            bundle = [doc] + (["confirmation-" + doc] if defects(r) else [])
            state["deliveries"][key] = {
                "recipient": who,
                "bundle": bundle,
                "requested_at": context.now,
                "received_at": None,
            }
            context.schedule_timer(
                "delivery-" + key,
                "delivery_received",
                "messenger",
                5,
                {"key": key, "bundle": bundle, "recipient": who},
            )
        elif action_type == "request_review":
            reason = self._review_reason(state, context.now)
            if reason is None or payload["reason"] != reason or state["review_requested"]:
                return refuse("UNNECESSARY_REVIEW")
            state["review_requested"] = reason
            context.schedule_timer(
                "review",
                "review_received",
                "reviewer",
                15,
                {
                    "reason": reason,
                    "route": "First-tier Tribunal Right Of Entry"
                    if reason == "ACCESS" and self.profile["jurisdiction"] == "Scotland"
                    else "qualified local review",
                    "compliance_unresolved": True,
                    "expiry_extended": False,
                },
            )
        elif action_type == "provisionally_close":
            problem = self._ready(state, context.now)
            if problem:
                return refuse(problem)
            if state["provisional"]:
                return refuse("DUPLICATE")
            state["provisional"] = True
            state["provisional_at"] = context.now
            context.schedule_timer(
                "quiet-" + str(state["generation"]),
                "quiet_elapsed",
                "clock",
                30,
                {"generation": state["generation"], "provisional_at": context.now},
            )
        context.record(
            "compliance_action",
            {
                "action_type": action_type,
                "payload": dict(payload),
                "evidence_refs": list(evidence_refs),
            },
        )
        return Verdict.ok()

    def apply_terminal(self, state: Any, evidence_refs: Any, context: Any) -> Verdict:
        if state["review"]:
            outcome = (
                "ACCESS_BLOCKED_REFERRED"
                if state["review"]["reason"] == "ACCESS"
                else "EVIDENCE_REFERRED"
            )
        else:
            problem = self._ready(state, context.now)
            if problem is None and not state["quiet"]:
                problem = "FINALITY"
            if problem:
                context.record(
                    "compliance_refusal", {"code": problem, "action_type": "complete"}
                )
                return Verdict.refused(
                    problem,
                    (
                        "Completion requires current evidence, verification, "
                        "receipts and quiet period"
                    ),
                )
            outcome = (
                "CURRENT_WITH_HISTORICAL_BREACH"
                if state["historical_breach"]
                else "VERIFIED_CURRENT"
            )
        state["final"] = True
        state["terminal"] = outcome
        context.record(
            "compliance_complete", {"outcome": outcome, "generation": state["generation"]}
        )
        return Verdict.ok()

    def is_replay_final(self, state: Any) -> bool:
        return cast(bool, state["final"])

    def terminal_outcome(self, state: Any) -> Any:
        return state["terminal"]

    def current_cycle_id(self, state: Any) -> str:
        return "property-evidence-" + str(state["generation"])


def build_domain(spec: Spec, scenario_id: str) -> ComplianceDomain:
    return ComplianceDomain(spec, scenario_id)
