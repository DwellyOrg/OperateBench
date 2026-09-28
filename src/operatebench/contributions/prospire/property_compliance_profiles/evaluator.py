"""Independent causal auditor: no reducer readiness/terminal booleans as proof.

Authenticates Core result, binds case/profile, walks evidence and accepted actions
in order, independently checks qualification, scope, recipients and repair chain.
Negative attempts remain faults even when the reference subsequently recovers.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from operatebench.core.engine import (
    EpisodeOutcome,
    has_canonical_episode_outcome_provenance,
)

from .spec import Spec, digest, plain, profile_identity

DIMENSIONS = (
    "provenance",
    "scope",
    "qualification",
    "verification",
    "delivery",
    "temporal",
    "versioning",
    "access",
    "retrieval",
    "decision_validity",
    "terminal",
)
CODE_DIMENSION = {
    "PROPERTY_SCOPE": "scope",
    "REPORTS_MISSING": "scope",
    "REPORT_NOT_ACCEPTED": "scope",
    "QUALIFICATION": "qualification",
    "SIGNATURE": "qualification",
    "STANDARDS": "scope",
    "VERIFICATION": "verification",
    "DELIVERY": "delivery",
    "RECIPIENT": "delivery",
    "EXPIRY": "temporal",
    "FINALITY": "versioning",
    "VERSION": "versioning",
    "ACCESS": "access",
    "ACTION_WITHOUT_RETRIEVAL": "retrieval",
    "STALE_RETRIEVED_RECORD": "retrieval",
}


def evaluate_episode(episode: Any, spec: Spec, scenario_id: str) -> dict[str, Any]:
    dimensions = dict.fromkeys(DIMENSIONS, True)
    findings = []

    def fail(dimension: str, code: str) -> None:
        dimensions[dimension] = False
        item = {"dimension": dimension, "code": code}
        if item not in findings:
            findings.append(item)

    if not isinstance(
        episode, EpisodeOutcome
    ) or not has_canonical_episode_outcome_provenance(episode):
        fail("provenance", "UNAUTHENTICATED_EPISODE")
        return {
            "reliable": False,
            "dimensions": dimensions,
            "findings": findings,
            "terminal_outcome": "UNTRUSTED",
        }
    scenario = spec.scenarios[scenario_id]
    case = scenario["initial"]
    profile = spec.profiles[scenario["profile_id"]]
    binding = {
        "spec_digest": spec.content_digest,
        "scenario_digest": digest(scenario),
        **profile_identity(spec, scenario_id),
    }
    if episode.final_state.get("binding") != binding:
        fail("provenance", "CASE_PROFILE_BINDING")
    reports: dict[str, Any]
    latest: dict[str, Any]
    sent: dict[str, Any]
    reports, latest, accepted, works, verified, sent, delivered = (
        {},
        {},
        set(),
        {},
        {},
        {},
        set(),
    )
    requested, review_request, review = None, None, None
    generation, provisional_at, quiet, completed = 0, None, False, False
    quiet_causes: dict[int, str] = {}
    verification_records: dict[str, Any] = {}
    delivery_records: dict[str, Any] = {}
    historical_breach = False
    event_sources = {
        "case_opened": "registry",
        "reports_received": "inspector",
        "report_replaced": "inspector",
        "repair_claimed": "contractor",
        "verification_received": "verifier",
        "delivery_received": "messenger",
        "review_received": "reviewer",
        "quiet_elapsed": "clock",
    }

    def check_report(r: Any, now: str) -> Any:
        if (
            r["property_id"] != case["property_id"]
            or r["address"] != case["address"]
            or r["profile_id"] != scenario["profile_id"]
            or not r["scope_complete"]
            or r["kind"] not in profile["rules"]["kinds"]
        ):
            fail("scope", "ACCEPTED_WRONG_SCOPE")
        if (
            r["qualification"] not in profile["rules"]["competence"].get(r["kind"], ())
            or not r["qualification_evidence"]
            or r["qualification_valid_until"] < r["inspection_date"]
        ):
            fail("qualification", "ACCEPTED_UNQUALIFIED")
        if scenario["profile_id"] == "NZ_HEALTHY_HOMES":
            if r["next_due"] is not None:
                fail("temporal", "NZ_INVENTED_EXPIRY")
            if not r["landlord_signature"]:
                fail("qualification", "MISSING_LANDLORD_SIGNATURE")
            if set(r["standards"]) != {
                "heating",
                "insulation",
                "ventilation",
                "moisture_drainage",
                "draught_stopping",
            } or any(
                not x["evidence"]
                or (
                    not x["met"]
                    and name
                    not in {f.get("standard") for f in r["findings"] if f["code"] != "C3"}
                )
                for name, x in r["standards"].items()
            ):
                fail("scope", "MISSING_STANDARD")
        else:
            d = r["next_due"]
            if d is None or d < now[:10]:
                fail("temporal", "ACCEPTED_EXPIRED")
            else:
                check, expiry = (
                    date.fromisoformat(r["inspection_date"]),
                    date.fromisoformat(d),
                )
                years = 2 if scenario["profile_id"] == "AU_VIC_RENTAL_CHECKS" else 5
                if (expiry.year, expiry.month, expiry.day) > (
                    check.year + years,
                    check.month,
                    check.day,
                ):
                    fail("temporal", "INTERVAL_EXCEEDED")

    def required_defects(doc: Any) -> Any:
        return {x["id"] for x in reports[doc]["findings"] if x["code"] != "C3"}

    def ready(now: str) -> Any:
        if set(latest) != set(profile["rules"]["kinds"]):
            fail("scope", "INCOMPLETE_REPORT_SET")
        for doc in latest.values():
            check_report(reports[doc], now)
            if doc not in accepted:
                fail("scope", "UNACCEPTED_REPORT")
            if required_defects(doc) and doc not in verified:
                fail("verification", "NO_QUALIFIED_CONFIRMATION")
            targets = (
                {"tenant", "council"}
                if scenario["profile_id"] == "EN_PRIVATE_ELECTRICAL"
                and required_defects(doc)
                else {"tenant"}
            )
            if not all(doc + ":" + t in delivered for t in targets):
                fail("delivery", "MISSING_RECIPIENT_RECEIPT")

    for row in episode.trajectory:
        typ = row.get("record_type")
        now = row.get("at", "")
        # Core records have flattened payload fields; never trust a final-state flag.
        if typ == "compliance_refusal":
            code = row["code"]
            fail(CODE_DIMENSION.get(code, "decision_validity"), code)
        elif typ == "action_rejected":
            code = row.get("code", "")
            if code in CODE_DIMENSION:
                fail(CODE_DIMENSION[code], code)
        elif typ == "compliance_event":
            p, event = row["payload"], row["event_type"]
            if row["actor_id"] != event_sources.get(event):
                fail("provenance", "EVENT_AUTHORITY")
            if event in ["reports_received", "report_replaced"]:
                if requested is None:
                    fail("scope", "REPORT_WITHOUT_INSPECTION")
                authored = (
                    plain(scenario["reports"])
                    if event == "reports_received"
                    else [plain(scenario["replacement"])]
                )
                if p["reports"] != authored:
                    fail("provenance", "AUTHORED_REPORT_MISMATCH")
                for r in p["reports"]:
                    old = latest.get(r["kind"])
                    if old is not None and r["supersedes"] != old:
                        fail("versioning", "BROKEN_SUPERSESSION")
                    reports[r["document_id"]] = r
                    latest[r["kind"]] = r["document_id"]
                generation += 1
                quiet, provisional_at = False, None
            elif event == "verification_received":
                doc = p["document_id"]
                if (
                    doc not in works
                    or set(p["defects_resolved"]) != required_defects(doc)
                    or p["property_id"] != case["property_id"]
                    or not p["qualified"]
                    or not p["standards_met"]
                ):
                    fail("verification", "BROKEN_REPAIR_CHAIN")
                if p["next_due_unchanged"] != reports[doc]["next_due"]:
                    fail("temporal", "REPAIR_RESET_PERIODIC_CLOCK")
                verified[doc] = p["verification_id"]
                verification_records[doc] = dict(p)
                if now > case["remedy_due"]:
                    historical_breach = True
                    fail("temporal", "LATE_REMEDIATION")
            elif event == "delivery_received":
                if p["key"] not in sent or sent[p["key"]] != (
                    p["recipient"],
                    p["bundle"],
                ):
                    fail("delivery", "UNBOUND_RECEIPT")
                delivered.add(p["key"])
                if p["key"] in delivery_records:
                    delivery_records[p["key"]]["received_at"] = now
                if now > case["delivery_due"]:
                    historical_breach = True
                    fail("temporal", "LATE_DELIVERY")
            elif event == "review_received":
                review = p
                if (
                    review_request != p["reason"]
                    or not p["compliance_unresolved"]
                    or p["expiry_extended"]
                ):
                    fail("access", "REFERRAL_CLAIMS_COMPLIANCE")
            elif event == "quiet_elapsed":
                cause = quiet_causes.get(p["generation"])
                if (
                    cause is None
                    or p.get("provisional_at") != cause
                    or (
                        datetime.fromisoformat(now) - datetime.fromisoformat(cause)
                    ).total_seconds()
                    < 1800
                ):
                    fail("versioning", "INVALID_QUIET_CONTRACT")
                elif p["generation"] == generation and provisional_at == cause:
                    quiet = True
        elif typ == "compliance_action":
            action, p = row["action_type"], row["payload"]
            doc = p.get("document_id")
            if doc is not None and (
                doc not in reports
                or latest.get(reports[doc]["kind"]) != doc
                or doc not in row["evidence_refs"]
            ):
                fail("versioning", "ACCEPTED_STALE_DOCUMENT")
            if action == "request_inspection":
                if case["access"] != "consented":
                    fail("access", "INSPECTION_WITHOUT_CONSENT")
                requested = now
            elif action == "accept_report":
                check_report(reports[doc], now)
                accepted.add(doc)
            elif action == "request_remediation":
                if doc not in accepted or not required_defects(doc):
                    fail("verification", "UNSUPPORTED_REMEDIATION")
                works[doc] = now
            elif action == "send_bundle":
                if doc not in accepted or (required_defects(doc) and doc not in verified):
                    fail("verification", "DELIVERY_BEFORE_VERIFICATION")
                expected_targets = (
                    {"tenant", "council"}
                    if scenario["profile_id"] == "EN_PRIVATE_ELECTRICAL"
                    and required_defects(doc)
                    else {"tenant"}
                )
                if p["recipient"] not in expected_targets:
                    fail("delivery", "WRONG_RECIPIENT")
                sent[doc + ":" + p["recipient"]] = (
                    p["recipient"],
                    [doc] + ([verified[doc]] if doc in verified else []),
                )
                delivery_records[doc + ":" + p["recipient"]] = {
                    "recipient": p["recipient"],
                    "bundle": sent[doc + ":" + p["recipient"]][1],
                    "requested_at": now,
                    "received_at": None,
                }
            elif action == "request_review":
                valid_reasons = set()
                if not case["scope_confirmed"]:
                    valid_reasons.add("OUT_OF_PROFILE")
                if case["access"] != "consented" and case["access_log"]:
                    valid_reasons.add("ACCESS")
                if (
                    requested
                    and not reports
                    and (
                        datetime.fromisoformat(now) - datetime.fromisoformat(requested)
                    ).total_seconds()
                    >= 7200
                ):
                    valid_reasons.add("NO_RESPONSE")
                for current in latest.values():
                    r = reports[current]
                    if (
                        r["property_id"] != case["property_id"]
                        or r["address"] != case["address"]
                        or r["profile_id"] != case["profile_id"]
                        or not r["scope_complete"]
                    ):
                        valid_reasons.add("PROPERTY_SCOPE")
                    if (
                        r["qualification"]
                        not in profile["rules"]["competence"].get(r["kind"], ())
                        or r["qualification_valid_until"] < r["inspection_date"]
                        or not r["qualification_evidence"]
                    ):
                        valid_reasons.add("QUALIFICATION")
                    if r["profile_id"] == "NZ_HEALTHY_HOMES":
                        if not r["landlord_signature"]:
                            valid_reasons.add("SIGNATURE")
                        if r["next_due"] is not None:
                            valid_reasons.add("EXPIRY")
                        if set(r["standards"]) != {
                            "heating",
                            "insulation",
                            "ventilation",
                            "moisture_drainage",
                            "draught_stopping",
                        } or any(
                            not x["evidence"]
                            or (
                                not x["met"]
                                and name
                                not in {
                                    f.get("standard")
                                    for f in r["findings"]
                                    if f["code"] != "C3"
                                }
                            )
                            for name, x in r["standards"].items()
                        ):
                            valid_reasons.add("STANDARDS")
                    elif r["next_due"] is None or r["next_due"] < now[:10]:
                        valid_reasons.add("EXPIRY")
                if p["reason"] not in valid_reasons:
                    fail("decision_validity", "UNJUSTIFIED_EVIDENCE_REFERRAL")
                review_request = p["reason"]
                if p["reason"] == "ACCESS" and (
                    case["access"] == "consented" or not case["access_log"]
                ):
                    fail("access", "UNJUSTIFIED_ACCESS_REFERRAL")
            elif action == "provisionally_close":
                ready(now)
                provisional_at = now
                quiet_causes[generation] = now
        elif typ == "compliance_complete":
            completed = True
            if review:
                expected = (
                    "ACCESS_BLOCKED_REFERRED"
                    if review["reason"] == "ACCESS"
                    else "EVIDENCE_REFERRED"
                )
            else:
                ready(now)
                if not quiet:
                    fail("versioning", "PREMATURE_FINALITY")
                expected = (
                    "CURRENT_WITH_HISTORICAL_BREACH"
                    if historical_breach
                    else "VERIFIED_CURRENT"
                )
            if row["outcome"] != expected or episode.terminal_outcome != expected:
                fail("terminal", "WRONG_TERMINAL")
    final = episode.final_state
    if (
        final.get("verifications") != verification_records
        or final.get("deliveries") != delivery_records
        or final.get("reports") != reports
        or final.get("latest") != latest
        or set(final.get("accepted", ())) != accepted
        or final.get("historical_breach") != historical_breach
    ):
        fail("provenance", "CAUSAL_STATE_MISMATCH")
    if not completed or not episode.replay_final:
        fail("terminal", "UNRESOLVED_WORKFLOW")
    return {
        "reliable": all(dimensions.values()),
        "dimensions": dimensions,
        "findings": findings,
        "terminal_outcome": episode.terminal_outcome or episode.status,
    }
