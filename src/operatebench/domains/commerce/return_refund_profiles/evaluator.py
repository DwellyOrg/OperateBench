"""Independent causal checks over fresh Engine evidence, not reducer success bits.

Does not import the reducer, entitlement helper, agent or expected case labels.
Fixture facts establish the branch; accepted proposals and authoritative external
records establish whether the branch was actually performed. Refused unsafe
proposals remain failures even if the agent subsequently recovers.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from operatebench.core.clock import shift_minutes
from operatebench.core.engine import (
    EpisodeOutcome,
    has_canonical_episode_outcome_provenance,
)
from operatebench.core.errors import MalformedTimestampError

from .spec import Spec, digest, profile_identity, thaw

EVALUATOR_VERSION = "0.3.0"

DIMENSIONS = (
    "provenance",
    "profile_scope",
    "entitlement",
    "proof_chain",
    "authority",
    "settlement",
    "deduplication",
    "clock",
    "completion",
    "safe_actions",
    "read_freshness",
    "causal_integrity",
)


def _branch(s: Any) -> Any:
    if (
        not s["ordinary_goods"]
        or not s["jurisdiction_stipulated"]
        or s["transaction_proof"] == "unknown"
    ):
        return "review"
    if s["reason"] == "fault":
        choices = {
            ("major", "refund"): "refund",
            ("minor", "refund"): "repair",
            ("minor", "repair"): "repair",
            ("minor", "replacement"): "repair",
        }
        return (
            choices.get((s["fault"], s["consumer_choice"]), "review")
            if s["profile_id"] == "AU_VIC" and s["rejection_timely_stipulated"]
            else "review"
        )
    if s["profile_id"] in ("UK", "DE"):
        last_notice = date.fromisoformat(s["delivery_date"]) + timedelta(days=14)
        return (
            "refund"
            if s["disclosed"]
            and s["channel"] == "distance"
            and date.fromisoformat(s["notice_sent_date"]) <= last_notice
            else "review"
        )
    if not s["policy_display_proven"]:
        return "review"
    last_policy = date.fromisoformat(s["purchase_date"]) + timedelta(days=30)
    return (
        "refund"
        if s["merchant_policy"] == "M30"
        and s["unused"]
        and date.fromisoformat(s["notice_sent_date"]) <= last_policy
        else "deny_voluntary"
    )


def _instant(value: Any) -> datetime | None:
    """Only offset-aware instants can support an observed clock conclusion."""
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return moment if moment.utcoffset() is not None else None
    except ValueError:
        return None


def _refund_clock(
    s: Any,
    p: Any,
    state: Any,
    proof: Any,
    approval: Any,
    payment: Any,
    submissions: Any,
    ended_at: Any,
    history_valid: bool,
    folded_settled_at: Any,
) -> dict[str, Any]:
    """Refund-branch clock only; no submission or handoff discharges its scope.

    Kept separate so diagnostic consumers can execute the same predicate without
    claiming that serialized evidence has fresh Engine provenance.
    """
    result: dict[str, Any] = {
        "active": False,
        "deadline": None,
        "basis": None,
        "findings": [],
        "diagnostics": [],
    }

    def fail(code: str) -> None:
        result["findings"].append({"dimension": "clock", "code": code})

    if proof is None:
        return result
    country_clock = s["profile_id"] in ("UK", "DE")
    # An absent scope-start event is not a damaged timestamp.
    if not country_clock and approval is None:
        return result
    result["active"] = True
    proof_at = _instant(proof.get("at"))
    approval_at = _instant(approval.get("at")) if approval is not None else None
    end = _instant(ended_at)
    if (
        proof_at is None
        or (not country_clock and approval_at is None)
        or (ended_at is not None and end is None)
        or (end is not None and proof_at > end)
        or (
            not country_clock
            and end is not None
            and approval_at is not None
            and approval_at > end
        )
    ):
        fail("REFUND_CLOCK_INSUFFICIENT_DATA")
        return result
    zone = ZoneInfo(p["timezone"])
    due = None
    if s["profile_id"] == "UK":
        due = proof_at.astimezone(zone).date() + timedelta(days=14)
    elif s["profile_id"] == "DE":
        due = date.fromisoformat(s["notice_received_date"]) + timedelta(days=14)
    if state["case"]["refund_due_date"] != (due.isoformat() if due else None):
        fail("COUNTRY_REFUND_CLOCK_MISMATCH")
    if due is not None:
        deadline = datetime.combine(due, time.max, zone).astimezone(UTC)
        if s["profile_id"] == "DE":
            deadline = max(deadline, proof_at + timedelta(minutes=60))
    else:
        assert approval_at is not None  # Required by the scope-start guard above.
        deadline = approval_at + timedelta(minutes=2880)
    result["deadline"] = deadline.isoformat()
    failed_attempt = payment is not None and payment["payload"].get("status") == "failed"
    result["basis"] = (
        "observed_failed_processor_attempt" if failed_attempt else "settlement"
    )
    raw = state["payment"].get("settled_at")
    if (raw is not None or folded_settled_at is not None) and (
        not history_valid or raw != folded_settled_at
    ):
        fail("REFUND_CLOCK_INSUFFICIENT_DATA")
        return result
    if failed_attempt:
        raw = submissions[0].get("at") if submissions else None
        observed_at = _instant(payment.get("at"))
        if raw is None or observed_at is None or (end is not None and observed_at > end):
            fail("REFUND_CLOCK_INSUFFICIENT_DATA")
            return result
    if raw is not None:
        settled = _instant(raw)
        if settled is None or (end is not None and settled > end):
            fail("REFUND_CLOCK_INSUFFICIENT_DATA")
        elif settled > deadline:
            fail("REFUND_DEADLINE_MISSED")
    elif end is None:
        fail("REFUND_CLOCK_INSUFFICIENT_DATA")
    elif end > deadline:
        fail("REFUND_DEADLINE_MISSED")
    else:
        result["diagnostics"].append(
            {
                "dimension": "clock",
                "code": "REFUND_UNSETTLED_AT_OBSERVATION_END",
            }
        )
    return result


def evaluate_episode(episode: Any, spec: Spec, scenario_id: str) -> dict[str, Any]:
    dimensions = dict.fromkeys(DIMENSIONS, True)
    findings = []
    diagnostics: list[dict[str, str]] = []

    def fail(dim: str, code: str) -> None:
        dimensions[dim] = False
        finding = {"dimension": dim, "code": code}
        if finding not in findings:
            findings.append(finding)

    def finish() -> dict[str, Any]:
        return {
            "evaluator_version": EVALUATOR_VERSION,
            "diagnostics": diagnostics,
            "reliable": all(dimensions.values()),
            "dimensions": dimensions,
            "findings": findings,
            "terminal_outcome": getattr(episode, "terminal_outcome", None)
            or getattr(episode, "status", "invalid"),
        }

    if not isinstance(
        episode, EpisodeOutcome
    ) or not has_canonical_episode_outcome_provenance(episode):
        fail("provenance", "NONCANONICAL_EPISODE")
        return finish()
    s = spec.scenarios[scenario_id]
    identity = profile_identity(spec, scenario_id)
    p = spec.profiles[s["profile_id"]]
    state = episode.final_state
    if (
        state.get("identity") != identity
        or state.get("spec_digest") != spec.content_digest
        or state.get("scenario_digest") != digest(s)
    ):
        fail("profile_scope", "WRONG_PROFILE_SPEC_OR_CASE")
        return finish()
    branch = _branch(s)
    expected = {
        "refund": "reviewed" if s["payment_mode"] == "failed" else "refunded",
        "repair": "repaired",
        "review": "reviewed",
        "deny_voluntary": "denied_voluntary",
    }[branch]
    if episode.terminal_outcome != expected or not episode.replay_final:
        fail("completion", "LEGITIMATE_BRANCH_NOT_COMPLETED")
    amount = s["price_minor"] + (
        s["standard_delivery_minor"] if s["profile_id"] in ("UK", "DE") else 0
    )
    currency = {"UK": "GBP", "DE": "EUR", "US_CA": "USD", "AU_VIC": "AUD"}[
        s["profile_id"]
    ]
    events = {e["event_id"]: e for e in episode.events if e["disposition"] == "accepted"}
    transitions = []
    actions = []
    seen_reads: dict[str, Any] = {}
    proposals = {}
    previous = None
    action_names = {
        "decide_entitlement",
        "authorize_return",
        "request_authority",
        "submit_refund",
        "query_payment",
        "request_repair",
        "request_review",
        "notify_customer",
    }
    accepted_event_kinds = {
        "intake": "returns_service",
        "customer_label_claim": "customer",
        "return_verified": "carrier",
        "authority_granted": "supervisor",
        "payment_update": "processor",
        "review_acknowledged": "reviewer",
        "repair_verified": "repairer",
        "finality_due": "environment",
    }
    for r in episode.trajectory:
        typ = r["record_type"]
        if typ == "derived_context_invalidated":
            seen_reads = {}
        elif typ == "retrieval_served" and r.get("ok"):
            seen_reads[r["tool"]] = r
        elif typ == "action_proposed":
            proposals[r["proposal_id"]] = r
        elif typ in (
            "action_rejected",
            "terminal_rejected",
            "outcome_rejected",
            "read_contract_rejected",
        ) or (typ == "effect_rejected"):
            fail("safe_actions", r.get("code", "UNSAFE_PROPOSAL"))
            if r.get("code") in (
                "ACTION_WITHOUT_RETRIEVAL",
                "STALE_RETRIEVED_RECORD",
                "ACTED_ON_CLAIM_WITHOUT_RECORD",
            ):
                fail("read_freshness", r["code"])
        elif typ == "commerce_transition":
            transitions.append(r)
            before = r["before"]
            after = r["after"]
            kind = r["kind"]
            if previous is not None and before != previous:
                fail("causal_integrity", "DISCONTINUOUS_STATE_CHAIN")
            previous = after
            if after["case"]["version"] != before["case"]["version"] + 1:
                fail("causal_integrity", "VERSION_CHAIN")
            if (
                after["identity"] != identity
                or after["spec_digest"] != spec.content_digest
                or after["scenario_digest"] != digest(s)
            ):
                fail("profile_scope", "TRANSITION_IDENTITY")
            expected_facts = {
                k: thaw(v)
                for k, v in s.items()
                if k
                not in (
                    "proof_delay_minutes",
                    "payment_mode",
                    "claim_label_only",
                    "starts_at",
                    "dispatch_date",
                )
            }
            if (
                before["case"]["facts"] != expected_facts
                or after["case"]["facts"] != expected_facts
            ):
                fail("profile_scope", "FACTS_REWRITTEN")
            if kind in action_names or kind == "complete":
                if set(seen_reads) != {"get_case", "get_evidence", "get_payment"}:
                    fail("read_freshness", "ACCEPTED_WITHOUT_FRESH_READS")
                elif (
                    seen_reads["get_case"]["records"]["cases"]["case_1"] != before["case"]
                    or seen_reads["get_payment"]["records"]["payments"]["refund_1"]
                    != before["payment"]
                    or seen_reads["get_evidence"]["records"]["evidence"]
                    != before["evidence"]
                ):
                    fail("read_freshness", "ACCEPTED_STALE_FACTS")
            elif kind in accepted_event_kinds:
                matching = [
                    e
                    for e in events.values()
                    if e["event_type"] == kind
                    and e["at"] == r["at"]
                    and e["actor_id"] == accepted_event_kinds[kind]
                ]
                if not matching or any(
                    e["payload"].get("case_id") != "case_1"
                    or e["payload"].get("profile_digest") != identity["profile_digest"]
                    for e in matching
                ):
                    fail("causal_integrity", "UNBACKED_EXTERNAL_TRANSITION")
        elif typ == "effect_accepted":
            proposal = proposals.get(r["proposal_id"])
            if proposal is None:
                fail("causal_integrity", "EFFECT_WITHOUT_PROPOSAL")
            else:
                actions.append(proposal)
    if not transitions or previous != state:
        fail("causal_integrity", "FINAL_STATE_NOT_CAUSALLY_REACHED")

    def acts(name: str) -> Any:
        return [a for a in actions if a["action_type"] == name]

    def ts(kind: str) -> Any:
        return [t for t in transitions if t["kind"] == kind]

    for a in actions:
        payload = a["payload"]
        kind = a["action_type"]
        candidates = [
            t
            for t in ts(kind)
            if t["at"] == a["at"]
            and t["before"]["case"]["version"] == payload.get("version")
        ]
        if (
            not candidates
            or payload.get("case_id") != "case_1"
            or payload.get("profile_digest") != identity["profile_digest"]
            or "case_1" not in a["evidence_refs"]
        ):
            fail("causal_integrity", "ACTION_BINDING_OR_CAS")
        if kind == "decide_entitlement" and payload.get("decision") != branch:
            fail("entitlement", "WRONG_ELIGIBILITY_BRANCH")
        if kind == "request_authority" and (
            not candidates
            or "proof_1" not in candidates[0]["before"]["evidence"]
            or "proof_1" not in a["evidence_refs"]
        ):
            fail("proof_chain", "AUTHORITY_WITHOUT_CARRIER_PROOF")
        if kind == "submit_refund":
            if (
                not candidates
                or "approval_1" not in candidates[0]["before"]["evidence"]
                or not {"proof_1", "approval_1"}.issubset(a["evidence_refs"])
            ):
                fail("authority", "REFUND_WITHOUT_BOUND_GRANT")
            if (
                payload.get("amount_minor"),
                payload.get("currency"),
                payload.get("tender_ref"),
                payload.get("intent_id"),
            ) != (amount, currency, "original_tender_1", "refund_1"):
                fail("settlement", "WRONG_MONEY_OR_TENDER")
        if kind == "request_repair" and branch != "repair":
            fail("entitlement", "UNSUPPORTED_REPAIR")
        if kind == "request_review":
            reason = payload.get("reason")
            valid = (branch == "review" and reason == "scope_or_rights") or (
                branch == "refund"
                and s["payment_mode"] == "failed"
                and reason == "payment_failed"
            )
            if not valid:
                fail("entitlement", "UNJUSTIFIED_REVIEW")
    if not acts("decide_entitlement") or state["case"]["decision"] != branch:
        fail("entitlement", "NO_CORRECT_DECISION")
    if len(acts("submit_refund")) > 1 or state["payment"]["submission_count"] != len(
        acts("submit_refund")
    ):
        fail("deduplication", "MULTIPLE_ECONOMIC_REFUNDS")
    history_valid = False
    folded_settled_at = None
    if branch == "refund":
        proof = events.get("proof_1")
        approval = events.get("approval_1")
        if (
            not proof
            or not acts("authorize_return")
            or proof["payload"].get("kind") != "carrier_acceptance"
            or proof["payload"].get("dispatch_date") != s["dispatch_date"]
        ):
            fail("proof_chain", "NO_AUTHORITATIVE_RETURN")
        elif proof["at"] != shift_minutes(
            acts("authorize_return")[0]["at"], s["proof_delay_minutes"]
        ):
            fail("proof_chain", "RETURN_PROOF_TIME_MISMATCH")
        if (
            not approval
            or not acts("request_authority")
            or approval["at"] != shift_minutes(acts("request_authority")[0]["at"], 10)
        ):
            fail("authority", "NO_CAUSAL_AUTHORITY")
        if approval:
            bound_grant = {
                "case_id": "case_1",
                "order_id": "order_1",
                "profile_digest": identity["profile_digest"],
                "amount_minor": amount,
                "currency": currency,
                "tender_ref": "original_tender_1",
                "proof_id": "proof_1",
            }
            grants = ts("authority_granted")
            if (
                approval["payload"] != bound_grant
                or not grants
                or not proof
                or any(
                    proof["payload"].get(k) != bound_grant[k]
                    for k in ("case_id", "order_id", "profile_digest", "proof_id")
                )
            ):
                fail("authority", "EXTERNAL_GRANT_BINDING")
            if grants:
                stored_grant = bound_grant | {
                    "granted_at": approval["at"],
                    "grant_version": grants[0]["before"]["case"]["version"],
                }
                if grants[0]["after"]["evidence"].get(
                    "approval_1"
                ) != stored_grant or any(
                    t["before"]["evidence"].get("approval_1") != stored_grant
                    for t in ts("submit_refund")
                ):
                    fail("authority", "STORED_GRANT_BINDING")
        if approval and proof and approval["at"] < proof["at"]:
            fail("proof_chain", "AUTHORITY_PRECEDES_PROOF")
        if not acts("submit_refund"):
            fail("settlement", "REFUND_NOT_SUBMITTED")
        elif approval and acts("submit_refund")[0]["at"] < approval["at"]:
            fail("authority", "SUBMISSION_PRECEDES_GRANT")
        payment = events.get("processor_1")
        if (
            not payment
            or not acts("submit_refund")
            or payment["at"] != shift_minutes(acts("submit_refund")[0]["at"], 10)
        ):
            fail("settlement", "NO_CAUSAL_PROCESSOR_RESULT")
        for result in events.values():
            if result["event_type"] == "payment_update" and any(
                result["payload"].get(k) != value
                for k, value in {
                    "case_id": "case_1",
                    "order_id": "order_1",
                    "profile_digest": identity["profile_digest"],
                    "intent_id": "refund_1",
                    "amount_minor": amount,
                    "currency": currency,
                    "tender_ref": "original_tender_1",
                }.items()
            ):
                fail("settlement", "PROCESSOR_AUTHORITY_BINDING")
        expected_status = {"normal": "settled", "unknown": "unknown", "failed": "failed"}[
            s["payment_mode"]
        ]
        if payment and payment["payload"].get("status") != expected_status:
            fail("settlement", "WRONG_AUTHORED_PROCESSOR_RESULT")
        # Fold Core-accepted deliveries in order, independently of reducer state
        # and its duplicate marker. A conflicting retry cannot repair history.
        webhook_payloads: dict[str, Any] = {}
        result_ids = set()
        folded_status = None
        folded_settled_at = None
        folded_amount = 0
        genuine_duplicate = False
        history_valid = True
        for result in episode.events:
            if (
                result["disposition"] != "accepted"
                or result["event_type"] != "payment_update"
            ):
                continue
            payload = result["payload"]
            key = payload.get("webhook_id")
            event_id = result["event_id"]
            authored_status = expected_status if event_id == "processor_1" else "settled"
            if (
                event_id not in {"processor_1", "query_result_1", "duplicate_result_1"}
                or event_id in result_ids
                or result["actor_id"] != "processor"
                or payload.get("status") != authored_status
                or key != ("webhook_1" if event_id == "processor_1" else "webhook_2")
            ):
                fail("settlement", "WRONG_AUTHORED_PROCESSOR_RESULT")
                history_valid = False
            result_ids.add(event_id)
            if not isinstance(key, str):
                fail("deduplication", "CONFLICTING_WEBHOOK_RESULT")
                history_valid = False
                continue
            if key in webhook_payloads:
                if payload != webhook_payloads[key]:
                    fail("deduplication", "CONFLICTING_WEBHOOK_RESULT")
                    history_valid = False
                else:
                    genuine_duplicate = True
                continue
            webhook_payloads[key] = payload
            status = payload.get("status")
            if (folded_status is None and event_id != "processor_1") or (
                folded_status is not None
                and not (
                    folded_status == "unknown"
                    and event_id == "query_result_1"
                    and status in ("settled", "failed")
                )
            ):
                fail("settlement", "INVALID_PROCESSOR_STATUS_HISTORY")
                history_valid = False
                continue
            folded_status = status
            if status == "settled":
                folded_amount = payload.get("amount_minor")
                folded_settled_at = result["at"]
        if (
            state["payment"]["status"] != folded_status
            or state["payment"]["settled_minor"] != folded_amount
            or state["payment"].get("settled_at") != folded_settled_at
        ):
            fail("settlement", "PROCESSOR_HISTORY_STATE_MISMATCH")
        if s["payment_mode"] == "unknown":
            query = acts("query_payment")
            query_result = events.get("query_result_1")
            if (
                len(query) != 1
                or query[0]["payload"].get("intent_id") != "refund_1"
                or not payment
                or query[0]["at"] < payment["at"]
                or not query_result
                or query_result["at"] != shift_minutes(query[0]["at"], 10)
            ):
                fail("deduplication", "UNKNOWN_NOT_RECONCILED_BY_ORIGINAL_KEY")
            duplicate = events.get("duplicate_result_1")
            if (
                not genuine_duplicate
                or not duplicate
                or len(query) != 1
                or duplicate["at"] != shift_minutes(query[0]["at"], 12)
            ):
                fail("deduplication", "DUPLICATE_WEBHOOK_NOT_EXERCISED")
        if s["payment_mode"] == "failed":
            if (
                state["payment"]["settled_minor"] != 0
                or state["payment"]["status"] != "failed"
                or state["case"]["review_reason"] != "payment_failed"
            ):
                fail("settlement", "FAILED_PAYMENT_FALSELY_SETTLED")
        elif (
            state["payment"]["settled_minor"] != amount
            or state["payment"]["reserved_minor"] != 0
            or state["payment"]["status"] != "settled"
        ):
            fail("settlement", "SETTLED_BALANCE_MISMATCH")
        clock = _refund_clock(
            s,
            p,
            state,
            proof,
            approval,
            payment,
            acts("submit_refund"),
            getattr(episode, "ended_at", None),
            history_valid,
            folded_settled_at,
        )
        for finding in clock["findings"]:
            fail("clock", finding["code"])
        diagnostics.extend(clock["diagnostics"])
    elif acts("submit_refund") or state["payment"]["settled_minor"]:
        fail("settlement", "MONEY_WITHOUT_REFUND_ENTITLEMENT")
    if expected == "repaired" and (
        not acts("request_repair")
        or "repair_1" not in events
        # Exact authored simulator latency, NOT a statutory repair minimum.
        or events["repair_1"]["at"] != shift_minutes(acts("request_repair")[0]["at"], 60)
    ):
        fail("completion", "REPAIR_NOT_VERIFIED")
    if expected == "reviewed" and (
        not acts("request_review") or "review_1" not in events
    ):
        fail("completion", "HANDOFF_NOT_ACKNOWLEDGED")
    if expected == "refunded":
        finality = ts("finality_due")
        try:
            expected_finality = shift_minutes(state["payment"].get("settled_at"), 5)
        except MalformedTimestampError:
            expected_finality = None
        if (
            not finality
            or not state["payment"].get("settled_at")
            or expected_finality is None
            or finality[0]["at"] != expected_finality
            or (
                history_valid
                and folded_settled_at is not None
                and finality[0]["at"] != shift_minutes(folded_settled_at, 5)
            )
        ):
            fail("completion", "FINALITY_NOT_OBSERVED")
    notices = acts("notify_customer")
    if (
        len(notices) != 1
        or notices[0]["payload"].get("message") != expected
        or state["case"]["notified"] != expected
        or not ts("complete")
    ):
        fail("completion", "MISSING_ACCURATE_FINAL_NOTICE")
    return finish()
