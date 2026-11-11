"""Invented, test-only freight contracts; no company policy or API."""

from collections.abc import Mapping
from copy import deepcopy

from operatebench.core.protocol import EpisodePlan, PlannedEvent, Verdict
from operatebench.core.read_contract import ActionEvidenceContract
from operatebench.core.retrieval import ToolResult
from operatebench.core.retrieval_evidence import public_record_version

LOAD = "synthetic-load-a"
VARIANTS = ("match", "missing", "clarified", "privacy", "correction")
ISSUERS = {
    "tracking": "mobile",
    "carrier": "carrier",
    "pod": "documents",
    "approval": "broker",
    "checkpoint": "clock",
}
SCHEMAS = {
    "notify": {"required": {"load": "string", "version": "integer"}, "optional": {}},
    "correct_notice": {
        "required": {
            "load": "string",
            "prior_notice": "string",
            "prior_version": "integer",
            "source_ref": "string",
            "recipient": "string",
        },
        "optional": {},
    },
    "handoff": {"required": {"reason": "string"}, "optional": {}},
}


class VisibilityDomain:
    def __init__(self, variant):
        if variant not in VARIANTS:
            raise ValueError("unknown synthetic variant")
        self.variant = variant

    def build_plan(self):
        events = []

        def add(kind, minute, payload):
            events.append(
                PlannedEvent(
                    f"synthetic-event-{len(events)}",
                    kind,
                    ISSUERS[kind],
                    len(events),
                    payload,
                    at=f"2026-01-01T09:{minute:02d}:00Z",
                )
            )

        add(
            "tracking",
            0,
            {
                "load": LOAD,
                "signal": "disabled" if self.variant == "privacy" else "geofence",
            },
        )
        add(
            "carrier",
            1,
            {
                "load": LOAD,
                "status": "in_transit" if self.variant == "clarified" else "delivered",
                "version": 1,
            },
        )
        if self.variant != "missing":
            add("pod", 2, {"load": LOAD, "version": 1, "status": "delivered"})
        if self.variant == "clarified":
            add("carrier", 3, {"load": LOAD, "status": "delivered", "version": 2})
            add("pod", 3, {"load": LOAD, "version": 2, "status": "delivered"})
        if self.variant != "missing":
            add(
                "approval",
                4,
                {"load": LOAD, "version": 2 if self.variant == "clarified" else 1},
            )
        if self.variant == "correction":
            add("pod", 5, {"load": LOAD, "version": 2, "status": "in_transit"})
            add("approval", 6, {"load": LOAD, "version": 1})
        add("checkpoint", 8, {})
        return EpisodePlan("2026-01-01T09:00:00Z", tuple(events), 10, 8, 20)

    def initial_state(self):
        return {
            "load": LOAD,
            "shipper": "synthetic-shipper",
            "carrier_actor": "carrier",
            "status_permission": True,
            "location_permission": False,
            "tracking": "unknown",
            "carrier": None,
            "pod": None,
            "approval": None,
            "notice": None,
            "outbound_notices": [],
            "pending_corrections": [],
            "handoff": None,
            "checkpoint": False,
            "final": None,
        }

    def canonical_state(self, state):
        return deepcopy(state)

    def coarse_phase(self, state):
        return "exception"

    def event_authority(self, actor_id, event_type):
        return "actor_claim" if actor_id in ("mobile", "carrier") else "system_event"

    def retrieval_catalogue(self):
        return {
            "get_shipment": {
                "source": "synthetic_shipment",
                "authority": "system_of_record",
                "schema_id": "synthetic.visibility.v1",
                "arguments": {},
            }
        }

    def retrieval_record_contract(self):
        return [LOAD, "synthetic shipment snapshot", "sha256-json-v1", 64]

    def read_requirements(self):
        return ActionEvidenceContract(
            {k: {"get_shipment": "required"} for k in (*SCHEMAS, "complete")}
        )

    def serve_retrieval(self, state, requests, as_of):
        records = {"record": deepcopy(state)}
        return [
            ToolResult(
                r.tool,
                "synthetic_shipment",
                "system_of_record",
                LOAD,
                public_record_version(
                    records, context="synthetic shipment snapshot", length=64
                ),
                as_of,
                "synthetic.visibility.v1",
                records,
            )
            for r in requests
        ]

    def policy_view(self):
        return {
            "invented": True,
            "notice": "matching carrier/POD + broker approval",
            "privacy": "status only; never disclose location",
            "completion": "checkpoint then verified notice or documented human hold",
        }

    def actor_roles(self):
        return {
            "carrier": "synthetic authorized carrier",
            "broker": "synthetic reviewer",
            "shipper": "synthetic status recipient",
        }

    def message_fixture_ids(self):
        return ()

    def action_schemas(self):
        schemas = deepcopy(SCHEMAS)
        for schema in schemas.values():
            schema["reads"] = {"get_shipment": "required"}
        schemas["notify"]["field_guidance"] = {
            "load": "shipment-of-record load; must match current POD",
            "version": "current POD version; carrier and broker approval must match",
            "evidence_refs": "exactly shipment load, pod:{version}, approval:{version}",
        }
        schemas["correct_notice"]["field_guidance"] = {
            "prior_notice": "exact pending outbound notice ID",
            "prior_version": "version of that sent notice",
            "source_ref": "current carrier/POD reference in pending obligation",
            "recipient": "original permitted status recipient",
            "evidence_refs": "ordered load, prior notice ID, corrected source reference",
        }
        schemas["handoff"]["field_guidance"] = {
            "reason": "nonempty explanation of the unresolved exception for the broker",
        }
        return schemas

    def wake_event_vocabulary(self):
        return tuple(ISSUERS)

    def reduce_event(self, state, event, context):
        kind, p = event.event_type, event.payload
        if kind not in ISSUERS:
            return Verdict.refused("EVENT_TYPE", "undeclared event")
        if event.actor_id != ISSUERS[kind]:
            return Verdict.refused("EVENT_ACTOR", "wrong synthetic issuer")
        fields = {
            "tracking": {"load", "signal"},
            "carrier": {"load", "status", "version"},
            "pod": {"load", "version", "status"},
            "approval": {"load", "version"},
            "checkpoint": set(),
        }[kind]
        if not isinstance(p, Mapping) or set(p) != fields:
            return Verdict.refused("EVENT_PAYLOAD", "exact payload required")
        if (
            ("version" in p and (type(p["version"]) is not int or p["version"] < 1))
            or ("load" in p and (type(p["load"]) is not str or not p["load"].strip()))
            or ("status" in p and p["status"] not in ("delivered", "in_transit"))
            or (
                kind == "tracking"
                and p["signal"] not in ("geofence", "disabled", "stale")
            )
        ):
            return Verdict.refused("EVENT_PAYLOAD", "invalid scalar")
        if "load" in p and p["load"] != LOAD:
            return Verdict.refused(
                "EVENT_LOAD", "event must bind shipment-of-record load"
            )
        if kind == "approval" and (
            not state["pod"] or p["version"] != state["pod"]["version"]
        ):
            return Verdict.refused("STALE_APPROVAL", "approval must bind current POD")
        if (
            kind in ("pod", "carrier")
            and state[kind]
            and p["version"] <= state[kind]["version"]
        ):
            return Verdict.ok("STALE_EVENT")
        if kind == "tracking":
            state["tracking"] = p["signal"] if p["signal"] == "geofence" else "unknown"
        elif kind == "checkpoint":
            state["checkpoint"] = True
        else:
            changed = state[kind] != p
            state[kind] = deepcopy(dict(p))
            if kind == "pod" or (kind == "carrier" and changed):
                state["approval"] = None
                for notice in state["outbound_notices"]:
                    if p["status"] != "delivered" or p["version"] != notice["version"]:
                        obligation = {
                            "load": LOAD,
                            "prior_notice": notice["notice_id"],
                            "prior_version": notice["version"],
                            "source_ref": f"{kind}:{p['version']}",
                            "recipient": notice["recipient"],
                        }
                        state["pending_corrections"] = [
                            item
                            for item in state["pending_corrections"]
                            if item["prior_notice"] != notice["notice_id"]
                        ] + [obligation]
                state["notice"] = None
        context.record(
            "visibility_event", {"kind": kind, "source": event.actor_id, "data": dict(p)}
        )
        return Verdict.ok()

    def apply_action(self, state, action_type, payload, evidence_refs, context):
        if action_type not in SCHEMAS or not isinstance(payload, Mapping):
            return Verdict.refused("ACTION_PAYLOAD", "unknown action/payload")
        required = SCHEMAS[action_type]["required"]
        if set(payload) != set(required) or any(
            type(payload[k]) is not (int if t == "integer" else str)
            for k, t in required.items()
        ):
            return Verdict.refused("ACTION_PAYLOAD", "exact typed fields required")
        if action_type == "handoff":
            if not payload["reason"].strip():
                return Verdict.refused("ACTION_PAYLOAD", "reason required")
            state["handoff"] = {
                "recipient": "broker",
                "reason": payload["reason"],
                "pod": deepcopy(state["pod"]),
            }
            context.record("human_hold", deepcopy(state["handoff"]))
            return Verdict.ok()
        if action_type == "correct_notice":
            if payload["load"] != LOAD:
                return Verdict.refused("CORRECTION_BINDING", "exact shipment required")
            if not state["status_permission"] or payload["recipient"] != "shipper":
                return Verdict.refused("STATUS_PERMISSION", "recipient not permitted")
            if dict(payload) not in state["pending_corrections"]:
                return Verdict.refused(
                    "CORRECTION_BINDING", "pending notice and current source required"
                )
            if tuple(evidence_refs) != (
                LOAD,
                payload["prior_notice"],
                payload["source_ref"],
            ):
                return Verdict.refused(
                    "PROVENANCE", "exact correction references required"
                )
            source_kind, source_version = payload["source_ref"].split(":")
            source = state[source_kind]
            if source["version"] != int(source_version):
                return Verdict.refused("CORRECTION_BINDING", "current source required")
            context.record(
                "notice_correction",
                {
                    **payload,
                    "source": deepcopy(source),
                    "evidence_refs": list(evidence_refs),
                },
            )
            state["pending_corrections"].remove(dict(payload))
            return Verdict.ok()
        pod, carrier = state["pod"], state["carrier"]
        if (
            not pod
            or not carrier
            or pod["status"] != "delivered"
            or (carrier["status"] != "delivered" or carrier["version"] != pod["version"])
        ):
            return Verdict.refused(
                "DELIVERY_EVIDENCE", "matching delivery claims required"
            )
        if pod["load"] != LOAD or payload != {"load": LOAD, "version": pod["version"]}:
            return Verdict.refused("POD_BINDING", "exact shipment/version required")
        if state["approval"] != {"load": LOAD, "version": pod["version"]}:
            return Verdict.refused(
                "NOTIFICATION_CHECKPOINT", "current broker approval required"
            )
        if not state["status_permission"]:
            return Verdict.refused("STATUS_PERMISSION", "recipient not permitted")
        if set(evidence_refs) != {
            LOAD,
            f"pod:{pod['version']}",
            f"approval:{pod['version']}",
        }:
            return Verdict.refused("PROVENANCE", "exact evidence references required")
        notice_id = f"notice:{len(state['outbound_notices']) + 1}"
        state["notice"] = dict(payload)
        state["outbound_notices"].append(
            {**payload, "notice_id": notice_id, "recipient": "shipper"}
        )
        context.record(
            "status_notice",
            {
                "recipient": "shipper",
                "notice_id": notice_id,
                **payload,
                "evidence_refs": list(evidence_refs),
            },
        )
        return Verdict.ok()

    def apply_terminal(self, state, evidence_refs, context):
        if state["pending_corrections"]:
            return Verdict.refused("PENDING_CORRECTION", "outbound compensation required")
        if not state["checkpoint"] or not (state["notice"] or state["handoff"]):
            return Verdict.refused(
                "OPEN_EXCEPTION", "checkpoint and disposition required"
            )
        state["final"] = "verified_notice" if state["notice"] else "human_hold"
        return Verdict.ok()

    def is_replay_final(self, state):
        return state["final"] is not None

    def terminal_outcome(self, state):
        return state["final"]

    def current_cycle_id(self, state):
        return LOAD
