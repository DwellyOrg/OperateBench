"""Core-owned event provenance must survive permissive domain producers."""

from dataclasses import replace

import pytest

from operatebench.contributions.prospire.property_compliance_profiles.operation import (
    ComplianceDomain,
)
from operatebench.core.engine import has_canonical_episode_outcome_provenance

from .test_pack import FIXTURE, load_spec, run_case


@pytest.mark.parametrize("corrupt", [False, True])
def test_original_event_authority_not_reducer_narrative(corrupt):
    class Domain(ComplianceDomain):
        def build_plan(self):
            plan = super().build_plan()
            return replace(
                plan,
                events=tuple(
                    replace(event, actor_id="contractor")
                    if corrupt and event.event_type == "reports_received"
                    else event
                    for event in plan.events
                ),
            )

        def reduce_event(self, state, event, context):
            if event.event_type == "reports_received":
                event = replace(event, actor_id="inspector")
            return super().reduce_event(state, event, context)

    episode, grade = run_case("EN_normal", domain=Domain(load_spec(FIXTURE), "EN_normal"))
    assert has_canonical_episode_outcome_provenance(episode)
    assert episode.terminal_outcome == "VERIFIED_CURRENT"
    assert grade["reliable"] is not corrupt
    assert {key for key, ok in grade["dimensions"].items() if not ok} == (
        {"provenance"} if corrupt else set()
    )
    if corrupt:
        assert {"dimension": "provenance", "code": "EVENT_AUTHORITY_JOIN"} in grade[
            "findings"
        ]


@pytest.mark.parametrize(
    "mode",
    [
        "valid",
        "missing",
        "duplicate",
        "payload",
        "actor",
        "type",
        "refused",
        "malformed_payload",
        "malformed_type",
        "malformed_boundary",
    ],
)
def test_next_core_boundary_owns_domain_row(mode):
    from operatebench.core.protocol import Verdict

    class Domain(ComplianceDomain):
        def reduce_event(self, state, event, context):
            class Proxy:
                def __getattr__(self, name):
                    return getattr(context, name)

                def record(self, kind, payload):
                    if kind == "compliance_event" and event.event_type == "case_opened":
                        if mode == "missing":
                            return None
                        payload = dict(payload)
                        if mode == "payload":
                            payload["payload"] = dict(payload["payload"]) | {
                                "invented": True
                            }
                        elif mode == "actor":
                            payload["actor_id"] = "inspector"
                        elif mode == "type":
                            payload["event_type"] = "invented"
                        elif mode == "malformed_payload":
                            payload["payload"] = None
                        elif mode == "malformed_type":
                            payload["event_type"] = []
                        if mode == "malformed_boundary":
                            context.record(kind, payload)
                            context.record(
                                "event_rejected", {"event_id": [], "event_type": []}
                            )
                            return None
                        if mode == "duplicate":
                            context.record(kind, payload)
                    return context.record(kind, payload)

            verdict = super().reduce_event(state, event, Proxy())
            if event.event_type == "case_opened" and mode == "refused":
                return Verdict.refused("TEST_REFUSAL", "recorded but not accepted")
            return verdict

    episode, grade = run_case("EN_normal", domain=Domain(load_spec(FIXTURE), "EN_normal"))
    assert has_canonical_episode_outcome_provenance(episode)
    assert grade["dimensions"]["provenance"] is (mode == "valid")
    assert {key for key, ok in grade["dimensions"].items() if not ok} == (
        set() if mode == "valid" else {"provenance"}
    )
    if mode != "valid":
        assert {"dimension": "provenance", "code": "EVENT_AUTHORITY_JOIN"} in grade[
            "findings"
        ]
