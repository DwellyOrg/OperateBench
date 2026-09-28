"""Adversarial strict-boundary and causal grading probes for the new namespace."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from operatebench.core.errors import SpecSchemaError
from operatebench.core.events import Event
from operatebench.domains.commerce.return_refund_profiles import (
    build_agent,
    build_domain,
    evaluate_episode,
    load_spec,
)
from tests.test_commerce_return_refund_profiles import FIXTURE, run


@pytest.mark.parametrize(
    "field,value",
    [
        ("price_minor", True),
        ("price_minor", -1),
        ("proof_delay_minutes", 0),
        ("profile_id", "UK_UNKNOWN"),
        ("delivery_date", "2026-02-30"),
        ("disclosed", "yes"),
        ("extra", 1),
        ("payment_mode", "refund_completed"),
        ("starts_at", "2026-01-12T12:00:00+00:00"),
        ("starts_at", "2026-1-12T12:00:00Z"),
        ("dispatch_date", "2026-01-29"),
    ],
)
def test_strict_scenario_rejection(tmp_path, field, value):
    raw = json.loads(FIXTURE.read_text())
    raw["scenarios"]["UK_NORMAL"][field] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(json.dumps(raw))
    with pytest.raises(SpecSchemaError):
        load_spec(path)


@pytest.mark.parametrize(
    "text",
    [
        "x: 1\nx: 2\n",
        "? [x, y]\n: z\n",
        "x: &a [*a]\n",
        "true: 1\n",
        "null",
        "[]",
        "x: !!python/object:foo {}",
    ],
)
def test_strict_yaml_keys_and_aliases(tmp_path, text):
    path = tmp_path / "invalid.yaml"
    path.write_text(text)
    with pytest.raises(SpecSchemaError):
        load_spec(path)


def test_profile_and_scenario_recursive_immutability(tmp_path):
    spec = load_spec(FIXTURE)
    with pytest.raises(TypeError):
        spec.scenarios["UK_NORMAL"]["price_minor"] = 1
    with pytest.raises(TypeError):
        spec.profiles["UK"]["legal_rules"][0]["sources"][0]["url"] = "bad"
    domain = build_domain(spec, "UK_NORMAL")
    view = domain.policy_view()
    view["legal_rules"][0]["sources"][0]["url"] = "bad"
    assert domain.policy_view()["legal_rules"][0]["sources"][0]["url"] != "bad"
    raw = json.loads(FIXTURE.read_text())
    raw["profiles"]["UK"] = "0" * 64
    path = tmp_path / "bad-profile.yaml"
    path.write_text(json.dumps(raw))
    with pytest.raises(SpecSchemaError):
        load_spec(path)


class Context:
    now = "2026-01-13T12:20:00Z"

    def record(self, *args):
        pass

    def schedule_timer(self, *args, **kwargs):
        pass


def test_authority_is_bound_to_exact_quote_not_just_present():
    spec = load_spec(FIXTURE)
    domain = build_domain(spec, "UK_NORMAL")
    e = run("UK_NORMAL")
    transition = next(
        r
        for r in e.trajectory
        if r["record_type"] == "commerce_transition" and r["kind"] == "submit_refund"
    )
    state = deepcopy(transition["before"])
    state["evidence"]["approval_1"]["amount_minor"] += 1
    c = state["case"]
    verdict = domain.apply_action(
        state,
        "submit_refund",
        {
            "case_id": "case_1",
            "profile_digest": state["identity"]["profile_digest"],
            "version": c["version"],
            "intent_id": "refund_1",
            "amount_minor": c["amount_minor"],
            "currency": c["currency"],
            "tender_ref": c["tender_ref"],
        },
        ("case_1", "proof_1", "approval_1"),
        Context(),
    )
    assert not verdict.accepted and verdict.code == "AUTHORITY_BINDING"


def test_de_review_does_not_invent_withdrawal_refund_clock():
    assert run("DE_FAULT_REVIEW").final_state["case"]["refund_due_date"] is None
    assert run("DE_MISSING_DISCLOSURE").final_state["case"]["refund_due_date"] is None


def test_grader_does_not_accept_self_asserted_summary():
    spec = load_spec(FIXTURE)
    episode = run("UK_NORMAL")
    forged = replace(episode, terminal_outcome="refunded", replay_final=True)
    assert not evaluate_episode(forged, spec, "UK_NORMAL")["reliable"]
    episode.final_state["case"]["amount_minor"] = 1
    assert not evaluate_episode(episode, spec, "UK_NORMAL")["reliable"]


def test_independent_grader_kills_runtime_entitlement_mutant():
    from operatebench.core.engine import Engine
    from operatebench.domains.commerce.return_refund_profiles.operation import (
        CommerceProfilesDomain,
    )

    class BrokenDomain(CommerceProfilesDomain):
        def apply_action(self, state, action_type, payload, evidence_refs, context):
            if action_type == "decide_entitlement":
                before = deepcopy(state)
                state["case"]["decision"] = "refund"
                state["case"]["phase"] = "refund"
                return self._commit(state, context, action_type, before)
            return super().apply_action(
                state, action_type, payload, evidence_refs, context
            )

    spec = load_spec(FIXTURE)
    episode = Engine(
        BrokenDomain(spec, "AU_CHANGE_OF_MIND"),
        build_agent("reference"),
        identity={"operation_id": spec.operation_id},
    ).run()
    assert episode.terminal_outcome == "refunded"
    grade = evaluate_episode(episode, spec, "AU_CHANGE_OF_MIND")
    assert not grade["reliable"]
    assert not grade["dimensions"]["entitlement"]
    assert not grade["dimensions"]["settlement"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("intent_id", "other_refund"),
        ("amount_minor", 1),
        ("currency", "USD"),
        ("order_id", "other_order"),
        ("unexpected", True),
    ],
)
def test_processor_events_have_exact_bindings(field, value):
    spec = load_spec(FIXTURE)
    domain = build_domain(spec, "UK_NORMAL")
    e = run("UK_NORMAL")
    before = deepcopy(
        next(
            r["before"]
            for r in e.trajectory
            if r["record_type"] == "commerce_transition" and r["kind"] == "payment_update"
        )
    )
    raw = next(x for x in e.events if x["event_id"] == "processor_1")
    payload = dict(raw["payload"])
    payload[field] = value
    event = Event(
        raw["event_id"],
        raw["event_type"],
        raw["actor_id"],
        raw["at"],
        raw["sequence"],
        payload,
        caused_by=raw["caused_by"],
    )
    ctx = Context()
    ctx.now = raw["at"]
    verdict = domain.reduce_event(before, event, ctx)
    assert not verdict.accepted


def test_finality_cannot_be_forged_early():
    spec = load_spec(FIXTURE)
    domain = build_domain(spec, "UK_NORMAL")
    e = run("UK_NORMAL")
    before = deepcopy(
        next(
            r["before"]
            for r in e.trajectory
            if r["record_type"] == "commerce_transition" and r["kind"] == "finality_due"
        )
    )
    raw = next(x for x in e.events if x["event_type"] == "finality_due")
    event = Event(
        "wrong_timer",
        "finality_due",
        "environment",
        before["payment"]["settled_at"],
        raw["sequence"],
        raw["payload"],
    )
    ctx = Context()
    ctx.now = event.at
    assert not domain.reduce_event(before, event, ctx).accepted


def test_conflicting_duplicate_webhook_is_not_silently_ignored():
    spec = load_spec(FIXTURE)
    domain = build_domain(spec, "UK_LABEL_UNKNOWN")
    e = run("UK_LABEL_UNKNOWN")
    before = deepcopy(
        next(
            r["after"]
            for r in e.trajectory
            if r["record_type"] == "commerce_transition"
            and r["kind"] == "payment_update"
            and r["after"]["payment"]["status"] == "settled"
        )
    )
    raw = next(x for x in e.events if x["event_id"] == "duplicate_result_1")
    payload = dict(raw["payload"])
    payload["status"] = "failed"
    event = Event(
        raw["event_id"],
        raw["event_type"],
        raw["actor_id"],
        raw["at"],
        raw["sequence"],
        payload,
        caused_by=raw["caused_by"],
    )
    ctx = Context()
    ctx.now = raw["at"]
    assert not domain.reduce_event(before, event, ctx).accepted


def test_temporal_metadata_is_self_contained():
    from operatebench.domains.commerce.return_refund_profiles.spec import PROFILES, thaw

    assert "RESEARCH.md" not in json.dumps(thaw(PROFILES))
    for profile in PROFILES.values():
        for rule in profile["legal_rules"]:
            assert "source" in rule["effective_date_status"]
            assert all(source["effective_date_status"] for source in rule["sources"])
