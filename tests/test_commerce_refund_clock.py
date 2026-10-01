"""Observed refund deadlines, independent of operational completion."""

from copy import deepcopy
from dataclasses import replace

import pytest

from operatebench.core.engine import has_canonical_episode_outcome_provenance
from operatebench.core.outcomes import Act, Wait
from operatebench.domains.commerce.return_refund_profiles import evaluator
from operatebench.domains.commerce.return_refund_profiles.agents import ReferenceAgent
from operatebench.domains.commerce.return_refund_profiles.operation import (
    CommerceProfilesDomain,
)
from operatebench.sdk.development_runtime import (
    DevelopmentRuntimeError,
    _evaluation_fields,
)
from tests.test_commerce_return_refund_profiles_remediation import FIXTURE, load_spec, run


@pytest.fixture
def inputs():
    return {
        "s": {"profile_id": "UK", "notice_received_date": "2026-01-13"},
        "p": {"timezone": "Europe/London"},
        "state": {"case": {"refund_due_date": "2026-01-27"}, "payment": {}},
        "proof": {"at": "2026-01-13T12:00:00Z"},
        "approval": {"at": "2026-01-13T12:10:00Z"},
        "payment": None,
        "submissions": [{"at": "2026-01-13T12:10:00Z"}],
        "ended_at": "2026-01-28T00:00:00Z",
        "history_valid": True,
        "folded_settled_at": None,
    }


def codes(result):
    return {finding["code"] for finding in result["findings"]}


@pytest.mark.parametrize("settled", [False, True])
@pytest.mark.parametrize(
    "instant,late",
    [
        ("2026-01-27T23:59:59Z", False),
        ("2026-01-27T23:59:59.999999Z", False),
        ("2026-01-28T00:00:00Z", True),
    ],
)
def test_inclusive_settlement_and_observation_boundaries(inputs, settled, instant, late):
    if settled:
        inputs["state"]["payment"]["settled_at"] = instant
        inputs["folded_settled_at"] = instant
    else:
        inputs["ended_at"] = instant
    result = evaluator._refund_clock(**inputs)
    assert codes(result) == ({"REFUND_DEADLINE_MISSED"} if late else set())
    assert bool(result["diagnostics"]) is (not settled and not late)


@pytest.mark.parametrize(
    "field", ["proof", "approval", "ended_at", "settled_at", "attempt"]
)
@pytest.mark.parametrize(
    "bad", [None, "bad", "2026-01-13T12:00:00", "2099-01-01T00:00:00Z"]
)
def test_unreliable_required_instants_fail_closed(inputs, field, bad):
    if field == "approval":
        inputs["s"]["profile_id"] = "US_CA"
        inputs["state"]["case"]["refund_due_date"] = None
    if field in ("proof", "approval"):
        inputs[field] = {} if bad is None else {"at": bad}
    elif field == "attempt":
        inputs["payment"] = {
            "payload": {"status": "failed"},
            "at": "2026-01-13T12:20:00Z",
        }
        inputs["submissions"] = [{} if bad is None else {"at": bad}]
    elif field == "settled_at":
        inputs["state"]["payment"]["settled_at"] = bad
        # Missing state timestamp despite a recorded settlement is insufficient.
        inputs["folded_settled_at"] = bad or "2026-01-13T12:20:00Z"
    else:
        inputs[field] = bad
        if bad == "2099-01-01T00:00:00Z":
            # An observation end is the bound itself; it cannot be "future"
            # relative to itself. A future proof is tested separately above.
            assert codes(evaluator._refund_clock(**inputs)) == {"REFUND_DEADLINE_MISSED"}
            return
    assert codes(evaluator._refund_clock(**inputs)) == {"REFUND_CLOCK_INSUFFICIENT_DATA"}


def test_settlement_can_stand_without_observation_end(inputs):
    inputs.update(ended_at=None, folded_settled_at="2026-01-13T12:20:00Z")
    inputs["state"]["payment"]["settled_at"] = inputs["folded_settled_at"]
    assert not codes(evaluator._refund_clock(**inputs))


@pytest.mark.parametrize("failed_attempt", [False, True])
@pytest.mark.parametrize("raw", ["2026-01-13T12:20:00Z", "bad"])
@pytest.mark.parametrize("invalid_history", [False, True])
def test_settlement_requires_matching_valid_processor_history(
    inputs, invalid_history, raw, failed_attempt
):
    if failed_attempt:
        inputs["payment"] = {
            "payload": {"status": "failed"},
            "at": "2026-01-13T12:20:00Z",
        }
    inputs["state"]["payment"]["settled_at"] = raw
    inputs["folded_settled_at"] = "2026-01-13T12:20:00Z" if invalid_history else None
    inputs["history_valid"] = not invalid_history
    assert codes(evaluator._refund_clock(**inputs)) == {"REFUND_CLOCK_INSUFFICIENT_DATA"}


@pytest.mark.parametrize("history_valid", [False, True])
def test_failed_attempt_without_settlement_requires_valid_history(inputs, history_valid):
    inputs["payment"] = {
        "payload": {"status": "failed"},
        "at": "2026-01-13T12:20:00Z",
    }
    inputs["history_valid"] = history_valid
    assert inputs["state"]["payment"].get("settled_at") is None
    assert inputs["folded_settled_at"] is None
    result = evaluator._refund_clock(**inputs)
    assert result["basis"] == "observed_failed_processor_attempt"
    assert codes(result) == (
        set() if history_valid else {"REFUND_CLOCK_INSUFFICIENT_DATA"}
    )
    assert not result["diagnostics"]


@pytest.mark.parametrize("valid_history", [False, True])
def test_engine_failed_attempt_history_controls_clock_dimension(valid_history):
    class Domain(CommerceProfilesDomain):
        def build_plan(self):
            plan = super().build_plan()
            return replace(
                plan,
                events=tuple(
                    replace(e, payload=dict(e.payload) | {"webhook_id": "wrong_webhook"})
                    if not valid_history and e.event_id == "processor_1"
                    else e
                    for e in plan.events
                ),
            )

    spec = load_spec(FIXTURE)
    case = "UK_PAYMENT_FAILED"
    episode = run(case, domain=Domain(spec, case))
    assert has_canonical_episode_outcome_provenance(episode)
    assert episode.terminal_outcome == "reviewed"
    payment = next(e for e in episode.events if e["event_id"] == "processor_1")
    assert payment["disposition"] == "accepted"
    assert payment["payload"]["status"] == "failed"
    assert episode.final_state["payment"]["status"] == "failed"
    assert episode.final_state["payment"].get("settled_at") is None
    assert episode.final_state["payment"]["settled_minor"] == 0
    grade = evaluator.evaluate_episode(episode, spec, case)
    # Binary reliability already rejects bad history; the clock dimension must
    # independently refuse its use as evidence of a timely failed attempt.
    assert grade["reliable"] is valid_history
    assert grade["dimensions"]["settlement"] is valid_history
    assert grade["dimensions"]["clock"] is valid_history
    assert codes(grade) == (
        set()
        if valid_history
        else {"WRONG_AUTHORED_PROCESSOR_RESULT", "REFUND_CLOCK_INSUFFICIENT_DATA"}
    )
    assert not grade["diagnostics"]


@pytest.mark.parametrize("profile", ["UK", "DE", "US_CA", "AU_VIC"])
@pytest.mark.parametrize(
    "proof,approval", [(False, False), (False, True), (True, False), (True, True)]
)
def test_obligation_activation_without_submit(inputs, profile, proof, approval):
    inputs["s"]["profile_id"] = profile
    inputs["p"]["timezone"] = "Europe/Berlin" if profile == "DE" else "Europe/London"
    inputs["proof"] = inputs["proof"] if proof else None
    inputs["approval"] = inputs["approval"] if approval else None
    inputs["submissions"] = []
    if profile not in ("UK", "DE"):
        inputs["state"]["case"]["refund_due_date"] = None
    active = proof and (profile in ("UK", "DE") or approval)
    result = evaluator._refund_clock(**inputs)
    assert result["active"] is active
    assert codes(result) == ({"REFUND_DEADLINE_MISSED"} if active else set())


@pytest.mark.parametrize(
    "profile,zone,proof,notice,due,deadline",
    [
        (
            "UK",
            "Europe/London",
            "2026-03-28T23:30:00Z",
            "2026-03-20",
            "2026-04-11",
            "2026-04-11T22:59:59.999999+00:00",
        ),
        (
            "UK",
            "Europe/London",
            "2026-06-01T23:30:00Z",
            "2026-06-01",
            "2026-06-16",
            "2026-06-16T22:59:59.999999+00:00",
        ),
        (
            "DE",
            "Europe/Berlin",
            "2026-03-29T22:30:00Z",
            "2026-03-20",
            "2026-04-03",
            "2026-04-03T21:59:59.999999+00:00",
        ),
        (
            "DE",
            "Europe/Berlin",
            "2026-04-10T12:00:00Z",
            "2026-03-20",
            "2026-04-03",
            "2026-04-10T13:00:00+00:00",
        ),
    ],
)
def test_local_dates_dst_and_de_withholding(
    inputs, profile, zone, proof, notice, due, deadline
):
    inputs["s"].update(profile_id=profile, notice_received_date=notice)
    inputs["p"]["timezone"] = zone
    inputs["proof"]["at"] = proof
    inputs["state"]["case"]["refund_due_date"] = due
    inputs["ended_at"] = deadline
    result = evaluator._refund_clock(**inputs)
    assert result["deadline"] == deadline
    assert not codes(result)
    assert result["diagnostics"]


@pytest.mark.parametrize("observed", [False, True])
def test_failed_mode_alone_does_not_activate_attempt_exception(inputs, observed):
    inputs["s"]["payment_mode"] = "failed"
    if observed:
        inputs["payment"] = {
            "payload": {"status": "failed"},
            "at": "2026-01-13T12:20:00Z",
        }
    result = evaluator._refund_clock(**inputs)
    assert codes(result) == (set() if observed else {"REFUND_DEADLINE_MISSED"})


class StopBefore(ReferenceAgent):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def decide(self, observation):
        decision = super().decide(observation)
        if isinstance(decision, Act) and decision.action_type == self.action:
            return Wait("leave work unfinished", observation.wake_event_types, 1440)
        return decision


@pytest.mark.parametrize("action", ["query_payment", "submit_refund"])
@pytest.mark.parametrize("late", [False, True])
def test_genuine_engine_censoring_and_no_submit_remain_incomplete(action, late):
    class Domain(CommerceProfilesDomain):
        def build_plan(self):
            return replace(super().build_plan(), horizon_minutes=43200 if late else 1500)

    spec = load_spec(FIXTURE)
    episode = run(
        "UK_LABEL_UNKNOWN", Domain(spec, "UK_LABEL_UNKNOWN"), StopBefore(action)
    )
    assert has_canonical_episode_outcome_provenance(episode)
    assert not episode.final_state["payment"].get("settled_at")
    grade = evaluator.evaluate_episode(episode, spec, "UK_LABEL_UNKNOWN")
    assert grade["evaluator_version"] == evaluator.EVALUATOR_VERSION
    assert grade["dimensions"]["clock"] is not late
    assert not grade["reliable"]
    assert not grade["dimensions"]["completion"]
    assert not grade["dimensions"]["settlement"]
    assert bool(grade["diagnostics"]) is not late
    assert ("REFUND_DEADLINE_MISSED" in codes(grade)) is late
    assert episode.final_state["payment"]["submission_count"] == (
        action == "query_payment"
    )


@pytest.mark.parametrize(
    "case,bad",
    [
        ("UK_NORMAL", bad)
        for bad in [None, "bad", "2026-01-13T12:20:00", "2099-01-01T00:00:00Z"]
    ]
    + [
        ("UK_PAYMENT_FAILED", bad)
        for bad in [
            "2026-01-13T12:20:00Z",
            "bad",
            "2026-01-13T12:20:00",
            "2099-01-01T00:00:00Z",
        ]
    ],
)
def test_real_evaluator_handles_unreliable_stored_settlement(case, bad):
    class Domain(CommerceProfilesDomain):
        def reduce_event(self, state, event, context):
            failed = case == "UK_PAYMENT_FAILED"
            if event.event_type == ("review_acknowledged" if failed else "finality_due"):
                before = deepcopy(state)
                state["payment"]["settled_at"] = bad
                state["case"]["review_ack" if failed else "finality_ready"] = True
                return self._commit(state, context, event.event_type, before)
            return super().reduce_event(state, event, context)

    spec = load_spec(FIXTURE)
    episode = run(case, domain=Domain(spec, case))
    assert has_canonical_episode_outcome_provenance(episode)
    assert episode.final_state["payment"]["settled_at"] == bad
    if case == "UK_PAYMENT_FAILED":
        assert episode.final_state["payment"]["status"] == "failed"
        assert episode.final_state["payment"]["settled_minor"] == 0
        assert episode.final_state["case"]["review_ack"]
    grade = evaluator.evaluate_episode(episode, spec, case)
    assert "REFUND_CLOCK_INSUFFICIENT_DATA" in codes(grade)
    assert "REFUND_DEADLINE_MISSED" not in codes(grade)
    assert not grade["dimensions"]["clock"]
    assert not grade["diagnostics"]
    assert not grade["dimensions"]["settlement"]
    if case == "UK_NORMAL":
        assert not grade["dimensions"]["completion"]


@pytest.mark.parametrize(
    "metadata",
    [
        {"evaluator_version": "0.3.1"},
        {"diagnostics": []},
        {"evaluator_version": "", "diagnostics": []},
        {"evaluator_version": "0.3.1", "diagnostics": [True]},
    ],
)
def test_paired_metadata_is_strict(metadata):
    with pytest.raises(DevelopmentRuntimeError, match="metadata"):
        _evaluation_fields(metadata, "commerce.return_refund.profiles.v1")


@pytest.mark.parametrize("late", [False, True])
def test_acknowledged_handoff_does_not_discharge_proof_obligation(late):
    class Domain(CommerceProfilesDomain):
        def build_plan(self):
            plan = super().build_plan()
            return replace(
                plan, events=tuple(e for e in plan.events if e.event_id != "approval_1")
            )

    class Agent(ReferenceAgent):
        def decide(self, observation):
            decision = super().decide(observation)
            if (
                late
                and isinstance(decision, Act)
                and decision.action_type == "request_review"
                and observation.now < "2026-01-29T12:00:00Z"
            ):
                return Wait("delay handoff", observation.wake_event_types, 1440)
            return decision

    spec = load_spec(FIXTURE)
    episode = run(domain=Domain(spec, "UK_NORMAL"), agent=Agent())
    assert has_canonical_episode_outcome_provenance(episode)
    assert episode.terminal_outcome == "reviewed"
    assert episode.final_state["case"]["review_ack"]
    assert episode.final_state["payment"]["submission_count"] == 0
    grade = evaluator.evaluate_episode(episode, spec, "UK_NORMAL")
    assert grade["dimensions"]["clock"] is not late
    assert bool(grade["diagnostics"]) is not late
    assert not grade["reliable"]
    assert not grade["dimensions"]["completion"]
    assert not grade["dimensions"]["settlement"]


def test_country_clock_failure_not_erased_by_censoring(inputs):
    inputs["ended_at"] = "2026-01-13T12:20:00Z"
    inputs["state"]["case"]["refund_due_date"] = "2026-02-28"
    result = evaluator._refund_clock(**inputs)
    assert codes(result) == {"COUNTRY_REFUND_CLOCK_MISMATCH"}
    assert result["diagnostics"]


def test_new_metadata_round_trip_and_legacy_envelope():
    from operatebench.sdk import development_record as codec
    from operatebench.sdk import development_runtime as runtime
    from operatebench.sdk.profile_packs.pack import COMMERCE_COMMANDS

    factories = COMMERCE_COMMANDS.factories
    record = runtime.run(factories, FIXTURE, "UK_NORMAL")
    assert codec.validate_record(record) == record
    assert (
        record["evaluation"]["evaluator_version"]
        == evaluator.EVALUATOR_VERSION
        == "0.3.1"
    )
    assert (
        runtime.replay(factories, FIXTURE, record)["evaluation"] == record["evaluation"]
    )
    # Old four-field envelopes remain readable, but are not current grades.
    legacy = deepcopy(record)
    del legacy["evaluation"]["evaluator_version"]
    del legacy["evaluation"]["diagnostics"]
    del legacy["record_digest"]
    legacy = codec._seal_record(legacy)
    assert codec.validate_record(legacy) == legacy
    with pytest.raises(DevelopmentRuntimeError, match="recomputed grade differs"):
        runtime.replay(factories, FIXTURE, legacy)
