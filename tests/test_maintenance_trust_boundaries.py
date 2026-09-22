"""Maintenance trust-boundary regressions: attacks, not implementation guards.

Every test here is an *attack* on the Lifecycle vertical written from outside
the implementation. The adversarial inputs target false reliability, incorrect
exit status, symlink traversal and unhandled errors so each trust-boundary
invariant is directly executable.

The attacks are grouped by the trust boundary they cross:

* the authored spec (immutability, authority, payload schema, identity);
* the authoritative event (settlement binding to one exact payment);
* the recorded run (causal and structural forgery of a trajectory);
* the artefact file (schema, complete replay comparison, write-once path).
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
import yaml

from operatebench.artifact import (
    ARTIFACT_NOTE,
    artifact_text,
    build_artifact,
    read_artifact,
    replay_artifact,
    replay_file,
    write_artifact,
)
from operatebench.cli import main
from operatebench.core.errors import (
    ArtifactError,
    SpecIdentityError,
    SpecSchemaError,
    UnknownTypeError,
)
from operatebench.core.events import Event
from operatebench.domains.lettings.maintenance.evaluator import evaluate
from operatebench.domains.lettings.maintenance.operation import (
    _EVENT_REDUCERS,
    MaintenanceOperation,
)
from operatebench.domains.lettings.maintenance.spec import (
    MAINTENANCE_EVENT_TYPES,
    ActorSpec,
    OperationSpec,
    load_spec,
)
from operatebench.runner import EpisodeRun, run_episode

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "examples" / "operatebench" / "maintenance_v0_1.yaml"


@pytest.fixture(scope="module")
def spec() -> OperationSpec:
    return load_spec(FIXTURE)


@pytest.fixture(scope="module")
def v1(spec: OperationSpec) -> EpisodeRun:
    return run_episode(spec, "V1", "reference")


@pytest.fixture(scope="module")
def run_v1(v1: EpisodeRun) -> EpisodeRun:
    """The same executed episode, named for the artefact-writing attacks."""
    return v1


@pytest.fixture()
def artifact(run_v1: EpisodeRun, tmp_path: Path) -> Path:
    path = tmp_path / "written" / "v1_reference.json"
    path.parent.mkdir(parents=True)
    path.write_text(artifact_text(run_v1) + "\n", encoding="utf-8")
    return path


def raw_spec() -> dict[str, Any]:
    """A fresh mutable copy of the shipped fixture, for authoring attacks."""
    loaded = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def build(raw: dict[str, Any]) -> OperationSpec:
    return OperationSpec.from_mapping(raw, source="<attack under test>")


def event_of(raw: dict[str, Any], scenario_id: str, event_id: str) -> dict[str, Any]:
    for event in raw["scenarios"][scenario_id]["events"]:
        if event["event_id"] == event_id:
            body: dict[str, Any] = event
            return body
    raise AssertionError(f"no authored event {event_id!r} in {scenario_id}")


# ------------------------------------------------- schema honesty: no ghost type


class TestDeclaredVocabularyIsExecutable:
    """A type the spec accepts and the engine cannot run is a false contract."""

    def test_every_accepted_event_type_has_an_executable_reducer(self) -> None:
        assert set(MAINTENANCE_EVENT_TYPES) == set(_EVENT_REDUCERS)

    def test_the_unimplemented_visit_status_type_is_refused_at_the_door(self) -> None:
        raw = raw_spec()
        event_of(raw, "V1", "v1_e02")["type"] = "supplier_visit_status"
        with pytest.raises(UnknownTypeError, match="supplier_visit_status"):
            build(raw)


# ----------------------------------------------------- per-event payload schema


class TestAuthoredPayloadSchema:
    """A payload the reducers index into is checked when the spec loads.

    A spec missing ``cycle_id`` is refused during validation rather than raising
    a raw ``KeyError`` from inside a reducer.
    """

    def test_a_missing_required_payload_field_is_refused_at_load(self) -> None:
        raw = raw_spec()
        del event_of(raw, "V1", "v1_e01")["payload"]["cycle_id"]
        with pytest.raises(SpecSchemaError, match="cycle_id"):
            build(raw)

    def test_an_unknown_nested_payload_field_is_refused_at_load(self) -> None:
        raw = raw_spec()
        event_of(raw, "V1", "v1_e01")["payload"]["escalate_to"] = "root"
        with pytest.raises(SpecSchemaError, match="escalate_to"):
            build(raw)

    def test_a_string_where_an_integer_is_required_is_refused(self) -> None:
        raw = raw_spec()
        event_of(raw, "V1", "v1_e05")["payload"]["amount_minor"] = "64000"
        with pytest.raises(SpecSchemaError, match="amount_minor"):
            build(raw)

    def test_a_boolean_is_not_an_integer(self) -> None:
        raw = raw_spec()
        event_of(raw, "V1", "v1_e05")["payload"]["quote_version"] = True
        with pytest.raises(SpecSchemaError, match="quote_version"):
            build(raw)

    def test_an_integer_is_not_a_boolean(self) -> None:
        raw = raw_spec()
        event_of(raw, "V1", "v1_e03")["payload"]["attended"] = 1
        with pytest.raises(SpecSchemaError, match="attended"):
            build(raw)

    def test_a_container_where_a_primitive_is_required_is_refused(self) -> None:
        raw = raw_spec()
        event_of(raw, "V1", "v1_e01")["payload"]["issue_id"] = ["issue_1"]
        with pytest.raises(SpecSchemaError, match="issue_id"):
            build(raw)

    def test_the_shipped_fixture_still_satisfies_every_payload_schema(self) -> None:
        spec = load_spec(FIXTURE)
        assert spec.scenario_ids == ("V1", "V2", "V3")


# ----------------------------------------- authoritative settlement binding


def run_v1_with_settlement(**overrides: Any) -> EpisodeRun:
    """Run V1 with the authoritative settlement event's payload overridden."""
    raw = raw_spec()
    event_of(raw, "V1", "v1_e12")["payload"].update(overrides)
    return run_episode(build(raw), "V1", "reference")


def run_v1_with_unattributed_settlement(**overrides: Any) -> EpisodeRun:
    """The same, for a settlement that is *not* caused by the payment request.

    ``payment_request_id`` is the one bound field an authored event cannot
    simply be given a wrong value for while still hanging off the request it
    would be wrong about: a conditional event is bound to the identity its cause
    established, so authoring ``payment_SUPERSEDED`` against
    ``on_action: request_payment`` describes a settlement of *that* request
    under another name, which is not a thing. To author a settlement that names
    a payment this operation never made, the settlement has to come from
    somewhere else — so here it hangs off the invoice validation instead, and
    arrives after the request the reference makes in response to it.
    """
    raw = raw_spec()
    event = event_of(raw, "V1", "v1_e12")
    event.pop("on_action")
    event["on_event"] = {"type": "invoice_validation_completed"}
    event["delay_minutes"] = 120
    event["payload"].update(overrides)
    return run_episode(build(raw), "V1", "reference")


def settlement_delivery(run: EpisodeRun) -> Mapping[str, Any]:
    for event in run.outcome.events:
        if event["event_id"] == "v1_e12":
            return event
    raise AssertionError("the settlement event was never delivered")


class TestSettlementBindsToOneExactPayment:
    """Money is the irreversible step, so its confirmation binds to everything.

    A settlement with any mismatched request, invoice, cycle, amount, or currency
    is rejected without mutating payment state.
    """

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("invoice_id", "invoice_WRONG"),
            ("cycle_id", "work_cycle_1"),
            ("amount_minor", 1),
            ("currency", "USD"),
        ],
    )
    def test_a_mismatch_in_any_bound_field_is_a_named_refusal(
        self, field_name: str, value: Any
    ) -> None:
        run = run_v1_with_settlement(**{field_name: value})
        delivery = settlement_delivery(run)
        assert delivery["disposition"] == "rejected"
        assert delivery["verdict_code"] in {
            "SETTLEMENT_BINDING_MISMATCH",
            "PAYMENT_REQUEST_MISMATCH",
        }
        assert not run.reliable

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("invoice_id", "invoice_WRONG"),
            ("cycle_id", "work_cycle_1"),
            ("amount_minor", 1),
            ("currency", "USD"),
        ],
    )
    def test_a_refused_settlement_mutates_nothing(
        self, field_name: str, value: Any
    ) -> None:
        run = run_v1_with_settlement(**{field_name: value})
        payment = run.outcome.final_state["payment"]
        assert payment["status"] == "REQUESTED"
        assert payment["settlement_id"] is None

    def test_a_settlement_naming_another_payment_request_is_refused(self) -> None:
        run = run_v1_with_unattributed_settlement(payment_request_id="payment_SUPERSEDED")
        delivery = settlement_delivery(run)
        assert delivery["disposition"] == "rejected"
        assert delivery["verdict_code"] == "PAYMENT_REQUEST_MISMATCH"
        payment = run.outcome.final_state["payment"]
        assert payment["status"] == "REQUESTED"
        assert payment["settlement_id"] is None
        assert not run.reliable
        assert "settlement_1" not in run.outcome.final_state["authoritative_records"]

    def test_positive_control_the_exact_settlement_still_settles(self) -> None:
        run = run_v1_with_settlement()
        delivery = settlement_delivery(run)
        assert delivery["disposition"] == "accepted"
        payment = run.outcome.final_state["payment"]
        assert payment["status"] == "SETTLED"
        assert payment["settlement_id"] == "settlement_1"
        assert run.reliable

    def test_the_settlement_record_carries_every_field_it_bound(self) -> None:
        run = run_v1_with_settlement()
        record = run.outcome.final_state["authoritative_records"]["settlement_1"]
        assert record["payment_request_id"] == "payment_1"
        assert record["invoice_id"] == "invoice_1"
        assert record["cycle_id"] == "work_cycle_2"
        assert record["amount_minor"] == 64000
        assert record["currency"] == "GBP"


class TestEvaluatorRederivesSettlementBinding:
    """The evaluator does not take the reducer's word that the binding held.

    It is handed a final state in which payment is SETTLED and the recorded
    authoritative settlement disagrees with the invoice it claims to have paid —
    the exact residue a broken or bypassed guard would leave — and must report a
    critical violation rather than a reliable episode.
    """

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("invoice_id", "invoice_WRONG"),
            ("cycle_id", "work_cycle_1"),
            ("amount_minor", 1),
            ("currency", "USD"),
            ("payment_request_id", "payment_SUPERSEDED"),
        ],
    )
    def test_a_doctored_settlement_record_is_a_critical_violation(
        self, field_name: str, value: Any
    ) -> None:
        spec = load_spec(FIXTURE)
        run = run_episode(spec, "V1", "reference")
        final_state = copy.deepcopy(dict(run.outcome.final_state))
        final_state["authoritative_records"]["settlement_1"][field_name] = value
        evaluation = evaluate(
            status=run.outcome.status,
            replay_final=run.outcome.replay_final,
            simulated_minutes=run.outcome.simulated_minutes,
            invocations=run.outcome.invocations,
            final_state=final_state,
            trajectory=run.outcome.trajectory,
            events=run.outcome.events,
            scenario=spec.scenario("V1"),
            replay_ok=True,
        )
        assert not evaluation.reliable
        assert "critical_invariants" in evaluation.failed_dimensions
        assert "CRITICAL_SETTLEMENT_BINDING_MISMATCH" in evaluation.finding_codes


# ------------------------------------ recursive spec immutability and authority


class StubEnvironment:
    """The smallest environment a reducer can be driven through directly."""

    def __init__(self, now: str = "2031-03-03T09:00:00Z") -> None:
        self.now = now
        self.records: list[tuple[str, Mapping[str, Any]]] = []

    def record(self, record_type: str, payload: Mapping[str, Any]) -> None:
        self.records.append((record_type, dict(payload)))

    def schedule_timer(self, *args: Any, **kwargs: Any) -> None:
        return None

    def cancel_timer(self, event_id: str) -> bool:
        return False

    def dispatch_fails(self, message_fixture_id: str) -> bool:
        return False


class TestSpecIsFrozenAllTheWayDown:
    """A loaded spec is the frozen artefact, not a mutable view of one.

    Actor registries and nested authored payloads reject writes, keeping the
    semantics that execute bound to ``spec_digest_sha256``.
    """

    def test_the_actor_registry_cannot_be_replaced(self) -> None:
        spec = load_spec(FIXTURE)
        with pytest.raises(TypeError):
            spec.actors["maintenance_system"] = spec.actors["supplier_1"]  # type: ignore[index]

    def test_the_scenario_registry_cannot_be_replaced(self) -> None:
        spec = load_spec(FIXTURE)
        with pytest.raises(TypeError):
            spec.scenarios["V1"] = spec.scenarios["V2"]  # type: ignore[index]

    def test_hidden_environment_truth_cannot_be_rewritten(self) -> None:
        spec = load_spec(FIXTURE)
        with pytest.raises(TypeError):
            spec.hidden_state["root_cause_fixture_id"] = "hidden_fixture_other"  # type: ignore[index]

    def test_message_fixtures_cannot_be_rewritten(self) -> None:
        spec = load_spec(FIXTURE)
        with pytest.raises(TypeError):
            spec.message_fixtures["msg_completion_notice"] = "anything at all"  # type: ignore[index]

    def test_a_nested_authored_payload_cannot_be_rewritten(self) -> None:
        event = load_spec(FIXTURE).scenario("V1").events[0]
        with pytest.raises(TypeError):
            event.payload["cycle_id"] = "work_cycle_9"  # type: ignore[index]

    def test_the_authored_event_sequence_cannot_be_reordered(self) -> None:
        scenario = load_spec(FIXTURE).scenario("V1")
        with pytest.raises(TypeError):
            scenario.events[0] = scenario.events[1]  # type: ignore[index]

    def test_an_actor_cannot_be_granted_an_authority_after_loading(self) -> None:
        spec = load_spec(FIXTURE)
        with pytest.raises((AttributeError, TypeError)):
            spec.actors["agent"].authority.add("confirm_settlement")  # type: ignore[attr-defined]
        with pytest.raises((AttributeError, TypeError)):
            spec.actors["agent"].authority = frozenset({"confirm_settlement"})  # type: ignore[misc]

    def test_a_policy_constant_cannot_be_rewritten(self) -> None:
        spec = load_spec(FIXTURE)
        with pytest.raises((AttributeError, TypeError)):
            spec.policy.approval_threshold_minor = 1  # type: ignore[misc]

    def test_a_conditional_trigger_cannot_be_repointed(self) -> None:
        events = load_spec(FIXTURE).scenario("V1").events
        conditional = next(event for event in events if event.trigger is not None)
        assert conditional.trigger is not None
        with pytest.raises((AttributeError, TypeError)):
            conditional.trigger.cycle_id = "work_cycle_9"  # type: ignore[misc]


class TestPublicProjectionsAreDetached:
    """What a caller is handed is a copy, so reading cannot become writing."""

    def test_the_identity_projection_is_plain_and_detached(self) -> None:
        spec = load_spec(FIXTURE)
        payload = spec.identity_payload()
        payload["scenario_ids"].append("V9")
        payload["operation_id"] = "something_else"
        assert spec.identity_payload()["scenario_ids"] == ["V1", "V2", "V3"]
        assert spec.operation_id == "lettings_maintenance_synthetic_v1"

    def test_an_authored_payload_snapshot_is_plain_and_detached(self) -> None:
        spec = load_spec(FIXTURE)
        event = spec.scenario("V1").events[0]
        snapshot = event.payload_snapshot()
        assert isinstance(snapshot, dict)
        snapshot["cycle_id"] = "work_cycle_9"
        assert spec.scenario("V1").events[0].payload["cycle_id"] == "work_cycle_1"

    def test_the_hidden_state_snapshot_is_plain_and_detached(self) -> None:
        spec = load_spec(FIXTURE)
        snapshot = spec.hidden_state_snapshot()
        assert isinstance(snapshot, dict)
        snapshot["root_cause_fixture_id"] = "hidden_fixture_other"
        assert (
            spec.hidden_state["root_cause_fixture_id"] == "hidden_fixture_perished_seal"
        )


class TestRuntimeRechecksActorAuthority:
    """Load-time authority is a claim about the file, not about the event.

    The reducer is driven directly with an event whose type requires an
    authority its actor does not hold. Nothing in the spec file authorises this,
    but nothing in the spec file is what the engine is holding at that moment
    either — so the check has to happen where the mutation would.
    """

    def test_an_event_whose_actor_lacks_the_authority_is_refused(self) -> None:
        spec = load_spec(FIXTURE)
        domain = MaintenanceOperation(spec, "V1")
        state = domain.initial_state()
        forged = Event(
            event_id="forged_1",
            event_type="work_evidence_verified",
            actor_id="supplier_1",
            at="2031-03-03T09:00:00Z",
            sequence=0,
            payload={
                "evidence_id": "evidence_9",
                "visit_id": "visit_1",
                "cycle_id": "work_cycle_1",
                "scope_digest": "a" * 64,
                "outcome": "COMPLETED",
            },
        )
        verdict = domain.reduce_event(state, forged, StubEnvironment())
        assert not verdict.accepted
        assert verdict.code == "EVENT_ACTOR_LACKS_AUTHORITY"
        assert state.authoritative_records == {}

    def test_a_malformed_event_payload_is_refused_not_raised(self) -> None:
        spec = load_spec(FIXTURE)
        domain = MaintenanceOperation(spec, "V1")
        state = domain.initial_state()
        forged = Event(
            event_id="forged_2",
            event_type="customer_issue_reported",
            actor_id="customer_1",
            at="2031-03-03T09:00:00Z",
            sequence=0,
            payload={
                "issue_id": "issue_1",
                "classification": "NON_EMERGENCY_RECURRING_LEAK",
            },
        )
        verdict = domain.reduce_event(state, forged, StubEnvironment())
        assert not verdict.accepted
        assert verdict.code == "MALFORMED_EVENT_PAYLOAD"
        assert state.issue_id is None


class TestIdentityIsRevalidatedBeforeAnEpisodeRuns:
    """A digest computed at load cannot bind semantics changed after it.

    The attack replaces the whole actor registry on a loaded spec, which leaves
    ``spec_digest_sha256`` exactly as it was. The episode must refuse before it
    executes anything rather than run changed semantics under a stale identity.
    """

    def test_a_replaced_actor_registry_is_caught_before_execution(self) -> None:
        spec = load_spec(FIXTURE)
        registry = dict(spec.actors)
        registry["maintenance_system"] = ActorSpec(
            actor_id="maintenance_system",
            role="supplier",
            authority=frozenset({"verify_attendance", "verify_work", "report_work"}),
        )
        object.__setattr__(spec, "actors", MappingProxyType(registry))
        assert spec.spec_digest_sha256 == load_spec(FIXTURE).spec_digest_sha256
        with pytest.raises(SpecIdentityError):
            run_episode(spec, "V1", "reference")

    def test_a_replaced_hidden_state_is_caught_before_execution(self) -> None:
        spec = load_spec(FIXTURE)
        object.__setattr__(
            spec, "hidden_state", MappingProxyType({"root_cause_fixture_id": "other"})
        )
        with pytest.raises(SpecIdentityError):
            run_episode(spec, "V1", "reference")

    def test_an_untampered_spec_verifies_and_runs(self) -> None:
        spec = load_spec(FIXTURE)
        spec.verify_identity()
        assert run_episode(spec, "V1", "reference").reliable


# ------------------------------ evaluator causal and structural independence


def evaluation_of(
    run: EpisodeRun,
    spec: OperationSpec,
    scenario_id: str,
    *,
    trajectory: Any = None,
    events: Any = None,
    final_state: Any = None,
) -> Any:
    """Re-evaluate a real run with one part of its record replaced."""
    return evaluate(
        status=run.outcome.status,
        replay_final=run.outcome.replay_final,
        simulated_minutes=run.outcome.simulated_minutes,
        invocations=run.outcome.invocations,
        final_state=run.outcome.final_state if final_state is None else final_state,
        trajectory=run.outcome.trajectory if trajectory is None else trajectory,
        events=run.outcome.events if events is None else events,
        scenario=spec.scenario(scenario_id),
        replay_ok=True,
    )


def forged_ordering(run: EpisodeRun) -> list[dict[str, Any]]:
    """A jointly coherent trajectory in which the money came first.

    The two irreversible accepted effects are moved to the front, indices are
    renumbered contiguously and the moved rows are re-stamped with the episode's
    opening instant, so the forgery survives every *structural* check: indices
    run 0..n-1, instants are canonical and non-decreasing, and every event row
    still binds exactly to the delivery it records. The only thing wrong with it
    is the order in which things were allowed to happen.
    """
    rows = [dict(record) for record in run.outcome.trajectory]
    irreversible = {"authorise_supplier_work", "request_payment"}
    moved = [
        row
        for row in rows
        if row.get("record_type") == "effect_accepted"
        and row.get("action_type") in irreversible
    ]
    assert len(moved) == 2, "V1 must commit both irreversible effects"
    remaining = [row for row in rows if row not in moved]
    opening = remaining[0]["at"]
    for row in moved:
        row["at"] = opening
    forged = [remaining[0], *moved, *remaining[1:]]
    for position, row in enumerate(forged):
        row["index"] = position
    return forged


class TestEvaluatorTreatsTheRecordAsUntrusted:
    """A trajectory is evidence offered to the evaluator, not testimony it takes.

    Moving accepted irreversible effects before the approval, work evidence, and
    invoice validation they require must fail causal checks even when final state
    is otherwise correct.
    """

    def test_the_valid_reference_runs_stay_reliable(self, spec: OperationSpec) -> None:
        for scenario_id in ("V1", "V2", "V3"):
            run = run_episode(spec, scenario_id, "reference")
            assert run.reliable is (scenario_id != "V2")
            assert run.evaluation.failed_dimensions == (
                ("recovery", "obligations") if scenario_id == "V2" else ()
            )

    def test_an_ordering_forgery_is_a_critical_failure(
        self, spec: OperationSpec, v1: EpisodeRun
    ) -> None:
        evaluation = evaluation_of(v1, spec, "V1", trajectory=forged_ordering(v1))
        assert not evaluation.reliable
        assert "critical_invariants" in evaluation.failed_dimensions
        assert {
            "CRITICAL_WORK_AUTHORISED_WITHOUT_PRIOR_APPROVAL",
            "CRITICAL_PAYMENT_WITHOUT_PRIOR_EVIDENCE",
        } <= set(evaluation.finding_codes)

    def test_the_forgery_is_structurally_clean_so_only_causality_catches_it(
        self, v1: EpisodeRun
    ) -> None:
        forged = forged_ordering(v1)
        assert [row["index"] for row in forged] == list(range(len(forged)))
        instants = [row["at"] for row in forged]
        assert instants == sorted(instants)

    def test_a_correct_final_state_does_not_repair_an_earlier_invalid_action(
        self, spec: OperationSpec, v1: EpisodeRun
    ) -> None:
        evaluation = evaluation_of(v1, spec, "V1", trajectory=forged_ordering(v1))
        payment = v1.outcome.final_state["payment"]
        assert payment["status"] == "SETTLED"
        assert not evaluation.reliable

    def test_non_contiguous_indices_are_refused(
        self, spec: OperationSpec, v1: EpisodeRun
    ) -> None:
        rows = [dict(record) for record in v1.outcome.trajectory]
        del rows[5]
        evaluation = evaluation_of(v1, spec, "V1", trajectory=rows)
        assert not evaluation.reliable
        assert "temporal_correctness" in evaluation.failed_dimensions
        assert "TRAJECTORY_INDICES_NOT_CONTIGUOUS" in evaluation.finding_codes

    def test_a_non_canonical_instant_is_refused(
        self, spec: OperationSpec, v1: EpisodeRun
    ) -> None:
        rows = [dict(record) for record in v1.outcome.trajectory]
        rows[4]["at"] = "2031-03-03 09:00:00+00:00"
        evaluation = evaluation_of(v1, spec, "V1", trajectory=rows)
        assert not evaluation.reliable
        assert "TRAJECTORY_TIMESTAMP_NOT_CANONICAL" in evaluation.finding_codes

    def test_a_rewound_instant_is_refused(
        self, spec: OperationSpec, v1: EpisodeRun
    ) -> None:
        rows = [dict(record) for record in v1.outcome.trajectory]
        rows[-2]["at"] = "2030-01-01T00:00:00Z"
        evaluation = evaluation_of(v1, spec, "V1", trajectory=rows)
        assert not evaluation.reliable
        assert "TRAJECTORY_TIMESTAMPS_NOT_MONOTONIC" in evaluation.finding_codes

    def test_an_event_row_that_does_not_bind_to_a_delivery_is_refused(
        self, spec: OperationSpec, v1: EpisodeRun
    ) -> None:
        rows = [dict(record) for record in v1.outcome.trajectory]
        observed = next(row for row in rows if row.get("record_type") == "event_observed")
        observed["event_type"] = "payment_settlement_confirmed"
        evaluation = evaluation_of(v1, spec, "V1", trajectory=rows)
        assert not evaluation.reliable
        assert "CRITICAL_EVENT_LEDGER_BINDING_MISMATCH" in evaluation.finding_codes

    def test_a_delivered_event_with_no_ledger_record_is_refused(
        self, spec: OperationSpec, v1: EpisodeRun
    ) -> None:
        events = [dict(event) for event in v1.outcome.events]
        events.append({**events[0], "event_id": "smuggled_1"})
        evaluation = evaluation_of(v1, spec, "V1", events=events)
        assert not evaluation.reliable
        assert "CRITICAL_EVENT_LEDGER_BINDING_MISMATCH" in evaluation.finding_codes

    def test_a_work_authorisation_with_no_approved_quote_is_refused(
        self, spec: OperationSpec, v1: EpisodeRun
    ) -> None:
        final_state = copy.deepcopy(dict(v1.outcome.final_state))
        final_state["quotes"]["quote_1:v1"]["status"] = "RECEIVED"
        evaluation = evaluation_of(v1, spec, "V1", final_state=final_state)
        assert not evaluation.reliable
        assert (
            "CRITICAL_WORK_AUTHORISED_WITHOUT_PRIOR_APPROVAL" in evaluation.finding_codes
        )

    def test_a_payment_whose_invoice_validation_never_arrived_is_refused(
        self, spec: OperationSpec, v1: EpisodeRun
    ) -> None:
        events = [dict(event) for event in v1.outcome.events]
        rows = [
            dict(record)
            for record in v1.outcome.trajectory
            if record.get("event_id") != "v1_e11"
        ]
        events = [event for event in events if event["event_id"] != "v1_e11"]
        for position, row in enumerate(rows):
            row["index"] = position
        evaluation = evaluation_of(v1, spec, "V1", trajectory=rows, events=events)
        assert not evaluation.reliable
        assert "CRITICAL_PAYMENT_WITHOUT_PRIOR_EVIDENCE" in evaluation.finding_codes


# ------------------------------------- the artefact file as a trust boundary


def written(run: EpisodeRun, path: Path) -> Path:
    return write_artifact(run, path)


def tampered(source: Path, target: Path, **fields: Any) -> Path:
    """A schema-valid artefact with one result-bearing field changed."""
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload.update(fields)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


class TestArtifactSchemaIsStrict:
    """An artefact is untrusted input, and a reader that indexes it says so.

    Wrong container types and unknown fields are structured ``ArtifactError``
    refusals rather than raw attribute errors.
    """

    def test_a_valid_artefact_round_trips(self, artifact: Path) -> None:
        payload = read_artifact(artifact)
        assert payload["scenario_id"] == "V1"
        assert payload["agent_kind"] == "reference"

    def test_an_unknown_top_level_field_is_refused(
        self, artifact: Path, tmp_path: Path
    ) -> None:
        path = tampered(artifact, tmp_path / "unknown.json", leaderboard_rank=1)
        with pytest.raises(ArtifactError, match="leaderboard_rank"):
            read_artifact(path)

    def test_a_missing_top_level_field_is_refused(
        self, artifact: Path, tmp_path: Path
    ) -> None:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        del payload["evaluation"]
        path = tmp_path / "missing.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ArtifactError, match="evaluation"):
            read_artifact(path)

    def test_a_list_where_a_mapping_is_required_is_refused_by_name(
        self, artifact: Path, tmp_path: Path
    ) -> None:
        path = tampered(artifact, tmp_path / "wrong_container.json", operation=[])
        with pytest.raises(ArtifactError, match="operation"):
            read_artifact(path)

    def test_a_mapping_where_a_list_is_required_is_refused(
        self, artifact: Path, tmp_path: Path
    ) -> None:
        path = tampered(artifact, tmp_path / "wrong_list.json", trajectory={})
        with pytest.raises(ArtifactError, match="trajectory"):
            read_artifact(path)

    def test_a_boolean_counter_is_refused(self, artifact: Path, tmp_path: Path) -> None:
        path = tampered(artifact, tmp_path / "bool_counter.json", agent_invocations=True)
        with pytest.raises(ArtifactError, match="agent_invocations"):
            read_artifact(path)

    def test_a_negative_counter_is_refused(self, artifact: Path, tmp_path: Path) -> None:
        path = tampered(artifact, tmp_path / "negative.json", simulated_minutes=-1)
        with pytest.raises(ArtifactError, match="simulated_minutes"):
            read_artifact(path)

    def test_a_malformed_timestamp_is_refused(
        self, artifact: Path, tmp_path: Path
    ) -> None:
        path = tampered(artifact, tmp_path / "clock.json", started_at="yesterday")
        with pytest.raises(ArtifactError, match="started_at"):
            read_artifact(path)

    def test_a_malformed_digest_is_refused(self, artifact: Path, tmp_path: Path) -> None:
        path = tampered(
            artifact, tmp_path / "digest.json", trajectory_digest_sha256="NOTADIGEST"
        )
        with pytest.raises(ArtifactError, match="trajectory_digest_sha256"):
            read_artifact(path)

    def test_an_unknown_nested_field_is_refused(
        self, artifact: Path, tmp_path: Path
    ) -> None:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        payload["operation"]["secret_weighting"] = 0.5
        path = tmp_path / "nested.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ArtifactError, match="secret_weighting"):
            read_artifact(path)

    def test_a_malformed_event_record_is_refused(
        self, artifact: Path, tmp_path: Path
    ) -> None:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        payload["events"][0]["disposition"] = "probably_fine"
        path = tmp_path / "event.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ArtifactError, match="disposition"):
            read_artifact(path)

    def test_a_malformed_trajectory_record_is_refused(
        self, artifact: Path, tmp_path: Path
    ) -> None:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        payload["trajectory"][3]["record_type"] = "definitely_a_record"
        path = tmp_path / "row.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ArtifactError, match="record_type"):
            read_artifact(path)

    def test_an_escaped_lone_surrogate_is_refused(
        self, artifact: Path, tmp_path: Path
    ) -> None:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        payload["scenario_label"] = "leak \\ud800 report".replace("\\ud800", "\ud800")
        path = tmp_path / "surrogate.json"
        path.write_bytes(
            json.dumps(payload, ensure_ascii=True).encode("ascii", "backslashreplace")
        )
        with pytest.raises(ArtifactError):
            read_artifact(path)

    def test_an_artefact_that_contains_itself_is_refused(self, spec: Any) -> None:
        run = run_episode(spec, "V1", "reference")
        payload = build_artifact(run)
        payload["final_state"] = payload
        with pytest.raises(ArtifactError):
            replay_artifact(spec, payload)

    def test_every_field_is_validated_before_identity_is_compared(
        self, artifact: Path, tmp_path: Path, spec: Any
    ) -> None:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        payload["operation"]["spec_digest_sha256"] = "0" * 64
        payload["agent_invocations"] = "several"
        path = tmp_path / "both_wrong.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ArtifactError, match="agent_invocations"):
            replay_file(spec, path)


class TestReplayComparesTheWholeResult:
    """Replay rebuilds the artefact the rerun would write and compares all of it.

    Changes to status, expected terminal, agent kind, timestamps, counters, or
    delivered events make replay report a mismatch.
    """

    def test_an_untouched_artefact_replays_clean(self, spec: Any, artifact: Path) -> None:
        report = replay_file(spec, artifact)
        assert report.ok
        assert report.differences == ()

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("status", "operational_horizon_exhausted"),
            ("terminal_outcome", "transferred_to_human_ownership"),
            # A terminal class this build accepts, but not the one V1 authored:
            # the deferred classes are no longer loadable at all, which is a
            # separate refusal asserted in the construct-integrity regressions.
            ("expected_terminal", "transferred_to_human_ownership"),
            ("agent_kind", "negative"),
            ("scenario_label", "a different story entirely"),
            ("started_at", "2031-03-04T09:00:00Z"),
            ("ended_at", "2031-03-30T09:00:00Z"),
            ("simulated_minutes", 7),
            ("agent_invocations", 99),
            ("replay_final", False),
            ("note", "these digests are cryptographic tamper evidence"),
        ],
    )
    def test_one_changed_field_makes_the_replay_report_not_ok(
        self, spec: Any, artifact: Path, tmp_path: Path, field_name: str, value: Any
    ) -> None:
        path = tampered(artifact, tmp_path / f"{field_name}.json", **{field_name: value})
        report = replay_file(spec, path)
        assert not report.ok
        assert any(field_name in difference for difference in report.differences)

    def test_a_changed_engine_is_refused_before_reproduction(
        self, spec: Any, artifact: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from operatebench import artifact as artifact_module

        path = tampered(
            artifact, tmp_path / "engine_version.json", engine_version="0.0.1"
        )

        def forbidden(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("different engine reached reproduction")

        monkeypatch.setattr(artifact_module, "_reproduce", forbidden)
        # Relabelling current state as historical fails its exact state schema
        # before replay admission. A structurally historical state reaches the
        # independent engine refusal; neither path may execute this runtime.
        with pytest.raises(
            ArtifactError, match=r"unknown field.*issue_reporting_actor_id"
        ):
            replay_file(spec, path)
        historical = (
            REPO_ROOT / "tests/fixtures/b3/artifact8-deterministic-reference-v1.json"
        )
        path = tampered(
            historical, tmp_path / "historical_engine.json", engine_version="0.0.1"
        )
        with pytest.raises(ArtifactError, match="records engine version '0\\.0\\.1'"):
            replay_file(spec, path)

    def test_a_changed_delivered_event_is_detected(
        self, spec: Any, artifact: Path, tmp_path: Path
    ) -> None:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        payload["events"][0]["verdict_code"] = "TOTALLY_FINE"
        path = tmp_path / "events.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        report = replay_file(spec, path)
        assert not report.ok
        assert not report.events_match
        assert any("events" in difference for difference in report.differences)

    def test_the_report_says_which_sections_matched(
        self, spec: Any, artifact: Path
    ) -> None:
        payload = replay_file(spec, artifact).as_dict()
        for section in (
            "identity",
            "metadata",
            "events",
            "trajectory",
            "final_state",
            "evaluation",
        ):
            assert payload["sections"][section]["matches"] is True
        assert payload["sections"]["identity"]["fields"]["agent_kind"] is True

    def test_the_report_claims_content_equality_and_not_tamper_evidence(self) -> None:
        assert "not tamper-evidence" in ARTIFACT_NOTE
        assert "make no" in ARTIFACT_NOTE
        assert "cryptographic claim" in ARTIFACT_NOTE

    def test_a_changed_field_exits_two_through_the_cli(
        self, artifact: Path, tmp_path: Path
    ) -> None:
        path = tampered(artifact, tmp_path / "cli.json", agent_invocations=99)
        code = main(["replay", "--spec", str(FIXTURE), "--run", str(path)])
        assert code == 2


@pytest.mark.usefixtures("require_openat2")
class TestArtifactWriteIsSafeAndWriteOnce:
    """An artefact path is created, never followed and never overwritten.

    A dangling or existing symlink is refused without creating or modifying its
    target.
    """

    def test_an_ancestor_swapped_for_a_symlink_at_the_seam_cannot_redirect_the_artifact(
        self,
        run_v1: EpisodeRun,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        output = tmp_path / "output"
        output.mkdir()
        attacker = tmp_path / "attacker"
        attacker.mkdir()
        moved = tmp_path / "output-as-validated"
        real_open = os.open
        state = {"swapped": False}

        def attacking_open(
            path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
        ) -> int:
            if flags & os.O_CREAT and not state["swapped"]:
                state["swapped"] = True
                os.rename(output, moved)
                os.symlink(attacker, output)
            return real_open(path, flags, mode, dir_fd=dir_fd)

        monkeypatch.setattr(os, "open", attacking_open)
        with pytest.raises(ArtifactError):
            write_artifact(run_v1, output / "run.json")
        monkeypatch.undo()

        assert state["swapped"]
        assert not (output / "run.json").exists()
        assert not (attacker / "run.json").exists()
        residue = moved / "run.json"
        assert residue.exists()
        with pytest.raises(ArtifactError, match="written once"):
            write_artifact(run_v1, residue)

    def test_a_dangling_symlink_is_refused_and_its_target_is_not_created(
        self, run_v1: EpisodeRun, tmp_path: Path
    ) -> None:
        target = tmp_path / "elsewhere.json"
        link = tmp_path / "run.json"
        link.symlink_to(target)
        with pytest.raises(ArtifactError, match="is a symlink"):
            write_artifact(run_v1, link)
        assert not target.exists()

    def test_a_symlink_to_an_existing_file_is_refused(
        self, run_v1: EpisodeRun, tmp_path: Path
    ) -> None:
        target = tmp_path / "existing.json"
        target.write_text("{}", encoding="utf-8")
        link = tmp_path / "run.json"
        link.symlink_to(target)
        with pytest.raises(ArtifactError, match="is a symlink"):
            write_artifact(run_v1, link)
        assert target.read_text(encoding="utf-8") == "{}"

    def test_a_symlinked_ancestor_directory_is_refused(
        self, run_v1: EpisodeRun, tmp_path: Path
    ) -> None:
        real = tmp_path / "real_runs"
        real.mkdir()
        link = tmp_path / "runs"
        link.symlink_to(real, target_is_directory=True)
        with pytest.raises(ArtifactError, match="is a symlink"):
            write_artifact(run_v1, link / "run.json")
        assert list(real.iterdir()) == []

    def test_deleted_cwd_relative_write_is_domain_error_without_open_or_leak(
        self,
        run_v1: EpisodeRun,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        open_fd_count: Callable[[], int],
    ) -> None:
        deleted_cwd = tmp_path / "deleted-cwd"
        deleted_cwd.mkdir()
        real_open = os.open
        open_calls = 0

        def recording_open(
            path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
        ) -> int:
            nonlocal open_calls
            open_calls += 1
            return real_open(path, flags, mode, dir_fd=dir_fd)

        monkeypatch.setattr(os, "open", recording_open)
        descriptors_before = open_fd_count()
        original_cwd = Path.cwd()
        os.chdir(deleted_cwd)
        deleted_cwd.rmdir()
        try:
            with pytest.raises(ArtifactError):
                write_artifact(run_v1, Path("run.json"))
        finally:
            os.chdir(original_cwd)

        assert Path.cwd() == original_cwd
        assert open_calls == 0
        assert open_fd_count() == descriptors_before
        assert list(tmp_path.iterdir()) == []

    def test_an_existing_regular_file_is_never_overwritten(
        self, run_v1: EpisodeRun, tmp_path: Path
    ) -> None:
        path = tmp_path / "run.json"
        path.write_text("previous run", encoding="utf-8")
        with pytest.raises(ArtifactError, match="written once"):
            write_artifact(run_v1, path)
        assert path.read_text(encoding="utf-8") == "previous run"

    def test_a_fresh_path_is_written_as_canonical_utf8_bytes(
        self, run_v1: EpisodeRun, tmp_path: Path
    ) -> None:
        path = write_artifact(run_v1, tmp_path / "nested" / "run.json")
        assert path.exists() and not path.is_symlink()
        assert path.read_bytes() == (artifact_text(run_v1) + "\n").encode("utf-8")


class TestMalformedInputNeverReachesTheOperatorAsATraceback:
    """Every refusal the CLI can produce is exit 1 and a named message."""

    def test_a_malformed_spec_exits_one_without_a_traceback(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        raw = raw_spec()
        del event_of(raw, "V1", "v1_e01")["payload"]["cycle_id"]
        path = tmp_path / "broken.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        assert main(["validate", str(path)]) == 1
        captured = capsys.readouterr()
        assert "Traceback" not in captured.err
        assert captured.err.startswith("operatebench: ")

    def test_a_malformed_artefact_exits_one_without_a_traceback(
        self, artifact: Path, tmp_path: Path, capsys: Any
    ) -> None:
        path = tampered(artifact, tmp_path / "broken.json", operation=[])
        assert main(["replay", "--spec", str(FIXTURE), "--run", str(path)]) == 1
        captured = capsys.readouterr()
        assert "Traceback" not in captured.err
        assert captured.err.startswith("operatebench: ")

    @pytest.mark.usefixtures("require_openat2")
    def test_a_symlinked_output_exits_one_without_a_traceback(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        link = tmp_path / "run.json"
        link.symlink_to(tmp_path / "elsewhere.json")
        code = main(
            [
                "run-maintenance",
                "--spec",
                str(FIXTURE),
                "--scenario",
                "V1",
                "--agent",
                "reference",
                "--output",
                str(link),
            ]
        )
        assert code == 1
        captured = capsys.readouterr()
        assert "Traceback" not in captured.err
        assert "symlink" in captured.err
