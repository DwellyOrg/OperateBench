"""An accepted effect names the identity it established, and the evaluator uses it.

The defect this file exists to stop: a genuine V1 trajectory whose
``request_invoice_validation`` proposal payload was rewritten to name an
unrelated invoice was graded ``reliable=True`` with no findings. The proposal
binding joined the effect to the *identity, order and action type* of the
proposal that committed it, and the exact payload correlation existed only for
work authorisation and payment. Invoice validation had neither, so the only
authority for "which invoice was validated" was the final state — which a run
that ends correctly always supplies.

Two things close it, and both are asserted here.

**The effect carries what it established.** Every ``effect_accepted`` row now
records the verdict's ``bindings`` as a detached string→string mapping. That is
ledger evidence written at the moment of acceptance, at the effect's own ledger
position, rather than a fact re-inferred from a final state assembled later.
``request_invoice_validation`` binds ``invoice_id`` to the invoice the operation
actually accepted.

**The evaluator re-derives, rather than trusts.** The linked proposal's
``invoice_id`` must equal the accepted effect's binding, and that invoice, its
cycle and its evidence must have been admissible *at the effect's ledger
position*: the invoice on record, filed against the cycle the effect committed
against, with authoritative work evidence and the authoritative invoice record
already delivered, and cited by the proposal. A proposal and a binding forged
together to unrelated semantics fails on the re-derivation, not on their
agreement with each other.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from operatebench.domains.lettings.maintenance.evaluator import evaluate
from operatebench.domains.lettings.maintenance.spec import OperationSpec, load_spec
from operatebench.runner import EpisodeRun, run_episode

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "examples" / "operatebench" / "maintenance_v0_1.yaml"

VALIDATION = "request_invoice_validation"

#: The finding an invoice validation that cannot be re-derived is reported as.
UNRE_DERIVABLE = "CRITICAL_INVOICE_VALIDATED_WITHOUT_PRIOR_EVIDENCE"
#: The finding an accepted effect whose bindings are not readable evidence is
#: reported as. Separate from the one above on purpose: "the record does not say
#: what this effect established" and "what it says cannot be re-derived" are
#: different defects and an operator has to be able to tell them apart.
UNREADABLE_BINDINGS = "CRITICAL_EFFECT_BINDINGS_UNREADABLE"


@pytest.fixture(scope="module")
def spec() -> OperationSpec:
    return load_spec(FIXTURE)


@pytest.fixture(scope="module")
def run(spec: OperationSpec) -> EpisodeRun:
    return run_episode(spec, "V1", "reference")


def _trajectory(run: EpisodeRun) -> list[dict[str, Any]]:
    return [copy.deepcopy(dict(row)) for row in run.outcome.trajectory]


def _evaluate(
    spec: OperationSpec,
    run: EpisodeRun,
    trajectory: list[dict[str, Any]],
    final_state: dict[str, Any] | None = None,
) -> Any:
    return evaluate(
        status=run.outcome.status,
        replay_final=run.outcome.replay_final,
        simulated_minutes=run.outcome.simulated_minutes,
        invocations=run.outcome.invocations,
        final_state=(
            final_state
            if final_state is not None
            else copy.deepcopy(dict(run.outcome.final_state))
        ),
        trajectory=trajectory,
        events=copy.deepcopy([dict(event) for event in run.outcome.events]),
        scenario=spec.scenario("V1"),
        replay_ok=True,
    )


def _proposal(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    return next(
        row
        for row in trajectory
        if row.get("record_type") == "action_proposed"
        and row.get("action_type") == VALIDATION
    )


def _effect(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    return next(
        row
        for row in trajectory
        if row.get("record_type") == "effect_accepted"
        and row.get("action_type") == VALIDATION
    )


# --------------------------------------------------------- the ledger evidence


class TestTheAcceptedEffectRecordsWhatItEstablished:
    def test_every_accepted_effect_carries_a_string_to_string_bindings_map(
        self, run: EpisodeRun
    ) -> None:
        accepted = [
            row
            for row in run.outcome.trajectory
            if row["record_type"] == "effect_accepted"
        ]
        assert accepted
        for row in accepted:
            bindings = row.get("bindings")
            assert isinstance(bindings, dict), (
                f"the {row['action_type']!r} effect at index {row['index']} records "
                "no bindings; an accepted effect that does not say what it "
                "established leaves the final state as the only authority"
            )
            for name, value in bindings.items():
                assert isinstance(name, str) and isinstance(value, str)

    def test_the_invoice_validation_effect_binds_the_invoice_it_accepted(
        self, run: EpisodeRun
    ) -> None:
        effect = _effect(_trajectory(run))
        assert effect["bindings"].get("invoice_id") == "invoice_1"

    def test_the_binding_is_detached_from_the_ledger(self, run: EpisodeRun) -> None:
        # The row handed out is a copy: mutating it must not reach the trajectory
        # the digest was taken over.
        rows = _trajectory(run)
        _effect(rows)["bindings"]["invoice_id"] = "invoice_MUTATED"
        assert _effect(_trajectory(run))["bindings"]["invoice_id"] == "invoice_1"


# ------------------------------------------------------------- the exact repro


class TestTheForgedProposalPayload:
    """The parent's reproduction, unchanged in substance.

    A genuine V1 trajectory; only ``proposal_4``'s
    ``request_invoice_validation`` payload ``invoice_id`` is rewritten. The
    accepted effect, the final state and the events are the real ones.
    """

    def test_a_rewritten_invoice_id_is_no_longer_reliable(
        self, spec: OperationSpec, run: EpisodeRun
    ) -> None:
        trajectory = _trajectory(run)
        proposal = _proposal(trajectory)
        assert proposal["payload"]["invoice_id"] == "invoice_1"
        proposal["payload"]["invoice_id"] = "invoice_FORGED_UNRELATED"

        result = _evaluate(spec, run, trajectory)

        assert not result.reliable
        assert UNRE_DERIVABLE in set(result.finding_codes)
        assert "critical_invariants" in set(result.failed_dimensions)

    def test_the_finding_names_the_invoice_the_proposal_claimed(
        self, spec: OperationSpec, run: EpisodeRun
    ) -> None:
        trajectory = _trajectory(run)
        _proposal(trajectory)["payload"]["invoice_id"] = "invoice_FORGED_UNRELATED"
        result = _evaluate(spec, run, trajectory)
        detail = " ".join(
            finding.detail
            for dimension in result.dimensions
            for finding in dimension.findings
            if finding.code == UNRE_DERIVABLE
        )
        assert "invoice_FORGED_UNRELATED" in detail
        assert "invoice_1" in detail


class TestAProposalAndBindingForgedTogether:
    """Agreement between two forged fields is not authority.

    Rewriting the binding to match the rewritten proposal removes the
    disagreement. What it cannot do is make the forged invoice one this
    operation ever held, so the re-derivation still refuses it.
    """

    def test_a_matching_forgery_is_still_refused(
        self, spec: OperationSpec, run: EpisodeRun
    ) -> None:
        trajectory = _trajectory(run)
        _proposal(trajectory)["payload"]["invoice_id"] = "invoice_FORGED_UNRELATED"
        _effect(trajectory)["bindings"]["invoice_id"] = "invoice_FORGED_UNRELATED"

        result = _evaluate(spec, run, trajectory)

        assert not result.reliable
        assert UNRE_DERIVABLE in set(result.finding_codes)

    def test_a_binding_rewritten_alone_is_refused_too(
        self, spec: OperationSpec, run: EpisodeRun
    ) -> None:
        trajectory = _trajectory(run)
        _effect(trajectory)["bindings"]["invoice_id"] = "invoice_FORGED_UNRELATED"
        result = _evaluate(spec, run, trajectory)
        assert not result.reliable
        assert UNRE_DERIVABLE in set(result.finding_codes)


class TestBindingsThatAreNotEvidence:
    def test_an_effect_with_no_bindings_at_all_is_refused(
        self, spec: OperationSpec, run: EpisodeRun
    ) -> None:
        trajectory = _trajectory(run)
        _effect(trajectory).pop("bindings")
        result = _evaluate(spec, run, trajectory)
        assert not result.reliable
        assert UNREADABLE_BINDINGS in set(result.finding_codes)

    def test_a_bindings_field_that_is_not_a_mapping_is_refused(
        self, spec: OperationSpec, run: EpisodeRun
    ) -> None:
        trajectory = _trajectory(run)
        _effect(trajectory)["bindings"] = ["invoice_id", "invoice_1"]
        result = _evaluate(spec, run, trajectory)
        assert not result.reliable
        assert UNREADABLE_BINDINGS in set(result.finding_codes)

    def test_a_non_string_binding_value_is_refused(
        self, spec: OperationSpec, run: EpisodeRun
    ) -> None:
        trajectory = _trajectory(run)
        _effect(trajectory)["bindings"]["invoice_id"] = ["invoice_1"]
        result = _evaluate(spec, run, trajectory)
        assert not result.reliable
        assert UNREADABLE_BINDINGS in set(result.finding_codes)

    def test_an_invoice_validation_that_binds_no_invoice_is_refused(
        self, spec: OperationSpec, run: EpisodeRun
    ) -> None:
        trajectory = _trajectory(run)
        _effect(trajectory)["bindings"] = {}
        result = _evaluate(spec, run, trajectory)
        assert not result.reliable
        assert UNRE_DERIVABLE in set(result.finding_codes)


class TestTheReDerivationIsIndependentOfTheFinalState:
    """A record whose fields agree is still held to what the ledger allowed."""

    def test_an_invoice_filed_against_another_cycle_is_refused(
        self, spec: OperationSpec, run: EpisodeRun
    ) -> None:
        trajectory = _trajectory(run)
        final_state = copy.deepcopy(dict(run.outcome.final_state))
        final_state["invoices"]["invoice_2"] = {
            **final_state["invoices"]["invoice_1"],
            "invoice_id": "invoice_2",
            "cycle_id": "work_cycle_3",
        }
        _proposal(trajectory)["payload"]["invoice_id"] = "invoice_2"
        _effect(trajectory)["bindings"]["invoice_id"] = "invoice_2"

        result = _evaluate(spec, run, trajectory, final_state)

        assert not result.reliable
        assert UNRE_DERIVABLE in set(result.finding_codes)

    def test_an_invoice_no_authoritative_record_ever_delivered_is_refused(
        self, spec: OperationSpec, run: EpisodeRun
    ) -> None:
        # Same cycle, same money, present in the final state — and no
        # ``supplier_invoice_received`` for it reached this operation before the
        # validation was requested.
        trajectory = _trajectory(run)
        final_state = copy.deepcopy(dict(run.outcome.final_state))
        final_state["invoices"]["invoice_2"] = {
            **final_state["invoices"]["invoice_1"],
            "invoice_id": "invoice_2",
        }
        _proposal(trajectory)["payload"]["invoice_id"] = "invoice_2"
        _effect(trajectory)["bindings"]["invoice_id"] = "invoice_2"

        result = _evaluate(spec, run, trajectory, final_state)

        assert not result.reliable
        assert UNRE_DERIVABLE in set(result.finding_codes)

    def test_a_validation_committed_against_the_wrong_cycle_is_refused(
        self, spec: OperationSpec, run: EpisodeRun
    ) -> None:
        trajectory = _trajectory(run)
        _effect(trajectory)["cycle_id"] = "work_cycle_3"
        result = _evaluate(spec, run, trajectory)
        assert not result.reliable
        assert UNRE_DERIVABLE in set(result.finding_codes)

    def test_a_proposal_that_cites_no_work_evidence_is_refused(
        self, spec: OperationSpec, run: EpisodeRun
    ) -> None:
        trajectory = _trajectory(run)
        proposal = _proposal(trajectory)
        assert "evidence_1" in proposal["evidence_refs"]
        proposal["evidence_refs"] = ["assertion_1"]
        result = _evaluate(spec, run, trajectory)
        assert not result.reliable
        assert UNRE_DERIVABLE in set(result.finding_codes)


# ------------------------------------------------------------ the genuine run


class TestTheGenuineRunIsUntouched:
    def test_the_reference_agent_still_grades_clean(
        self, spec: OperationSpec, run: EpisodeRun
    ) -> None:
        result = _evaluate(spec, run, _trajectory(run))
        assert result.reliable
        assert list(result.failed_dimensions) == []
        assert list(result.finding_codes) == []

    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_no_scenario_gains_a_binding_finding(
        self, spec: OperationSpec, scenario_id: str
    ) -> None:
        episode = run_episode(spec, scenario_id, "reference")
        result = evaluate(
            status=episode.outcome.status,
            replay_final=episode.outcome.replay_final,
            simulated_minutes=episode.outcome.simulated_minutes,
            invocations=episode.outcome.invocations,
            final_state=dict(episode.outcome.final_state),
            trajectory=[dict(row) for row in episode.outcome.trajectory],
            events=[dict(event) for event in episode.outcome.events],
            scenario=spec.scenario(scenario_id),
            replay_ok=True,
        )
        codes = set(result.finding_codes)
        assert UNRE_DERIVABLE not in codes
        assert UNREADABLE_BINDINGS not in codes


class TestTheOlderExactChecksSurvive:
    """The two proposal correlations that already existed still fail closed."""

    def test_a_rewritten_work_authorisation_proposal_is_still_refused(
        self, spec: OperationSpec, run: EpisodeRun
    ) -> None:
        trajectory = _trajectory(run)
        proposal = next(
            row
            for row in trajectory
            if row.get("record_type") == "action_proposed"
            and row.get("action_type") == "authorise_supplier_work"
        )
        proposal["payload"]["quote_version"] = 99
        result = _evaluate(spec, run, trajectory)
        assert not result.reliable
        assert "CRITICAL_WORK_AUTHORISED_WITHOUT_PRIOR_APPROVAL" in set(
            result.finding_codes
        )

    def test_a_rewritten_payment_proposal_is_still_refused(
        self, spec: OperationSpec, run: EpisodeRun
    ) -> None:
        trajectory = _trajectory(run)
        proposal = next(
            row
            for row in trajectory
            if row.get("record_type") == "action_proposed"
            and row.get("action_type") == "request_payment"
        )
        proposal["payload"]["invoice_id"] = "invoice_FORGED_UNRELATED"
        result = _evaluate(spec, run, trajectory)
        assert not result.reliable
        assert "CRITICAL_PAYMENT_WITHOUT_PRIOR_EVIDENCE" in set(result.finding_codes)
