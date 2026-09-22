"""The causal acceptance contract: reference accepted, negatives rejected on purpose.

The gate this file guards is stricter than "reference passes, negatives fail".
An evaluator that rejected everything except one memorised trajectory would
satisfy that. What is asserted here is that each negative fails *exactly* the
dimensions oracle-declared for it, *for* the finding oracle-declared for it, and
that the dimensions oracle-declared as unrelated survive — which is only true if
the dimensions measure different things.

Those expectations are not in this module and not in the agent registry: they
come from the separate oracle manifest, and the tests that hold *it* to
account live in ``test_operatebench_oracle.py``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from operatebench.core.errors import AgentRegistryError
from operatebench.domains.lettings.maintenance.agents import (
    AGENTS,
    NEGATIVE_AGENTS,
    REFERENCE_AGENT,
    AgentEntry,
    build_agent,
)
from operatebench.domains.lettings.maintenance.evaluator import DIMENSIONS
from operatebench.domains.lettings.maintenance.oracle import negative_control_oracle
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.runner import (
    CheckReport,
    check_agent,
    check_maintenance,
    run_episode,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "examples" / "operatebench" / "maintenance_v0_1.yaml"

#: The negatives the Architecture RFC names for the first Maintenance prototype.
REQUIRED_NEGATIVES = frozenset(
    {
        "complete_early",
        "always_act",
        "always_wait",
        "always_escalate",
        "ignore_new_events",
        "trust_actor_claim",
    }
)


@pytest.fixture(scope="module")
def spec() -> Any:
    return load_spec(FIXTURE)


class TestRegistry:
    def test_every_negative_named_by_the_rfc_is_shipped(self) -> None:
        assert set(AGENTS) >= REQUIRED_NEGATIVES

    def test_every_negative_has_a_non_empty_target_and_reason(self) -> None:
        # Read through the entry, which is how the rest of the build sees them,
        # but sourced from the oracle: an entry with nothing oracle-declared would
        # report empty tuples here rather than inventing its own contract.
        for entry in NEGATIVE_AGENTS:
            assert entry.targets, f"{entry.agent_id} has no oracle-declared closure"
            assert entry.expected_findings, f"{entry.agent_id} has no declared reason"
            assert entry.must_pass_dimensions, f"{entry.agent_id} declares no survivors"
            assert set(entry.targets) <= set(DIMENSIONS)
            assert entry.scenarios

    def test_the_reference_is_declared_against_every_required_scenario(self) -> None:
        assert REFERENCE_AGENT.scenarios == ("V1", "V2", "V3")

    def test_agents_are_built_fresh_each_time(self) -> None:
        assert build_agent("reference") is not build_agent("reference")

    def test_an_unknown_agent_is_a_named_domain_error(self) -> None:
        with pytest.raises(AgentRegistryError) as excinfo:
            build_agent("gpt_oracle")
        assert "gpt_oracle" in str(excinfo.value)


class TestReference:
    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_the_reference_is_reliable_on_every_required_scenario(
        self, spec: Any, scenario_id: str
    ) -> None:
        run = run_episode(spec, scenario_id, "reference")
        assert run.reliable is (scenario_id != "V2"), run.evaluation.as_dict()
        assert run.evaluation.failed_dimensions == (
            ("recovery", "obligations") if scenario_id == "V2" else ()
        )
        assert set(run.evaluation.finding_codes) == (
            {"REQUIRED_NOTIFICATION_UNDELIVERED", "OBLIGATION_MESSAGE_NOT_ESTABLISHED"}
            if scenario_id == "V2"
            else set()
        )

    def test_the_reference_reports_every_dimension(self, spec: Any) -> None:
        run = run_episode(spec, "V1", "reference")
        names = tuple(dimension.name for dimension in run.evaluation.dimensions)
        assert names == DIMENSIONS

    def test_the_determinism_self_check_actually_ran(self, spec: Any) -> None:
        run = run_episode(spec, "V1", "reference")
        replay = run.evaluation.dimension("deterministic_replay")
        assert replay is not None and replay.ok is True
        unchecked = run_episode(spec, "V1", "reference", self_check=False)
        assert unchecked.evaluation.dimension("deterministic_replay").ok is False


class TestNegatives:
    @pytest.mark.parametrize("entry", NEGATIVE_AGENTS, ids=lambda e: e.agent_id)
    def test_each_negative_fails_exactly_its_declared_targets(
        self, spec: Any, entry: AgentEntry
    ) -> None:
        for scenario_id in entry.scenarios:
            run = run_episode(spec, scenario_id, entry.agent_id)
            assert run.reliable is False
            assert set(run.evaluation.failed_dimensions) == set(entry.targets), (
                f"{entry.agent_id} on {scenario_id}: "
                f"{run.evaluation.as_dict()['failed_dimensions']}"
            )

    @pytest.mark.parametrize("entry", NEGATIVE_AGENTS, ids=lambda e: e.agent_id)
    def test_each_negative_fails_for_its_intended_reason(
        self, spec: Any, entry: AgentEntry
    ) -> None:
        for scenario_id in entry.scenarios:
            run = run_episode(spec, scenario_id, entry.agent_id)
            assert set(entry.expected_findings) <= set(run.evaluation.finding_codes)

    @pytest.mark.parametrize("entry", NEGATIVE_AGENTS, ids=lambda e: e.agent_id)
    def test_no_negative_is_a_generic_crash(self, spec: Any, entry: AgentEntry) -> None:
        for scenario_id in entry.scenarios:
            run = run_episode(spec, scenario_id, entry.agent_id)
            # Unrelated dimensions keep working: a negative is a targeted
            # failure, not an episode that fell over.
            assert len(run.evaluation.passed_dimensions) >= 3

    def test_the_negatives_are_distinguishable_from_each_other(self, spec: Any) -> None:
        signatures = {
            entry.agent_id: (
                frozenset(run.evaluation.failed_dimensions),
                frozenset(run.evaluation.finding_codes),
            )
            for entry in NEGATIVE_AGENTS
            for run in [run_episode(spec, entry.scenarios[0], entry.agent_id)]
        }
        assert len(set(signatures.values())) == len(NEGATIVE_AGENTS)

    def test_the_claim_trusting_agent_still_completes_the_operation(
        self, spec: Any
    ) -> None:
        # The sharpest negative in the set: everything except the boundary it
        # crossed keeps working, so the failure is attributable.
        run = run_episode(spec, "V1", "trust_actor_claim")
        assert run.outcome.status == "completed_successfully"
        assert set(run.evaluation.failed_dimensions) == {
            "authority_boundaries",
            "action_validity",
            "retrieval_discipline",
        }


class TestAcceptanceGate:
    def test_the_shipped_gate_passes(self, spec: Any) -> None:
        report = check_maintenance(spec)
        assert report.ok is True, [
            check.as_dict() for check in report.checks if not check.ok
        ]
        assert report.spec_digest_sha256 == spec.spec_digest_sha256

    def test_the_gate_covers_the_reference_on_all_three_variants(self, spec: Any) -> None:
        report = check_maintenance(spec)
        covered = {
            (check.agent_id, check.scenario_id)
            for check in report.checks
            if check.kind == "reference"
        }
        assert covered == {("reference", "V1"), ("reference", "V2"), ("reference", "V3")}

    def test_the_gate_rejects_a_negative_that_does_not_fail(self, spec: Any) -> None:
        # The teeth test: a "negative" that behaves like the reference must break
        # the gate rather than pass it quietly. The label is the registry's to
        # give; the expectation it is then held to is the oracle's.
        impostor = AgentEntry(
            agent_id="reference",
            factory=AGENTS["reference"].factory,
            kind="negative",
            description="a reference wearing a negative's label",
            scenarios=("V1",),
        )
        check = check_agent(spec, impostor, "V1")
        assert check.ok is False
        assert any("no oracle-declared control" in problem for problem in check.problems)

    def test_the_gate_rejects_a_negative_that_fails_the_wrong_dimension(
        self, spec: Any
    ) -> None:
        oracle = negative_control_oracle()
        control = oracle.control("always_wait")
        mislabelled = oracle.replacing(
            dataclasses.replace(
                control,
                expected_failed_dimensions=("recovery",),
                must_pass_dimensions=tuple(
                    name for name in DIMENSIONS if name != "recovery"
                ),
            )
        )
        check = check_agent(spec, AGENTS["always_wait"], "V1", oracle=mislabelled)
        assert check.ok is False
        assert any(
            "oracle-declared causal failure closure" in problem
            for problem in check.problems
        )

    def test_the_gate_rejects_a_negative_that_fails_for_the_wrong_reason(
        self, spec: Any
    ) -> None:
        oracle = negative_control_oracle()
        wrong_reason = oracle.replacing(
            dataclasses.replace(
                oracle.control("always_wait"),
                required_finding_codes=("CLAIM_TREATED_AS_AUTHORITATIVE",),
            )
        )
        check = check_agent(spec, AGENTS["always_wait"], "V1", oracle=wrong_reason)
        assert check.ok is False
        assert any("intended reason is missing" in problem for problem in check.problems)

    def test_an_empty_selection_is_the_whole_agent_set(self, spec: Any) -> None:
        # The exported SDK takes this selection from callers that may have built
        # it by filtering. A selection read as "no agents at all" would run no
        # reference and no negative control while still carrying a verdict, and a
        # 0/0 report is indistinguishable from a passing gate to anything
        # downstream. The other operation packs read a falsy selection as the
        # whole set, so this one does too.
        report = check_maintenance(spec, agent_ids=())
        assert report.ok is True
        assert {check.agent_id for check in report.checks} == {
            REFERENCE_AGENT.agent_id,
            *(entry.agent_id for entry in NEGATIVE_AGENTS),
        }

    def test_a_report_that_checked_nothing_is_not_an_acceptance(self) -> None:
        # Belt and braces under the selection rule above: no future path gets to
        # publish a vacuous pass, because `all()` over nothing is True and this
        # field is what automation reads.
        vacuous = CheckReport(
            operation_id="lettings_maintenance_synthetic",
            spec_digest_sha256="0" * 64,
            oracle_id="oracle_probe",
            oracle_version="0.0.0",
            oracle_digest_sha256="0" * 64,
            checks=(),
        )
        assert vacuous.ok is False

    def test_the_gate_names_the_oracle_it_graded_against(self, spec: Any) -> None:
        report = check_maintenance(spec)
        oracle = negative_control_oracle()
        assert report.oracle_id == oracle.oracle_id
        assert report.oracle_digest_sha256 == oracle.oracle_digest_sha256
