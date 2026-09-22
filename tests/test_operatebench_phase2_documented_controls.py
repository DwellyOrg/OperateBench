"""Phase 2, part 7: the published description of the gate is derived, not typed.

The methodology document named an oracle file the package stopped loading, said
there were six controls when the shipped oracle declares eight, said the gate
runs nine checks when it runs eleven, and stated that `trust_actor_claim` leaves
`retrieval_discipline` standing when the oracle it is quoting declares that
dimension inside the control's failure closure. Each of those is a number or a
name a reader can check, and each was wrong in the direction that flatters the
build.

So they are checked here against the executable thing they describe: the oracle
the package actually loads, and the report `check-maintenance` actually
produces. Prose is asserted only where the prose *is* the fact — a path, a count,
a closure — because a test that pins a sentence stops the sentence being
improved and catches nothing a reader would care about.

Version-scoped history is left alone on purpose. A document that says what was
true at contract 1, and says so under a heading that dates it, is evidence; the
tests below reach only for claims a reader would take as current.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from operatebench.domains.lettings.maintenance.evaluator import DIMENSIONS
from operatebench.domains.lettings.maintenance.oracle import (
    ORACLE_RESOURCE_NAME,
    negative_control_oracle,
)
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.runner import check_maintenance

SPEC = "examples/operatebench/maintenance_v0_1.yaml"
METHODOLOGY = Path("docs/METHODOLOGY.md")
DRAFT = Path("examples/operatebench/controls/matched_decision_points_draft_v0_3.yaml")
VERIFICATION = Path("PUBLIC_CANDIDATE_VERIFICATION.md")

ORACLE_PATH = (
    "src/operatebench/domains/lettings/maintenance/oracles/"
    "maintenance_negative_controls_v0_5.yaml"
)

NUMBER_WORDS = {
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
}


@pytest.fixture(scope="module")
def oracle():
    return negative_control_oracle()


@pytest.fixture(scope="module")
def report():
    return check_maintenance(load_spec(SPEC))


def _flowed(path: Path) -> str:
    """The document with line wrapping removed, so a claim spanning two lines
    is one string rather than two halves neither of which matches anything."""
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))


def _backticked(fragment: str) -> list[str]:
    return re.findall(r"`([a-z_]+)`", fragment)


class TestTheMethodologyNamesTheOracleThePackageLoads:
    def test_the_path_it_prints_is_the_resource_that_ships(self, oracle) -> None:
        assert ORACLE_PATH.endswith(ORACLE_RESOURCE_NAME)
        assert Path(ORACLE_PATH).is_file()
        assert ORACLE_PATH in _flowed(METHODOLOGY)

    def test_it_no_longer_prints_the_superseded_resource_as_current(self) -> None:
        superseded = (
            "src/operatebench/domains/lettings/maintenance/oracles/"
            "maintenance_negative_controls.yaml"
        )
        assert superseded not in _flowed(METHODOLOGY)

    def test_it_states_the_number_of_controls_the_oracle_declares(self, oracle) -> None:
        word = NUMBER_WORDS[len(oracle.controls)]
        flowed = _flowed(METHODOLOGY)
        assert f"For each of the {word} controls" in flowed
        assert f"The {word} controls and the closures" in flowed

    def test_it_states_the_number_of_checks_the_gate_runs(self, report) -> None:
        flowed = _flowed(METHODOLOGY)
        assert f"It runs {len(report.checks)} checks" in flowed
        assert "It runs 9 checks" not in flowed

    def test_it_names_the_variant_count_the_negatives_actually_use(
        self, oracle, report
    ) -> None:
        negatives = [check for check in report.checks if check.agent_id != "reference"]
        word = NUMBER_WORDS[len(negatives)]
        assert f"`V1` for all {word} in this build" in _flowed(METHODOLOGY)
        assert {check.scenario_id for check in negatives} == {"V1"}

    def test_it_states_the_width_of_the_vector_it_grades(self) -> None:
        word = NUMBER_WORDS[len(DIMENSIONS)]
        flowed = _flowed(METHODOLOGY)
        assert f"the whole {word}-dimension vector" in flowed
        assert f"all {word}" in flowed


class TestThePublishedClosuresAreTheOracleClosures:
    def test_every_control_has_a_row_and_the_row_is_its_closure(self, oracle) -> None:
        lines = METHODOLOGY.read_text(encoding="utf-8").splitlines()
        rows = {}
        for line in lines:
            match = re.match(r"^\| `([a-z_]+)` \|(.+)\|(.+)\|$", line)
            if match:
                rows[match.group(1)] = _backticked(match.group(3))
        for control in oracle.controls:
            assert control.agent_id in rows, f"{control.agent_id} has no published row"
            assert rows[control.agent_id] == list(control.expected_failed_dimensions)
        assert set(rows) == set(oracle.agent_ids)

    def test_the_isolation_control_paragraph_states_its_exact_closure(
        self, oracle
    ) -> None:
        control = next(
            item for item in oracle.controls if item.agent_id == "trust_actor_claim"
        )
        flowed = _flowed(METHODOLOGY)
        assert "It must fail exactly" in flowed
        failing, rest = flowed.split("It must fail exactly", 1)[1].split("while", 1)
        holding = rest.split("still hold", 1)[0]
        assert _backticked(failing) == list(control.expected_failed_dimensions)
        assert _backticked(holding) == list(control.must_pass_dimensions)

    def test_the_isolation_control_is_not_published_as_reading_cleanly(
        self, oracle
    ) -> None:
        control = next(
            item for item in oracle.controls if item.agent_id == "trust_actor_claim"
        )
        assert "retrieval_discipline" in control.expected_failed_dimensions
        assert "retrieval_discipline" not in control.must_pass_dimensions
        flowed = _flowed(METHODOLOGY)
        assert "`environment_integrity` and `retrieval_discipline` still hold" not in (
            flowed
        )


class TestTheShippedOracleCommentsCountTheControlsItDeclares:
    def test_no_dimension_note_still_says_six(self, oracle) -> None:
        word = NUMBER_WORDS[len(oracle.controls)]
        text = Path(ORACLE_PATH).read_text(encoding="utf-8")
        flowed = re.sub(r"\s+", " ", text)
        assert f"None of the {word} controls attacks" in flowed
        assert "None of the six controls attacks" not in flowed


class TestTheMatchedControlDraftDescribesThePhaseItWasWrittenFor:
    def test_the_whole_current_draft_no_longer_claims_state_roots(self) -> None:
        assert "state roots" not in _flowed(DRAFT).lower()

    def _surface(self) -> str:
        """The declared surface and the comments that explain it, unwrapped.

        Comment markers are stripped before the lines are joined, so a sentence
        the author wrapped across two lines is one string here rather than two
        halves with a ``#`` between them.
        """
        text = DRAFT.read_text(encoding="utf-8")
        body = text.split("matched_surface:", 1)[1].split("decision_points:", 1)[0]
        lines = [re.sub(r"^\s*#\s?", "", line) for line in body.splitlines()]
        return re.sub(r"\s+", " ", " ".join(lines))

    def test_the_surface_comments_no_longer_call_this_phase_one_a(self) -> None:
        assert "Phase 1A" not in self._surface()

    def test_the_surface_comments_no_longer_say_state_is_still_published(
        self,
    ) -> None:
        surface = self._surface()
        assert "`state` is still beside them" not in surface
        assert "Phase 2 drops `state`" not in surface

    def test_the_surface_comments_name_what_phase_two_leaves(self) -> None:
        surface = self._surface()
        for stated in (
            "observation_state_roots",
            "`state`",
            "read catalogue",
            "read budget",
        ):
            assert stated in surface

    def test_the_declared_surface_carries_no_state_root(self) -> None:
        import yaml

        manifest = yaml.safe_load(DRAFT.read_text(encoding="utf-8"))
        surface = manifest["matched_surface"]
        assert "observation_state_roots" not in surface
        assert "state" not in surface["observation_projection_fields"]
        for required in (
            "retrieval_catalogue",
            "retrieval_batch_budget",
            "action_read_requirements",
            "outcome_kinds",
        ):
            assert required in surface

    def test_it_is_still_a_draft_with_no_arms_and_no_independent_evidence(
        self,
    ) -> None:
        import yaml

        manifest = yaml.safe_load(DRAFT.read_text(encoding="utf-8"))
        provenance = manifest["provenance"]
        assert provenance["independence"] == "PROJECT_AUTHORED_DRAFT_NOT_INDEPENDENT"
        assert provenance["evidence_class"] == "DRAFT_TEMPLATE_NOT_VALIDATION"
        assert provenance["review_status"] == "AWAITING_INDEPENDENT_REVIEW"
        assert provenance["independent_review"] is None
        assert manifest["denominator"]["semantic_scenarios"] == 1
        assert "arms" not in manifest
        surface = self._surface().lower()
        assert "none of this builds an arm" in surface
        assert "not independent evidence" in surface


class TestTheVerificationRecordCarriesTheCurrentGateCount:
    def test_the_recorded_exercise_matches_the_current_oracle(
        self, report, oracle
    ) -> None:
        flowed = _flowed(VERIFICATION)
        word = NUMBER_WORDS[len(oracle.controls)]
        assert report.ok and all(check.ok for check in report.checks)
        assert f"{len(report.checks)} checks" in flowed
        assert f"{word} targeted negatives" in flowed
        assert "maintenance check made no model or provider call" in flowed
        assert "Changed-candidate hosted CI: pending" in flowed
        assert "supersedes it" not in flowed.lower()
