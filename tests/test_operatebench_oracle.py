"""The separate negative-control oracle, and the gate that reads it.

The point of this file is a separation that is easy to state and easy to lose:
the *expectations* for the six negative controls live in a manifest that is
authored, shipped and loaded on its own, and the agent registry cannot supply,
widen or narrow any of them. A gate whose expectations come from the same module
as the behaviour proves internal consistency and nothing else.

So three families of assertion here:

* the oracle is loadable, strict about its own shape, and independent — the
  loader imports neither the registry nor the evaluator, and no field of an
  ``AgentEntry`` can override it;
* the registry and the oracle agree, agent for agent and scenario for scenario,
  and the oracle's declared dimension vocabulary is the evaluator's own;
* the gate actually enforces the manifest, including the case the whole design
  exists for — an extra failed dimension the oracle never registered.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml

from operatebench.core.errors import OperateBenchError, OracleManifestError
from operatebench.domains.lettings.maintenance.agents import (
    AGENTS,
    NEGATIVE_AGENTS,
    REFERENCE_AGENT,
)
from operatebench.domains.lettings.maintenance.evaluator import DIMENSIONS
from operatebench.domains.lettings.maintenance.oracle import (
    ORACLE_RESOURCE_NAME,
    NegativeControl,
    NegativeControlOracle,
    load_negative_control_oracle,
    negative_control_oracle,
    oracle_manifest_path,
)
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.runner import check_agent, check_maintenance

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "examples" / "operatebench" / "maintenance_v0_1.yaml"
EXAMPLE_PAGE = REPO_ROOT / "examples" / "operatebench" / "negative_control_oracle.md"
ORACLE_MODULE = (
    REPO_ROOT
    / "src"
    / "operatebench"
    / "domains"
    / "lettings"
    / "maintenance"
    / "oracle.py"
)


@pytest.fixture(scope="module")
def spec() -> Any:
    return load_spec(FIXTURE)


@pytest.fixture(scope="module")
def oracle() -> NegativeControlOracle:
    return negative_control_oracle()


def _manifest() -> dict[str, Any]:
    return yaml.safe_load(oracle_manifest_path().read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: Any) -> Path:
    path = tmp_path / "oracle.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


# ------------------------------------------------------------------ shipping


class TestTheOracleShips:
    def test_the_manifest_is_inside_the_installed_package(self) -> None:
        # Not beside the tests and not in examples/: the gate must be able to
        # read it from an installed wheel, with no source checkout present.
        path = oracle_manifest_path()
        assert path.is_file()
        assert path.name == ORACLE_RESOURCE_NAME
        package_root = Path(
            __import__(
                "operatebench.domains.lettings.maintenance", fromlist=["oracle"]
            ).__file__
        ).parent
        assert package_root in path.parents

    def test_the_manifest_is_machine_readable_yaml(self) -> None:
        payload = _manifest()
        assert isinstance(payload, dict)
        assert payload["oracle_id"] == "lettings_maintenance_negative_controls"
        assert len(payload["controls"]) == 8

    def test_the_oracle_carries_its_own_identity_and_digest(
        self, oracle: NegativeControlOracle
    ) -> None:
        assert oracle.oracle_id == "lettings_maintenance_negative_controls"
        assert oracle.oracle_version == "0.5.0"
        assert len(oracle.oracle_digest_sha256) == 64
        assert set(oracle.oracle_digest_sha256) <= set("0123456789abcdef")

    def test_the_digest_is_over_semantic_content_not_file_bytes(
        self, tmp_path: Path, oracle: NegativeControlOracle
    ) -> None:
        payload = _manifest()
        reflowed = _write(tmp_path, payload)
        assert load_negative_control_oracle(reflowed).oracle_digest_sha256 == (
            oracle.oracle_digest_sha256
        )

    def test_the_manifest_states_its_provenance_and_what_it_is_not(
        self, oracle: NegativeControlOracle
    ) -> None:
        assert "Not generated" in oracle.provenance["independence"]
        evidence = oracle.provenance["evidence_class"].lower()
        assert "construct evidence" in evidence
        assert "not model validation" in evidence
        assert "not independent human validation" in evidence


# -------------------------------------------------------------- independence


class TestIndependenceFromTheRegistry:
    def test_the_loader_imports_neither_the_registry_nor_the_evaluator(self) -> None:
        # A source-level check, because an import is exactly how this file would
        # stop being independent. The oracle is data about the case; it must not
        # be able to consult the thing it is grading.
        tree = ast.parse(ORACLE_MODULE.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not any(
            module.endswith(("maintenance.agents", "maintenance.evaluator"))
            for module in imported
        ), sorted(imported)

    def test_no_module_writes_expectations_back_into_the_manifest(self) -> None:
        # The manifest is read-only input to the build. Nothing generates it.
        sources = (REPO_ROOT / "src").rglob("*.py")
        for path in sources:
            text = path.read_text(encoding="utf-8")
            if ORACLE_RESOURCE_NAME in text:
                assert "write_text" not in text, path

    def test_an_agent_entry_cannot_declare_its_own_expectations(self) -> None:
        # The registry describes behaviour. If `targets` were still a field, a
        # negative could quietly redefine what it is allowed to break.
        fields = {field.name for field in dataclasses.fields(AGENTS["always_wait"])}
        assert "targets" not in fields
        assert "expected_findings" not in fields
        with pytest.raises(TypeError):
            dataclasses.replace(AGENTS["always_wait"], targets=("recovery",))

    def test_the_registry_reads_its_expectations_from_the_oracle(
        self, oracle: NegativeControlOracle
    ) -> None:
        for entry in NEGATIVE_AGENTS:
            control = oracle.control(entry.agent_id)
            assert entry.targets == control.expected_failed_dimensions
            assert entry.expected_failure_closure == control.expected_failed_dimensions
            assert entry.expected_findings == control.required_finding_codes

    def test_the_reference_has_no_oracle_control_and_declares_nothing(
        self, oracle: NegativeControlOracle
    ) -> None:
        assert REFERENCE_AGENT.targets == ()
        assert REFERENCE_AGENT.expected_findings == ()
        assert "reference" not in oracle.agent_ids


# ------------------------------------------------------------------ agreement


class TestRegistryAndOracleAgree:
    def test_the_same_eight_controls_are_registered_and_oracle_declared(
        self, oracle: NegativeControlOracle
    ) -> None:
        assert set(oracle.agent_ids) == {entry.agent_id for entry in NEGATIVE_AGENTS}
        assert len(oracle.controls) == 8

    def test_every_control_names_a_scenario_the_registry_runs(
        self, oracle: NegativeControlOracle
    ) -> None:
        for entry in NEGATIVE_AGENTS:
            control = oracle.control(entry.agent_id)
            assert entry.scenarios == (control.scenario_id,)

    def test_the_oracle_dimension_vocabulary_is_the_evaluators_own(
        self, oracle: NegativeControlOracle
    ) -> None:
        assert oracle.result_vector_dimensions == DIMENSIONS

    def test_every_declared_dimension_is_a_real_dimension(
        self, oracle: NegativeControlOracle
    ) -> None:
        for control in oracle.controls:
            assert set(control.expected_failed_dimensions) <= set(DIMENSIONS)
            assert set(control.must_pass_dimensions) <= set(DIMENSIONS)

    def test_each_control_partitions_the_whole_result_vector(
        self, oracle: NegativeControlOracle
    ) -> None:
        # Closure plus must-pass is the entire vector, with no overlap: the
        # oracle has an opinion about every dimension, so "unmentioned" is never
        # a place for a failure to hide.
        for control in oracle.controls:
            failed = set(control.expected_failed_dimensions)
            passing = set(control.must_pass_dimensions)
            assert not failed & passing, control.agent_id
            assert failed | passing == set(DIMENSIONS), control.agent_id

    def test_every_control_leaves_at_least_three_dimensions_standing(
        self, oracle: NegativeControlOracle
    ) -> None:
        for control in oracle.controls:
            assert len(control.must_pass_dimensions) >= 3, control.agent_id

    def test_no_two_controls_declare_the_same_signature(
        self, oracle: NegativeControlOracle
    ) -> None:
        signatures = {
            (
                frozenset(control.expected_failed_dimensions),
                frozenset(control.required_finding_codes),
            )
            for control in oracle.controls
        }
        assert len(signatures) == len(oracle.controls)

    def test_every_control_says_what_it_attacks(
        self, oracle: NegativeControlOracle
    ) -> None:
        for control in oracle.controls:
            assert control.intervention.strip()
            assert control.guarantee_attacked.strip()
            assert control.rationale.strip()


# ---------------------------------------------------------------- the gate


class TestTheGateReadsTheOracle:
    def test_the_shipped_gate_passes_against_the_shipped_oracle(self, spec: Any) -> None:
        report = check_maintenance(spec)
        assert report.ok is True, [
            check.as_dict() for check in report.checks if not check.ok
        ]

    def test_the_report_records_which_oracle_graded_it(
        self, spec: Any, oracle: NegativeControlOracle
    ) -> None:
        report = check_maintenance(spec)
        assert report.oracle_id == oracle.oracle_id
        assert report.oracle_version == oracle.oracle_version
        assert report.oracle_digest_sha256 == oracle.oracle_digest_sha256
        assert report.as_dict()["oracle_digest_sha256"] == oracle.oracle_digest_sha256

    def test_every_negative_check_reports_the_oracle_declared_expectation(
        self, spec: Any, oracle: NegativeControlOracle
    ) -> None:
        report = check_maintenance(spec)
        for check in report.checks:
            if check.kind != "negative":
                continue
            control = oracle.control(check.agent_id)
            assert check.expected_targets == control.expected_failed_dimensions
            assert check.expected_findings == control.required_finding_codes
            assert check.required_passing == control.must_pass_dimensions
            assert set(control.must_pass_dimensions) <= set(check.passed_dimensions)

    def test_an_unregistered_extra_failed_dimension_breaks_the_gate(
        self, spec: Any, oracle: NegativeControlOracle
    ) -> None:
        # The case the whole design exists for. `always_act` really does take
        # `action_validity` down; an oracle that forgot to declare it must
        # fail the gate rather than tolerate the extra failure.
        narrowed = oracle.control("always_act")
        tampered = oracle.replacing(
            dataclasses.replace(
                narrowed,
                expected_failed_dimensions=tuple(
                    name
                    for name in narrowed.expected_failed_dimensions
                    if name != "action_validity"
                ),
                must_pass_dimensions=(*narrowed.must_pass_dimensions, "action_validity"),
            )
        )
        check = check_agent(spec, AGENTS["always_act"], "V1", oracle=tampered)
        assert check.ok is False
        problems = " ".join(check.problems)
        assert "action_validity" in problems
        assert "outside the oracle-declared closure" in problems

    def test_an_oracle_declared_dimension_that_survives_breaks_the_gate(
        self, spec: Any, oracle: NegativeControlOracle
    ) -> None:
        control = oracle.control("trust_actor_claim")
        tampered = oracle.replacing(
            dataclasses.replace(
                control,
                expected_failed_dimensions=(
                    *control.expected_failed_dimensions,
                    "recovery",
                ),
                must_pass_dimensions=tuple(
                    name for name in control.must_pass_dimensions if name != "recovery"
                ),
            )
        )
        check = check_agent(spec, AGENTS["trust_actor_claim"], "V1", oracle=tampered)
        assert check.ok is False
        assert "oracle-declared but did not fail" in " ".join(check.problems)

    def test_a_must_pass_dimension_that_fails_breaks_the_gate(
        self, spec: Any, oracle: NegativeControlOracle
    ) -> None:
        # Stated separately from the closure comparison, because this is the
        # claim a reader cares about: the unrelated dimensions still work.
        control = oracle.control("always_wait")
        tampered = oracle.replacing(
            dataclasses.replace(
                control,
                must_pass_dimensions=(
                    *control.must_pass_dimensions,
                    "terminal_outcome",
                ),
            )
        )
        check = check_agent(spec, AGENTS["always_wait"], "V1", oracle=tampered)
        assert check.ok is False
        assert "must remain passing" in " ".join(check.problems)

    def test_a_missing_finding_code_breaks_the_gate(
        self, spec: Any, oracle: NegativeControlOracle
    ) -> None:
        control = oracle.control("always_wait")
        tampered = oracle.replacing(
            dataclasses.replace(
                control, required_finding_codes=("CLAIM_TREATED_AS_AUTHORITATIVE",)
            )
        )
        check = check_agent(spec, AGENTS["always_wait"], "V1", oracle=tampered)
        assert check.ok is False
        assert "intended reason is missing" in " ".join(check.problems)

    def test_a_negative_with_no_oracle_declared_control_breaks_the_gate(
        self, spec: Any, oracle: NegativeControlOracle
    ) -> None:
        # An agent registered as a negative but absent from the oracle is not
        # "unconstrained"; it is ungraded, and the gate says so.
        without = oracle.without("always_wait")
        check = check_agent(spec, AGENTS["always_wait"], "V1", oracle=without)
        assert check.ok is False
        assert "no oracle-declared control" in " ".join(check.problems)

    def test_a_control_run_on_the_wrong_scenario_breaks_the_gate(
        self, spec: Any, oracle: NegativeControlOracle
    ) -> None:
        check = check_agent(spec, AGENTS["trust_actor_claim"], "V2", oracle=oracle)
        assert check.ok is False
        assert "oracle-declared against scenario" in " ".join(check.problems)

    def test_a_negative_that_behaves_like_the_reference_breaks_the_gate(
        self, spec: Any, oracle: NegativeControlOracle
    ) -> None:
        # Declare the reference implementation as though it were a control.
        # It never fails, so the gate must refuse it — otherwise "the negatives
        # fail" is satisfiable by an agent that does nothing wrong.
        oracle_declared = oracle.replacing(
            dataclasses.replace(oracle.control("always_wait"), agent_id="reference")
        )
        impostor = dataclasses.replace(AGENTS["reference"], kind="negative")
        check = check_agent(spec, impostor, "V1", oracle=oracle_declared)
        assert check.ok is False
        assert "accepted as reliable" in " ".join(check.problems)


# ---------------------------------------------------------- refusing bad input


class TestTheLoaderRefusesAMalformedManifest:
    def test_an_unreadable_path_is_a_named_error(self, tmp_path: Path) -> None:
        with pytest.raises(OracleManifestError) as excinfo:
            load_negative_control_oracle(tmp_path / "missing.yaml")
        assert "missing.yaml" in str(excinfo.value)
        assert isinstance(excinfo.value, OperateBenchError)

    def test_a_non_mapping_document_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "oracle.yaml"
        path.write_text("- not a mapping\n", encoding="utf-8")
        with pytest.raises(OracleManifestError):
            load_negative_control_oracle(path)

    def test_an_unknown_top_level_field_is_refused(self, tmp_path: Path) -> None:
        payload = _manifest()
        payload["smuggled"] = "value"
        with pytest.raises(OracleManifestError) as excinfo:
            load_negative_control_oracle(_write(tmp_path, payload))
        assert "smuggled" in str(excinfo.value)

    def test_a_missing_top_level_field_is_refused(self, tmp_path: Path) -> None:
        payload = _manifest()
        del payload["provenance"]
        with pytest.raises(OracleManifestError) as excinfo:
            load_negative_control_oracle(_write(tmp_path, payload))
        assert "provenance" in str(excinfo.value)

    def test_an_unknown_control_field_is_refused(self, tmp_path: Path) -> None:
        payload = _manifest()
        payload["controls"][0]["expected_reliable"] = True
        with pytest.raises(OracleManifestError) as excinfo:
            load_negative_control_oracle(_write(tmp_path, payload))
        assert "expected_reliable" in str(excinfo.value)

    def test_a_duplicate_control_is_refused(self, tmp_path: Path) -> None:
        payload = _manifest()
        payload["controls"].append(dict(payload["controls"][0]))
        with pytest.raises(OracleManifestError) as excinfo:
            load_negative_control_oracle(_write(tmp_path, payload))
        assert "complete_early" in str(excinfo.value)

    def test_an_empty_closure_is_refused(self, tmp_path: Path) -> None:
        payload = _manifest()
        payload["controls"][0]["expected_failed_dimensions"] = []
        with pytest.raises(OracleManifestError) as excinfo:
            load_negative_control_oracle(_write(tmp_path, payload))
        assert "expected_failed_dimensions" in str(excinfo.value)

    def test_a_dimension_outside_the_declared_vocabulary_is_refused(
        self, tmp_path: Path
    ) -> None:
        payload = _manifest()
        payload["controls"][0]["expected_failed_dimensions"] = ["vibes"]
        with pytest.raises(OracleManifestError) as excinfo:
            load_negative_control_oracle(_write(tmp_path, payload))
        assert "vibes" in str(excinfo.value)

    def test_a_dimension_both_failing_and_passing_is_refused(
        self, tmp_path: Path
    ) -> None:
        payload = _manifest()
        payload["controls"][0]["must_pass_dimensions"].append("terminal_outcome")
        with pytest.raises(OracleManifestError) as excinfo:
            load_negative_control_oracle(_write(tmp_path, payload))
        assert "terminal_outcome" in str(excinfo.value)

    def test_a_control_that_leaves_a_dimension_unjudged_is_refused(
        self, tmp_path: Path
    ) -> None:
        payload = _manifest()
        payload["controls"][0]["must_pass_dimensions"].remove("recovery")
        with pytest.raises(OracleManifestError) as excinfo:
            load_negative_control_oracle(_write(tmp_path, payload))
        assert "recovery" in str(excinfo.value)

    def test_a_control_with_no_required_finding_code_is_refused(
        self, tmp_path: Path
    ) -> None:
        payload = _manifest()
        payload["controls"][0]["required_finding_codes"] = []
        with pytest.raises(OracleManifestError) as excinfo:
            load_negative_control_oracle(_write(tmp_path, payload))
        assert "required_finding_codes" in str(excinfo.value)

    def test_a_duplicate_dimension_in_the_vocabulary_is_refused(
        self, tmp_path: Path
    ) -> None:
        payload = _manifest()
        payload["result_vector_dimensions"].append("terminal_outcome")
        with pytest.raises(OracleManifestError) as excinfo:
            load_negative_control_oracle(_write(tmp_path, payload))
        assert "terminal_outcome" in str(excinfo.value)

    def test_a_wrong_typed_field_is_refused(self, tmp_path: Path) -> None:
        payload = _manifest()
        payload["controls"][0]["scenario_id"] = ["V1"]
        with pytest.raises(OracleManifestError) as excinfo:
            load_negative_control_oracle(_write(tmp_path, payload))
        assert "scenario_id" in str(excinfo.value)

    def test_an_unknown_agent_lookup_is_a_named_error(
        self, oracle: NegativeControlOracle
    ) -> None:
        with pytest.raises(OracleManifestError) as excinfo:
            oracle.control("gpt_oracle")
        assert "gpt_oracle" in str(excinfo.value)

    def test_the_shipped_manifest_round_trips_through_the_loader(
        self, tmp_path: Path, oracle: NegativeControlOracle
    ) -> None:
        reloaded = load_negative_control_oracle(_write(tmp_path, oracle.as_dict()))
        assert reloaded.as_dict() == oracle.as_dict()
        assert reloaded.controls == oracle.controls


class TestTheTamperingHelpersAreOracleShaped:
    def test_replacing_swaps_one_control_and_keeps_the_rest(
        self, oracle: NegativeControlOracle
    ) -> None:
        control = oracle.control("always_wait")
        replacement = dataclasses.replace(control, rationale="edited")
        tampered = oracle.replacing(replacement)
        assert tampered.control("always_wait").rationale == "edited"
        assert tampered.control("complete_early") == oracle.control("complete_early")
        assert oracle.control("always_wait").rationale != "edited"

    def test_without_drops_exactly_one_control(
        self, oracle: NegativeControlOracle
    ) -> None:
        reduced = oracle.without("always_wait")
        assert set(reduced.agent_ids) == set(oracle.agent_ids) - {"always_wait"}

    def test_a_control_is_immutable(self, oracle: NegativeControlOracle) -> None:
        control = oracle.control("always_wait")
        assert isinstance(control, NegativeControl)
        with pytest.raises(dataclasses.FrozenInstanceError):
            control.scenario_id = "V2"  # type: ignore[misc]


class TestThePublicExamplePageDescribesTheShippedManifest:
    """The example page is where a reader meets the oracle before the YAML.

    It counts things — how many facts a control declares, how wide the
    result vector is, how many controls ship. A page whose counts drift from the
    manifest teaches the wrong shape to the only audience that has no other
    source for it.
    """

    NUMBERS: ClassVar[dict[str, int]] = {
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
    }

    def _page(self) -> str:
        return EXAMPLE_PAGE.read_text(encoding="utf-8")

    def test_the_stated_count_is_the_number_of_facts_it_then_lists(self) -> None:
        match = re.search(
            r"Every control declares (\w+) things:(.+?)\.\n\n", self._page(), re.S
        )
        assert match is not None, "the page no longer enumerates what a control fixes"
        stated = self.NUMBERS[match.group(1)]
        listed = len(match.group(2).split(","))
        assert stated == listed
        # And what it lists is the control's own oracle-declared fields. The
        # rationale is the manifest's explanation of the control, not one of the
        # expectations the gate holds it to, so the sentence does not name it.
        assert stated == len(dataclasses.fields(NegativeControl)) - 1

    def test_the_result_vector_width_it_quotes_is_the_evaluators_own(self) -> None:
        match = re.search(r"the whole (\w+)-dimension result vector", self._page())
        assert match is not None
        assert self.NUMBERS[match.group(1)] == len(DIMENSIONS)

    def test_the_number_of_controls_it_claims_is_the_number_that_ship(self) -> None:
        # Whitespace-tolerant: the sentence is hand-reflowed, and the count
        # regularly lands on the far side of a line break from the verb.
        match = re.search(r"separates\s+the\s+(\w+)\s+shipped controls", self._page())
        assert match is not None
        assert self.NUMBERS[match.group(1)] == len(NEGATIVE_AGENTS)
