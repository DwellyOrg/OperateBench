"""Distribution, engine, operation and artefact contract are four identities.

The defect this file exists to stop is a quiet one: the runtime and the
evaluator both changed, and the number that names them did not, because the
release scanner required ``OPERATEBENCH_VERSION`` to *equal* the distribution
version. An engine change then looked like a packaging change — nothing to bump,
nothing to declare — and artefacts produced under different execution and
grading semantics both said ``engine_version 0.1.0``, which is the one thing
docs/VERSIONING.md says a version must never do.

So each identity is pinned here by value, separately, together with the
documents that publish it. Drift in any direction fails: bumping the engine
without the manifest, bumping the distribution and dragging the engine with it,
adding a dimension without updating the prose that counts them.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from operatebench.artifact import ARTIFACT_VERSION, build_artifact
from operatebench.domains.lettings.maintenance.evaluator import DIMENSIONS
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.runner import run_episode
from operatebench.version import (
    CARD_SCHEMA_VERSION,
    OPERATEBENCH_VERSION,
    __version__,
)
from tools import check_public_release as checks

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "examples" / "operatebench" / "maintenance_v0_1.yaml"
VERSION_FILE = REPO_ROOT / "src" / "operatebench" / "version.py"
ARTIFACT_FILE = REPO_ROOT / "src" / "operatebench" / "artifact.py"

#: What this branch ships, stated once. Every assertion below reads from here so
#: a deliberate bump is one edit and an accidental one is a failure.
DISTRIBUTION_VERSION = "0.1.0"
ENGINE_VERSION = "0.12.0"
OPERATION_VERSION = "0.6.0"
#: The artefact contract this build writes. Moved to 2 when the record grew the
#: decision tape, its digest and the agent execution record, to 3 when every
#: ``effect_accepted`` row grew the canonical identities that effect
#: established, and to 4 when the record grew the run's operation instance
#: identity and the declared-wait row dropped its reachability claim: a reader
#: that met a document carrying either under the other contract would have to
#: guess what it means.
ARTIFACT_CONTRACT_VERSION = 8

#: How many dimensions the result vector carries, spelled the way the public
#: prose spells it.
_NUMBER_WORDS = {8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def _every_key(value: object) -> list[str]:
    """Every mapping key anywhere in a JSON-shaped structure."""
    if isinstance(value, dict):
        keys: list[str] = []
        for key, nested in value.items():
            keys.append(str(key))
            keys.extend(_every_key(nested))
        return keys
    if isinstance(value, list):
        return [key for item in value for key in _every_key(item)]
    return []


def _manifest() -> dict[str, object]:
    loaded = json.loads((REPO_ROOT / "PUBLICATION_MANIFEST.json").read_text("utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _release() -> dict[str, object]:
    release = _manifest()["release"]
    assert isinstance(release, dict)
    return release


@pytest.fixture(scope="module")
def artifact() -> dict[str, object]:
    spec = load_spec(FIXTURE)
    return build_artifact(run_episode(spec, "V1", "reference", self_check=False))


class TestTheIdentitiesAreSeparate:
    def test_card_contract_version_is_independent_literal(self) -> None:
        assert CARD_SCHEMA_VERSION == 1
        assert OPERATEBENCH_VERSION == "0.12.0"
        assert ARTIFACT_VERSION == 8

    def test_the_distribution_version_is_the_wheel_version(self) -> None:
        assert __version__ == DISTRIBUTION_VERSION
        assert checks.DISTRIBUTION_VERSION == DISTRIBUTION_VERSION

    def test_the_engine_version_is_the_runtime_and_evaluator_version(self) -> None:
        assert OPERATEBENCH_VERSION == ENGINE_VERSION
        assert checks.ENGINE_VERSION == ENGINE_VERSION

    def test_the_two_are_not_the_same_number(self) -> None:
        # Not a permanent property — they may coincide again after a release —
        # but on this branch they differ, and a test that asserted equality is
        # what hid the last engine change.
        assert __version__ != OPERATEBENCH_VERSION

    def test_neither_literal_is_derived_from_the_other(self) -> None:
        # An AST read rather than an import: ``OPERATEBENCH_VERSION = __version__``
        # would satisfy every value assertion above while making an engine bump
        # impossible without a distribution bump.
        tree = ast.parse(VERSION_FILE.read_text(encoding="utf-8"))
        literals: dict[str, ast.expr] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    literals[target.id] = node.value
        for name in ("__version__", "OPERATEBENCH_VERSION"):
            assigned = literals.get(name)
            assert isinstance(assigned, ast.Constant), (
                f"{name} must be its own string literal, not derived from another "
                "version name"
            )
            assert isinstance(assigned.value, str)

    def test_the_artifact_contract_version_is_the_one_this_build_writes(self) -> None:
        # Owned by the artefact contract, not by this branch, and it moved: the
        # record now carries `decisions`, `decisions_digest_sha256` and
        # `agent_execution`. It is a fourth identity and is pinned by value, the
        # same way the other three are, so a shape change that did not move it
        # fails here rather than shipping as the same contract.
        assert ARTIFACT_VERSION == ARTIFACT_CONTRACT_VERSION
        assert checks.ARTIFACT_VERSION == ARTIFACT_CONTRACT_VERSION

    def test_the_artifact_contract_is_its_own_literal(self) -> None:
        # An AST read, for the same reason the engine one is: a contract number
        # computed from a field count or from another version would satisfy
        # every value assertion above and could not be pinned by a scanner that
        # must not import the package to read it.
        tree = ast.parse(ARTIFACT_FILE.read_text(encoding="utf-8"))
        literals = {
            target.id: node.value
            for node in tree.body
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        assigned = literals.get("ARTIFACT_VERSION")
        assert isinstance(assigned, ast.Constant)
        assert assigned.value == ARTIFACT_CONTRACT_VERSION
        assert isinstance(assigned.value, int) and not isinstance(assigned.value, bool)

    def test_the_operation_version_is_the_fixtures_own(self) -> None:
        spec = load_spec(FIXTURE)
        assert spec.operation_version == OPERATION_VERSION


class TestTheManifestPublishesTheSameIdentities:
    def test_the_manifest_records_the_engine_this_build_runs(self) -> None:
        assert _release()["engine_version"] == OPERATEBENCH_VERSION

    def test_the_manifest_records_the_distribution_this_build_ships(self) -> None:
        distribution = _release()["distribution"]
        assert isinstance(distribution, dict)
        assert distribution["version"] == __version__

    def test_the_current_ship_note_names_the_manifest_engine_and_distribution(
        self,
    ) -> None:
        release = _release()
        distribution = release["distribution"]
        assert isinstance(distribution, dict)
        note = release["engine_version_independence_note"]
        assert isinstance(note, str)
        current_claim = (
            f"Engine {release['engine_version']} ships inside distribution "
            f"{distribution['version']}."
        )
        assert current_claim in note
        claims = re.findall(
            r"Engine \d+\.\d+\.\d+ ships inside distribution \d+\.\d+\.\d+\.",
            note,
        )
        assert claims == [current_claim]
        assert "Historically, Engine 0.2.0 changed event delivery" in note
        assert "Engine 0.4.0 stopped disclosing the scenario identity" in note

    def test_the_manifest_records_the_artifact_contract(self) -> None:
        assert _release()["artifact_version"] == ARTIFACT_VERSION

    def test_the_manifest_records_the_operation_version(self) -> None:
        spec = load_spec(FIXTURE)
        versions = _release()["operation_spec_versions"]
        assert isinstance(versions, list)
        assert versions == [f"{spec.operation_id}@{spec.operation_version}"]

    def test_the_release_scanner_accepts_the_current_identities(self) -> None:
        assert checks.check_package_versions(REPO_ROOT) == []


class TestTheScannerCatchesDrift:
    """The scanner has to fail on the states it is supposed to prevent.

    Built in a temporary tree rather than by editing the real one: a check that
    is only ever run against a passing repository is not evidence that it can
    fail.
    """

    def _tree(
        self,
        tmp_path: Path,
        *,
        distribution: str = DISTRIBUTION_VERSION,
        engine: str = ENGINE_VERSION,
        manifest_engine: str = ENGINE_VERSION,
        artifact: str = str(ARTIFACT_CONTRACT_VERSION),
        manifest_artifact: int = ARTIFACT_CONTRACT_VERSION,
    ) -> Path:
        (tmp_path / "src" / "operatebench").mkdir(parents=True)
        (tmp_path / "src" / "boundarybench").mkdir(parents=True)
        (tmp_path / "src" / "operatebench" / "version.py").write_text(
            f'__version__ = "{distribution}"\nOPERATEBENCH_VERSION = "{engine}"\n',
            encoding="utf-8",
        )
        (tmp_path / "src" / "operatebench" / "artifact.py").write_text(
            f"ARTIFACT_VERSION = {artifact}\nARTIFACT_VERSION_V1 = 1\n"
            "ARTIFACT_VERSION_V2 = 2\n",
            encoding="utf-8",
        )
        (tmp_path / "src" / "boundarybench" / "version.py").write_text(
            f'__version__ = "{checks.BOUNDARY_PACKAGE_VERSION}"\n', encoding="utf-8"
        )
        (tmp_path / "PUBLICATION_MANIFEST.json").write_text(
            json.dumps(
                {
                    "release": {
                        "engine_version": manifest_engine,
                        "artifact_version": manifest_artifact,
                        "distribution": {"version": distribution},
                    }
                }
            ),
            encoding="utf-8",
        )
        return tmp_path

    def test_a_faithful_tree_passes(self, tmp_path: Path) -> None:
        assert checks.check_package_versions(self._tree(tmp_path)) == []

    def test_an_engine_that_did_not_move_is_a_problem(self, tmp_path: Path) -> None:
        problems = checks.check_package_versions(
            self._tree(tmp_path, engine="0.2.0", manifest_engine="0.2.0")
        )
        assert any("OPERATEBENCH_VERSION" in problem for problem in problems)

    def test_an_engine_bump_the_manifest_did_not_publish_is_a_problem(
        self, tmp_path: Path
    ) -> None:
        problems = checks.check_package_versions(
            self._tree(tmp_path, manifest_engine="0.2.0")
        )
        assert any("PUBLICATION_MANIFEST.json" in problem for problem in problems)

    def test_a_distribution_bump_that_dragged_the_engine_with_it_is_a_problem(
        self, tmp_path: Path
    ) -> None:
        problems = checks.check_package_versions(
            self._tree(tmp_path, distribution="0.3.0", engine="0.3.0")
        )
        assert any("__version__" in problem for problem in problems)

    def test_an_artifact_contract_that_did_not_move_is_a_problem(
        self, tmp_path: Path
    ) -> None:
        # The same defect as the engine one, one identity over: the record
        # shapes changed and the number that names the contract did not, so two
        # documents with different field sets both say `artifact_version 1`.
        problems = checks.check_package_versions(
            self._tree(tmp_path, artifact="2", manifest_artifact=2)
        )
        assert any("ARTIFACT_VERSION" in problem for problem in problems)

    def test_an_artifact_bump_the_manifest_did_not_publish_is_a_problem(
        self, tmp_path: Path
    ) -> None:
        problems = checks.check_package_versions(
            self._tree(tmp_path, manifest_artifact=2)
        )
        assert any("PUBLICATION_MANIFEST.json" in problem for problem in problems)
        assert any("artifact_version" in problem for problem in problems)

    def test_an_artifact_contract_that_is_not_an_integer_literal_is_a_problem(
        self, tmp_path: Path
    ) -> None:
        # Read as a literal, not imported. A contract version computed from
        # something else is one no scanner can pin, and "could not read it" is
        # not the same as "it agrees".
        problems = checks.check_package_versions(
            self._tree(tmp_path, artifact="len(_FIELDS)")
        )
        assert any("ARTIFACT_VERSION" in problem for problem in problems)

    def test_a_manifest_artifact_version_that_is_not_an_integer_is_a_problem(
        self, tmp_path: Path
    ) -> None:
        tree = self._tree(tmp_path)
        (tree / "PUBLICATION_MANIFEST.json").write_text(
            json.dumps(
                {
                    "release": {
                        "engine_version": ENGINE_VERSION,
                        "artifact_version": str(ARTIFACT_CONTRACT_VERSION),
                        "distribution": {"version": DISTRIBUTION_VERSION},
                    }
                }
            ),
            encoding="utf-8",
        )
        problems = checks.check_package_versions(tree)
        assert any("artifact_version" in problem for problem in problems)

    def test_a_sdist_surface_without_the_manifest_is_not_a_disagreement(
        self, tmp_path: Path
    ) -> None:
        """The sdist ships the code but not the manifest, and still must pass.

        ``PUBLICATION_MANIFEST.json`` is a source-control file: the scanner
        bundled inside an extracted sdist has no published record to compare
        the literals against. Reading nothing there is not the same as reading
        the wrong number, and treating it as one made an absent-by-design file
        look like two version defects.
        """
        tree = self._tree(tmp_path)
        (tree / "PUBLICATION_MANIFEST.json").unlink()
        assert checks.check_package_versions(tree, surface="sdist") == []

    def test_the_sdist_omission_is_the_whole_run_not_one_check(
        self, tmp_path: Path
    ) -> None:
        # The surface has to reach check_package_versions through run_all too,
        # or the whole-scanner sdist run reintroduces the failure the check
        # above exists to avoid.
        tree = self._tree(tmp_path)
        (tree / "PUBLICATION_MANIFEST.json").unlink()
        assert checks.run_all(tree, surface="sdist")["package versions"] == []
        assert checks.run_all(tree, surface="source")["package versions"] != []


class TestAManifestThatCannotBeReadIsNotAManifestThatAgrees:
    """The source surface must never *skip* the cross-check in silence.

    ``_publication_manifest_release`` answered ``None`` for four different
    situations — absent, unparseable, not an object, and carrying no release
    mapping — and the caller treated all four as "this surface does not answer
    the question". Only the first is that. For the other three the manifest is
    present and wrong, and returning ``None`` made "engine and distribution
    agree" and "nobody compared them" the same printed ``ok``. The
    required-files check does not cover it: that check only asks whether the
    text contains ``manifest_schema_version``, which every case below carries.
    """

    def _tree(self, tmp_path: Path, manifest: str) -> Path:
        (tmp_path / "src" / "operatebench").mkdir(parents=True)
        (tmp_path / "src" / "boundarybench").mkdir(parents=True)
        (tmp_path / "src" / "operatebench" / "version.py").write_text(
            f'__version__ = "{DISTRIBUTION_VERSION}"\n'
            f'OPERATEBENCH_VERSION = "{ENGINE_VERSION}"\n',
            encoding="utf-8",
        )
        (tmp_path / "src" / "operatebench" / "artifact.py").write_text(
            f"ARTIFACT_VERSION = {ARTIFACT_CONTRACT_VERSION}\n", encoding="utf-8"
        )
        (tmp_path / "src" / "boundarybench" / "version.py").write_text(
            f'__version__ = "{checks.BOUNDARY_PACKAGE_VERSION}"\n', encoding="utf-8"
        )
        (tmp_path / "PUBLICATION_MANIFEST.json").write_text(manifest, encoding="utf-8")
        return tmp_path

    @pytest.mark.parametrize(
        ("label", "manifest"),
        [
            (
                "invalid JSON that still reads as a manifest",
                '{"manifest_schema_version": "0.2.0", "release": {,}',
            ),
            (
                "valid JSON whose top level is not a mapping",
                '["manifest_schema_version", "0.2.0"]',
            ),
            (
                "a mapping with no release block",
                '{"manifest_schema_version": "0.2.0", "gates": []}',
            ),
            (
                "a release that is not a mapping",
                '{"manifest_schema_version": "0.2.0", "release": "0.1.0"}',
            ),
            (
                "a release whose distribution is not a mapping",
                '{"manifest_schema_version": "0.2.0", "release": '
                '{"engine_version": "0.2.0", "distribution": "0.1.0"}}',
            ),
        ],
    )
    def test_the_source_surface_names_a_manifest_it_cannot_cross_check(
        self, tmp_path: Path, label: str, manifest: str
    ) -> None:
        problems = checks.check_package_versions(
            self._tree(tmp_path, manifest), surface="source"
        )
        assert problems, f"{label} silently skipped the engine/distribution check"
        assert all("PUBLICATION_MANIFEST.json" in problem for problem in problems)

    def test_an_absent_manifest_is_a_problem_on_the_source_surface(
        self, tmp_path: Path
    ) -> None:
        tree = self._tree(tmp_path, "{}")
        (tree / "PUBLICATION_MANIFEST.json").unlink()
        problems = checks.check_package_versions(tree, surface="source")
        assert any("PUBLICATION_MANIFEST.json is missing" in p for p in problems)

    def test_a_manifest_that_is_not_utf8_text_is_a_problem(self, tmp_path: Path) -> None:
        tree = self._tree(tmp_path, "{}")
        (tree / "PUBLICATION_MANIFEST.json").write_bytes(b'{"release": \x00\xff}')
        problems = checks.check_package_versions(tree, surface="source")
        assert any("not readable UTF-8 text" in problem for problem in problems)

    def test_a_present_but_unreadable_manifest_fails_on_the_sdist_surface_too(
        self, tmp_path: Path
    ) -> None:
        # Only *absence* is out of scope for the sdist, because only absence is
        # the design. A manifest that is there and unreadable is a defect on
        # whichever surface finds it.
        problems = checks.check_package_versions(
            self._tree(tmp_path, '{"manifest_schema_version": "0.2.0",'),
            surface="sdist",
        )
        assert any("does not parse as JSON" in problem for problem in problems)

    def test_the_real_manifest_still_cross_checks_on_both_surfaces(self) -> None:
        assert checks.check_package_versions(REPO_ROOT, surface="source") == []
        assert checks.check_package_versions(REPO_ROOT, surface="sdist") == []


def test_current_engine_replays_its_own_ordinary_reference() -> None:
    from operatebench.artifact import replay_artifact

    spec = load_spec(FIXTURE)
    artifact = build_artifact(run_episode(spec, "V1", "reference"))
    assert artifact["engine_version"] == "0.12.0"
    assert replay_artifact(spec, artifact).ok


def test_historical_engine_is_readable_but_refused_before_execution(monkeypatch) -> None:
    import operatebench.artifact as artifacts

    historical = artifacts.read_artifact(
        REPO_ROOT / "tests/fixtures/b3/artifact8-deterministic-reference-v1.json"
    )
    assert historical["engine_version"] == "0.9.0"
    assert historical["artifact_version"] == ARTIFACT_VERSION == 8

    def forbidden(*args, **kwargs):
        pytest.fail("an older engine trajectory must not execute under this runtime")

    monkeypatch.setattr(artifacts, "_reproduce", forbidden)
    with pytest.raises(
        artifacts.ArtifactError, match=r"engine version.*0\.9\.0.*0\.12\.0"
    ):
        artifacts.replay_artifact(load_spec(FIXTURE), historical)


class TestTheArtefactCarriesTheIdentitiesOfTheRun:
    """What a downstream reader of an artefact can actually verify.

    The engine identity is only useful if it travels with the run, next to the
    contract version and the operation's own version and digest. This asserts
    those are present in one artefact and agree with the code that produced it —
    and that the distribution version is not among them. An artefact records
    what it was *run* under; which wheel was installed at the time is packaging
    provenance, and putting it in the record would invite exactly the comparison
    it does not license.
    """

    def test_the_artefact_records_the_engine_version(
        self, artifact: dict[str, object]
    ) -> None:
        assert artifact["engine_version"] == OPERATEBENCH_VERSION

    def test_the_artefact_records_the_artifact_contract_version(
        self, artifact: dict[str, object]
    ) -> None:
        assert artifact["artifact_version"] == ARTIFACT_VERSION

    def test_the_artefact_records_the_operation_identity(
        self, artifact: dict[str, object]
    ) -> None:
        operation = artifact["operation"]
        assert isinstance(operation, dict)
        assert operation["operation_version"] == OPERATION_VERSION
        assert len(str(operation["spec_digest_sha256"])) == 64

    def test_the_artefact_does_not_record_the_distribution_version(
        self, artifact: dict[str, object]
    ) -> None:
        # The whole record, not just its identity section: the defect this
        # prevents is re-coupling, and re-coupling can be done from any field.
        assert DISTRIBUTION_VERSION not in json.dumps(artifact)
        assert artifact["engine_version"] != DISTRIBUTION_VERSION

    def test_no_field_anywhere_in_the_artefact_names_the_distribution(
        self, artifact: dict[str, object]
    ) -> None:
        # A value assertion alone would pass the day the distribution and the
        # engine happen to share a number again, which is exactly when a
        # re-coupled field would be invisible. So the *names* are checked too.
        forbidden = ("distribution", "wheel", "package_version", "dist_version")
        for key in _every_key(artifact):
            lowered = key.lower()
            assert not any(word in lowered for word in forbidden), (
                f"artefact field {key!r} names the packaging, not the run"
            )

    def test_the_readme_does_not_claim_the_artefact_carries_both(self) -> None:
        # The prose this replaces said "every artefact carries both". It does
        # not, by design, and a README that says otherwise teaches a reader to
        # look for a field that will never be there.
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        flowed = " ".join(readme.split())
        assert "every artefact carries both" not in flowed
        assert (
            "Every artefact carries the engine version; none carries the "
            "distribution version." in flowed
        )


class TestThePublishedDocumentsStateTheSameNumbers:
    def _read(self, relative: str) -> str:
        return (REPO_ROOT / relative).read_text(encoding="utf-8")

    def test_the_versioning_preview_table_matches_the_code(self) -> None:
        text = self._read("docs/VERSIONING.md")
        assert "| `engine_version` | `OPERATEBENCH_VERSION` in " in text
        assert f"`src/operatebench/version.py` | `{OPERATEBENCH_VERSION}` |" in text
        assert (
            "| `operation.operation_version` | the fixture, echoed into every "
            f"artefact | `{OPERATION_VERSION}` |" in text
        )
        assert (
            f"| `artifact_version` | the artefact contract itself | "
            f"`{ARTIFACT_VERSION}` |" in text
        )

    def test_the_versioning_document_states_what_contract_2_carries(self) -> None:
        """The published document names the fields, and the limit on them.

        A contract number that moves without saying what moved is a number a
        reader has to reverse-engineer from a diff. What contract 2 adds is the
        decision tape, its digest and the execution record; what those enable is
        replaying a stochastic run without calling a provider again; and what
        none of it establishes is that a provider was ever called at all.
        """
        text = self._read("docs/VERSIONING.md")
        # Flowed, the way the README claims are checked: a sentence that wrapped
        # differently is the same published claim.
        flowed = " ".join(text.split())
        for field in ("`decisions`", "`decisions_digest_sha256`", "`agent_execution`"):
            assert field in text, f"docs/VERSIONING.md does not name {field}"
        assert "without calling a provider again" in flowed
        assert "This is consistency, not provider attestation." in flowed
        assert "attests" not in flowed.lower(), (
            "no document may claim the artefact attests that a provider answered; "
            "nothing in a self-describing local file can establish that"
        )

    def test_the_versioning_document_states_the_v1_replay_position(self) -> None:
        flowed = " ".join(self._read("docs/VERSIONING.md").split())
        assert "A contract 1 artefact still validates under this build" in flowed
        assert "named incompatibility" in flowed

    def test_the_versioning_document_keeps_the_distribution_separate(self) -> None:
        text = self._read("docs/VERSIONING.md")
        assert f"Distribution `operatebench` is **{DISTRIBUTION_VERSION}**" in text

    def test_the_readme_states_the_engine_version_it_ships(self) -> None:
        assert f"`engine_version` **{OPERATEBENCH_VERSION}**" in self._read("README.md")

    @pytest.mark.parametrize(
        ("relative", "template"),
        [
            ("README.md", "conjunction of legitimate completion with all {word}\n"),
            ("docs/METHODOLOGY.md", "cover the whole {word}-dimension vector"),
            ("docs/METHODOLOGY.md", "{word} boolean dimensions"),
            ("docs/METHODOLOGY.md", "legitimate completion with all {word}."),
            ("docs/ARCHITECTURE_RFC.md", "emits the {word} per-episode dimensions"),
        ],
    )
    def test_the_prose_counts_the_dimensions_the_evaluator_reports(
        self, relative: str, template: str
    ) -> None:
        text = self._read(relative)
        current = _NUMBER_WORDS[len(DIMENSIONS)]
        assert template.format(word=current) in text, (
            f"{relative} does not say {template.format(word=current)!r}; the "
            f"evaluator reports {len(DIMENSIONS)} dimensions"
        )
        for count, word in _NUMBER_WORDS.items():
            if count != len(DIMENSIONS):
                assert template.format(word=word) not in text

    def test_the_readme_table_lists_every_dimension_once(self) -> None:
        text = self._read("README.md")
        for name in DIMENSIONS:
            assert text.count(f"| `{name}` |") == 1

    def test_the_successor_record_does_not_relabel_historical_validation(self) -> None:
        text = self._read("PUBLIC_CANDIDATE_VERIFICATION.md")
        flowed = " ".join(text.split())
        assert "| Lifecycle engine | 0.7.0 |" not in text
        assert f"| Lifecycle engine | {OPERATEBENCH_VERSION} |" in text
        assert "Changed-candidate hosted CI: pending." in flowed
        assert (
            "That is scoped base evidence, not a claim that the changed candidate "
            "has completed its full matrix." in flowed
        )
        assert (
            "Historical results are not rerun, regraded or relabeled by this record."
            in flowed
        )
        correction = self._read("docs/CORRECTIONS.md")
        assert "no rerun is part of this correction" in correction
        assert f"| Result dimensions | {len(DIMENSIONS)} |" in text

    def test_the_current_record_names_the_current_contract(self) -> None:
        text = self._read("PUBLIC_CANDIDATE_VERIFICATION.md")
        assert f"| Artefact contract | {ARTIFACT_CONTRACT_VERSION} |" in text
