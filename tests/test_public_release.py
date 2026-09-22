"""The public-release gate, asserted as tests.

Everything a reviewer would otherwise have to eyeball before this tree becomes
public: that the distribution names and versions itself correctly, that both
tracks ship, that the licensing and governance files are present and say what
they claim, that no credential shape or inaccessible reference survived the export, that
the shipped fixture declares itself synthetic, that every relative link
resolves, and that no generated artefact is being carried as content.

These call the same functions as ``tools/check_public_release.py``, so the
command a release engineer runs and the suite CI runs cannot disagree. Each test
asserts on the *problem list* rather than a boolean, so a failure names what is
wrong instead of only that something is.

Nothing here reads ``.git``, a virtual environment or a cache. Every other
regular file is attempted, with NUL-bearing or invalid UTF-8 data classified as
binary — see the scanner's module docstring for why.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import re
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

from tools import check_public_release as checks

REPO_ROOT = Path(__file__).resolve().parents[1]


def _canonical_distribution(directory: Path) -> tuple[Path, Path]:
    """Write a small byte-canonical distribution from the reviewed source tree."""
    directory.mkdir()
    wheel = directory / "operatebench-0.1.0-py3-none-any.whl"
    source_members = {
        path.relative_to(REPO_ROOT / "src").as_posix(): path.read_bytes()
        for package in ("operatebench", "boundarybench")
        for path in sorted((REPO_ROOT / "src" / package).rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    dist_info = "operatebench-0.1.0.dist-info"
    project_readme = (REPO_ROOT / "README.md").read_bytes()
    wheel_members = {
        **source_members,
        f"{dist_info}/METADATA": (
            b"Metadata-Version: 2.5\n"
            b"Name: operatebench\n"
            b"Version: 0.1.0\n"
            b"Summary: OperateBench \xe2\x80\x94 a deterministic, replayable benchmark "
            b"for agents that own longitudinal business operations\n"
            b"Author-email: Dmitry Khanukov <dmitry@dwelly.group>\n"
            b"Maintainer-email: Dmitry Khanukov <dmitry@dwelly.group>\n"
            b"Keywords: agent-benchmark,business-operations,deterministic-replay,"
            b"event-driven,human-in-the-loop,synthetic-fixtures\n"
            b"License-Expression: Apache-2.0\n"
            b"License-File: LICENSE\n"
            b"License-File: LICENSE-DATA\n"
            b"License-File: NOTICE\n"
            b"License-File: LICENSES/Apache-2.0.txt\n"
            b"License-File: LICENSES/CC-BY-4.0.txt\n"
            b"Requires-Python: >=3.11\n"
            b"Classifier: Development Status :: 3 - Alpha\n"
            b"Classifier: Intended Audience :: Developers\n"
            b"Classifier: Intended Audience :: Science/Research\n"
            b"Classifier: Operating System :: POSIX :: Linux\n"
            b"Classifier: Programming Language :: Python :: 3\n"
            b"Classifier: Programming Language :: Python :: 3.11\n"
            b"Classifier: Programming Language :: Python :: 3.12\n"
            b"Classifier: Programming Language :: Python :: 3.13\n"
            b"Classifier: Programming Language :: Python :: 3.14\n"
            b"Classifier: Topic :: Scientific/Engineering :: Artificial Intelligence\n"
            b"Classifier: Topic :: Software Development :: Testing\n"
            b"Classifier: Typing :: Typed\n"
            b"Project-URL: Homepage, https://github.com/DwellyOrg/OperateBench\n"
            b"Project-URL: Repository, https://github.com/DwellyOrg/OperateBench\n"
            b"Project-URL: Documentation, https://github.com/DwellyOrg/OperateBench/blob/main/docs/METHODOLOGY.md\n"
            b"Project-URL: Issues, https://github.com/DwellyOrg/OperateBench/issues\n"
            b"Requires-Dist: anthropic<1,>=0.121.0\n"
            b"Requires-Dist: openai<3,>=2.53.0\n"
            b"Requires-Dist: mistralai<3,>=2.9.2\n"
            b"Requires-Dist: pyyaml>=6.0\n"
            b"Description-Content-Type: text/markdown\n"
            b"\n" + project_readme
        ),
        f"{dist_info}/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: hatchling 1.32.0\n"
            b"Root-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        f"{dist_info}/entry_points.txt": (
            b"[console_scripts]\n"
            b"boundarybench = boundarybench.cli:main\n"
            b"operatebench = operatebench.cli:main\n"
        ),
        **{
            f"{dist_info}/licenses/{name}": (REPO_ROOT / name).read_bytes()
            for name in (
                "LICENSE",
                "LICENSE-DATA",
                "NOTICE",
                "LICENSES/Apache-2.0.txt",
                "LICENSES/CC-BY-4.0.txt",
            )
        },
    }
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name in sorted(wheel_members):
        raw = wheel_members[name]
        digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=")
        writer.writerow((name, f"sha256={digest.decode()}", str(len(raw))))
    record_name = f"{dist_info}/RECORD"
    writer.writerow((record_name, "", ""))
    wheel_members[record_name] = record.getvalue().encode()
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in wheel_members.items():
            archive.writestr(name, raw)

    sdist = directory / "operatebench-0.1.0.tar.gz"
    sdist_members = {
        f"operatebench-0.1.0/src/{name}": raw for name, raw in source_members.items()
    }
    for name in ("pyproject.toml", "uv.lock", ".github/workflows/ci.yml"):
        sdist_members[f"operatebench-0.1.0/{name}"] = (REPO_ROOT / name).read_bytes()
    with tarfile.open(sdist, "w:gz") as archive:
        for name, raw in sdist_members.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    return wheel, sdist


def _empty_distribution(directory: Path) -> None:
    """Create one benign archive of each kind for single-archive scanner probes."""
    directory.mkdir()
    with zipfile.ZipFile(directory / "pkg-0.1.0-py3-none-any.whl", "w") as wheel:
        wheel.writestr("pkg/placeholder.bin", b"\x00")
    with tarfile.open(directory / "pkg-0.1.0.tar.gz", "w:gz") as sdist:
        raw = b"\x00"
        member = tarfile.TarInfo("pkg-0.1.0/placeholder.bin")
        member.size = len(raw)
        sdist.addfile(member, io.BytesIO(raw))


def _rewrite_wheel(
    wheel: Path,
    *,
    remove: set[str] = frozenset(),
    replace: dict[str, bytes] | None = None,
    append: tuple[str, bytes] | None = None,
) -> None:
    replacement = wheel.with_name("replacement.whl")
    with (
        zipfile.ZipFile(wheel) as source,
        zipfile.ZipFile(replacement, "w", compression=zipfile.ZIP_DEFLATED) as target,
    ):
        for info in source.infolist():
            if info.filename not in remove:
                target.writestr(
                    info, (replace or {}).get(info.filename, source.read(info))
                )
        if append is not None:
            target.writestr(*append)
    replacement.replace(wheel)


def _regenerate_record(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as source:
        members = {
            info.filename: (info, source.read(info))
            for info in source.infolist()
            if not info.filename.endswith(".dist-info/RECORD")
        }
        record_name = next(
            name for name in source.namelist() if name.endswith(".dist-info/RECORD")
        )
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name, (_, raw) in sorted(members.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=")
        writer.writerow((name, f"sha256={digest.decode()}", str(len(raw))))
    writer.writerow((record_name, "", ""))
    replacement = wheel.with_name("recorded.whl")
    with zipfile.ZipFile(replacement, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for info, raw in members.values():
            target.writestr(info, raw)
        target.writestr(record_name, record.getvalue().encode())
    replacement.replace(wheel)


def _rewrite_sdist_member(sdist: Path, name: str, raw: bytes) -> None:
    replacement = sdist.with_name("replacement.tar.gz")
    with (
        tarfile.open(sdist, "r:gz") as source,
        tarfile.open(replacement, "w:gz") as target,
    ):
        for member in source.getmembers():
            payload = source.extractfile(member) if member.isfile() else None
            if member.name == name:
                member.size = len(raw)
                payload = io.BytesIO(raw)
            target.addfile(member, payload)
    replacement.replace(sdist)


# Hatchling's own default pattern for `[tool.hatch.version] path = ...` with no
# explicit `pattern` configured (hatchling.version.core.DEFAULT_PATTERN). Not
# reimplemented from scratch: copied verbatim from the dependency our own build
# invokes, so this test fails exactly when `uv build` would.
_HATCH_DEFAULT_VERSION_PATTERN = (
    r'(?i)^(__version__|VERSION) *= *([\'"])v?(?P<version>.+?)\2'
)


class TestDistributionMetadata:
    def test_the_distribution_is_named_versioned_and_licensed_correctly(self) -> None:
        assert checks.check_distribution_metadata(REPO_ROOT) == []

    def test_both_console_scripts_and_both_packages_ship(self) -> None:
        import tomllib

        data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        assert data["project"]["scripts"] == {
            "operatebench": "operatebench.cli:main",
            "boundarybench": "boundarybench.cli:main",
        }
        assert set(data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]) == {
            "src/operatebench",
            "src/boundarybench",
        }

    def test_the_build_backend_is_one_exact_reviewed_hatchling_release(self) -> None:
        import tomllib

        data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        assert data["build-system"] == {
            "requires": ["hatchling==1.32.0"],
            "build-backend": "hatchling.build",
        }

    @pytest.mark.parametrize(
        "requires",
        [
            '["hatchling>=1.27"]',
            '["hatchling==1.31.0"]',
            '["hatchling==1.32.0", "setuptools==80.0.0"]',
        ],
    )
    def test_release_scanner_rejects_build_backend_dependency_drift(
        self, tmp_path: Path, requires: str
    ) -> None:
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        pyproject = pyproject.replace(
            'requires = ["hatchling==1.32.0"]', f"requires = {requires}"
        )
        (tmp_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")

        problems = checks.check_distribution_metadata(tmp_path)

        assert any(
            "build-system.requires" in problem and "hatchling==1.32.0" in problem
            for problem in problems
        )

    def test_the_release_tooling_is_not_packaged_into_the_wheel(self) -> None:
        # tools/ is repository machinery. A wheel that carries the release
        # checker would ship the allowlist of synthetic key literals to users.
        import tomllib

        data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        packages = data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
        assert not any(package.startswith("tools") for package in packages)


class TestPackageVersions:
    def test_both_packages_state_their_own_version(self) -> None:
        assert checks.check_package_versions(REPO_ROOT) == []

    def test_the_hatch_dynamic_version_source_can_parse_the_version_file(self) -> None:
        # ``[tool.hatch.version] path = "src/operatebench/version.py"`` with no
        # explicit source is hatchling's "regex" source: it re.search()s the
        # file's *text* for a line matching DEFAULT_PATTERN — it never imports
        # the module. An AST-level check (check_package_versions above) can be
        # satisfied by a chained assignment like
        # ``__version__ = OPERATEBENCH_VERSION = "0.1.0"`` while the regex
        # source still fails to parse it, which is exactly what breaks
        # ``uv build``. This asserts the regex contract directly, and that the
        # value it yields is the same one the running module reports. It is the
        # *distribution* version throughout: the wheel is built from this line,
        # and the engine identity is a separate literal in the same file.
        from operatebench.version import __version__

        text = (REPO_ROOT / "src" / "operatebench" / "version.py").read_text(
            encoding="utf-8"
        )
        match = re.search(_HATCH_DEFAULT_VERSION_PATTERN, text, flags=re.MULTILINE)
        assert match is not None, (
            "hatchling's default version-source regex cannot find a "
            '`__version__ = "..."` line in src/operatebench/version.py'
        )
        assert match.group("version") == checks.DISTRIBUTION_VERSION
        assert __version__ == checks.DISTRIBUTION_VERSION

    def test_the_installed_modules_agree_with_the_files(self) -> None:
        import boundarybench
        import operatebench
        from operatebench.version import OPERATEBENCH_VERSION, __version__

        assert __version__ == checks.DISTRIBUTION_VERSION
        assert OPERATEBENCH_VERSION == checks.ENGINE_VERSION
        assert operatebench.OPERATEBENCH_VERSION == checks.ENGINE_VERSION
        assert boundarybench.__version__ == checks.BOUNDARY_PACKAGE_VERSION

    def test_the_boundary_track_keeps_its_historical_protocol_version(self) -> None:
        # The compatibility package is not renumbered into the OperateBench line.
        # Its manifest semantics are the historical ones: its own benchmark
        # version, its own manifest protocol version, and neither derived from
        # the distribution version they now ship inside.
        import boundarybench
        from boundarybench.manifest import MANIFEST_VERSION

        assert boundarybench.BENCHMARK_VERSION == "0.2.0-dev.1"
        assert boundarybench.BENCHMARK_VERSION != checks.DISTRIBUTION_VERSION
        assert MANIFEST_VERSION == 1
        assert boundarybench.__version__ != checks.DISTRIBUTION_VERSION


class TestLicensingAndGovernance:
    def test_every_required_file_is_present_and_reads_as_itself(self) -> None:
        assert checks.check_required_files(REPO_ROOT) == []

    def test_candidate_verification_and_lockfile_are_required_sources(self) -> None:
        source_required = dict(checks.SOURCE_CONTROL_FILES)
        required = dict(checks.REQUIRED_FILES)
        assert (
            source_required["PUBLIC_CANDIDATE_VERIFICATION.md"]
            == "Public candidate verification"
        )
        assert required["uv.lock"] == "version = 1"

    def test_no_contributor_copyright_assignment_is_asserted(self) -> None:
        for name in ("CONTRIBUTING.md", "LICENSE-DATA", "NOTICE"):
            text = (REPO_ROOT / name).read_text(encoding="utf-8").lower()
            assert "copyright assignment" not in text or "no copyright assignment" in text
            assert "assigns copyright" not in text

    def test_the_maintainer_contact_is_reachable_from_the_governance_files(self) -> None:
        for name in ("SECURITY.md", "CITATION.cff"):
            assert "dmitry@dwelly.group" in (REPO_ROOT / name).read_text(encoding="utf-8")

    def test_the_citation_does_not_claim_a_release_that_has_not_happened(self) -> None:
        # The key, not the word: the file explains in a comment why the field is
        # absent, and that explanation must not itself trip the check.
        import re

        text = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
        assert not re.search(r"^date-released\s*:", text, re.MULTILINE), (
            "date-released asserts a publication date; until the release actually "
            "happens the field is omitted rather than filled with today"
        )
        assert not re.search(r"^commit\s*:", text, re.MULTILINE)
        assert re.search(
            r"^version:\s*" + checks.DISTRIBUTION_VERSION, text, re.MULTILINE
        )


class TestNoPrivateContent:
    def test_no_credential_shape_outside_the_synthetic_allowlist(self) -> None:
        assert checks.check_no_credential_shapes(REPO_ROOT) == []

    def test_the_synthetic_key_allowlist_is_exact_rather_than_a_prefix(self) -> None:
        # A prefix allowlist would accept a real key that happens to start the
        # same way. Every entry must be a whole literal.
        for literal in checks.ALLOWED_SYNTHETIC_KEY_LITERALS:
            assert "not-a-real" in literal or "should-never-appear" in literal

    def test_the_allowlist_does_not_admit_an_arbitrary_key(self) -> None:
        assert "sk-ant-" + "a" * 40 not in checks.ALLOWED_SYNTHETIC_KEY_LITERALS

    def test_no_private_path_hostname_or_foreign_repository(self) -> None:
        assert checks.check_no_private_references(REPO_ROOT) == []

    def test_no_bare_internal_commit_identifier(self) -> None:
        assert checks.check_no_bare_commit_identifiers(REPO_ROOT) == []

    def test_no_non_public_document_reference(self) -> None:
        assert checks.check_no_internal_documents(REPO_ROOT) == []

    def test_no_private_audit_or_provider_run_provenance(self) -> None:
        assert checks.check_no_private_provenance(REPO_ROOT) == []


def _label(*parts: str) -> str:
    """A scanner rule label, assembled at run time.

    Several of the labels contain the very phrases their rules match. Written
    out they would make this file a finding, so they are joined rather than
    quoted — and the join is checked against the scanner's own table, so a
    renamed rule fails loudly here instead of silently matching nothing.
    """
    label = " ".join(parts)
    known = {name for name, _ in checks.PRIVATE_PROVENANCE_PATTERNS}
    assert label in known, f"no such provenance rule: {label!r}"
    return label


class TestPrivateProvenanceRules:
    """The provenance gate fails on each generic public-policy class.

    Every planted string, and every rule label asserted on, is assembled at run
    time from parts. A literal here would make this file the finding, which is
    how a scanner ends up with an exemption for the only test that proves it
    works.
    """

    def _finding(self, tmp_path: Path, body: str) -> list[str]:
        (tmp_path / "note.md").write_text(body, encoding="utf-8")
        return checks.check_no_private_provenance(tmp_path)

    def test_a_non_public_audit_claim_is_caught(self, tmp_path: Path) -> None:
        body = "Evidence came from " + " ".join(["an", "internal", "audit"]) + ".\n"
        problems = self._finding(tmp_path, body)
        assert any(_label("non-public", "audit", "claim") in p for p in problems)
        assert all("note.md:1" in problem for problem in problems)

    @pytest.mark.parametrize("qualifier", ["real", "live"])
    @pytest.mark.parametrize("prefix", ["Evidence came from ", ""])
    def test_a_positive_provider_execution_claim_is_caught(
        self, tmp_path: Path, qualifier: str, prefix: str
    ) -> None:
        article = "a" if prefix else "one"
        body = prefix + " ".join([article, qualifier, "provider", "run"])
        problems = self._finding(tmp_path, body + ".\n")
        assert any(
            _label("claimed", "live/real", "provider", "execution") in problem
            for problem in problems
        )

    def test_rule_vocabulary_and_honest_negation_are_not_findings(
        self, tmp_path: Path
    ) -> None:
        vocabulary = " ".join(["real", "provider", "run"])
        benign = f"Scanner vocabulary: {vocabulary}.\nno {vocabulary} artifact exists.\n"
        assert self._finding(tmp_path, benign) == []

    @pytest.mark.parametrize("visibility", ["confidential", "restricted", "proprietary"])
    @pytest.mark.parametrize("review", ["assessment", "evaluation", "validation"])
    def test_non_public_production_provider_validation_synonyms_are_caught(
        self, tmp_path: Path, visibility: str, review: str
    ) -> None:
        body = " ".join(
            [
                "A",
                visibility,
                review,
                "verified these benchmark findings using production-provider executions.",
            ]
        )
        problems = self._finding(tmp_path, body)
        assert any(
            _label("non-public", "production-provider", "validation", "claim") in p
            for p in problems
        )

    def test_private_diagnostic_aggregate_with_explicit_non_evidence_limit_is_allowed(
        self, tmp_path: Path
    ) -> None:
        body = (
            "A private diagnostic aggregate from provider executions is disclosed "
            "only as history, not benchmark capability or public evidence.\n"
        )
        assert self._finding(tmp_path, body) == []

    def test_an_external_or_private_path_is_caught(self, tmp_path: Path) -> None:
        body = "Evidence is stored at " + " ".join(["an", "external", "path"]) + ".\n"
        assert any(
            _label("external/private", "path") in problem
            for problem in self._finding(tmp_path, body)
        )

    @pytest.mark.parametrize("noun", ["configuration", "evidence"])
    def test_a_non_public_run_reference_is_caught(
        self, tmp_path: Path, noun: str
    ) -> None:
        body = "See the " + " ".join(["non-public", "run", noun, "reference"]) + ".\n"
        assert any(
            _label("non-public", "run", "configuration/evidence", "reference") in problem
            for problem in self._finding(tmp_path, body)
        )

    def test_a_non_public_configuration_identifier_is_caught(
        self, tmp_path: Path
    ) -> None:
        body = "the " + " ".join(["private", "run", "configuration", "id"]) + " is set\n"
        assert any(
            _label("non-public", "run", "configuration/evidence", "reference") in problem
            for problem in self._finding(tmp_path, body)
        )

    def test_a_bare_content_digest_is_not_a_finding(self, tmp_path: Path) -> None:
        (tmp_path / "digest.md").write_text("content_digest: " + "ab" * 32 + "\n")
        assert checks.check_no_private_provenance(tmp_path) == []

    def test_the_policy_document_exemption_is_by_exact_path_only(self) -> None:
        # The exemption exists because those files enumerate the forbidden
        # classes by name. It must never widen into "documents are exempt".
        assert "tools/check_public_release.py" in checks.PROVENANCE_POLICY_DOCUMENTS
        assert "tests/test_public_release.py" not in checks.PROVENANCE_POLICY_DOCUMENTS
        for entry in checks.PROVENANCE_POLICY_DOCUMENTS:
            assert (REPO_ROOT / entry).is_file(), entry
            assert not entry.endswith("/")


class TestPublicationManifestContentIdentity:
    def test_spec_and_oracle_digests_match_the_executable_content(self) -> None:
        import json

        from operatebench.domains.lettings.maintenance.oracle import (
            negative_control_oracle,
        )
        from operatebench.domains.lettings.maintenance.spec import load_spec

        manifest = json.loads((REPO_ROOT / "PUBLICATION_MANIFEST.json").read_text())
        digests = manifest["release"]["content_digests"]
        spec = load_spec(REPO_ROOT / "examples/operatebench/maintenance_v0_1.yaml")
        oracle = negative_control_oracle()
        assert digests["operation_fixture_spec_digest_sha256"] == (
            spec.spec_digest_sha256
        )
        assert digests["negative_control_oracle_id"] == oracle.oracle_id
        assert digests["negative_control_oracle_version"] == oracle.oracle_version
        assert digests["negative_control_oracle_digest_sha256"] == (
            oracle.oracle_digest_sha256
        )


class TestBuiltDistributions:
    def test_the_built_wheel_and_sdist_carry_no_private_marker(self) -> None:
        # Passes silently when dist/ has not been built: a release gate that
        # demands a build step it cannot perform is one people stop running.
        assert checks.check_built_distributions(REPO_ROOT) == []

    def test_an_extracted_sdist_rejects_the_test_only_matched_grammar(
        self, tmp_path: Path
    ) -> None:
        helper = tmp_path / "tests" / "matched_control_fixtures.py"
        helper.parent.mkdir()
        helper.write_text("test-only grammar\n", encoding="utf-8")
        assert checks.check_no_distribution_test_helpers(tmp_path, surface="sdist") == [
            "tests/matched_control_fixtures.py: test-only helper shipped in sdist"
        ]

    def test_a_planted_marker_inside_a_wheel_is_caught(self, tmp_path: Path) -> None:
        import zipfile

        distribution = tmp_path / "dist"
        _empty_distribution(distribution)
        planted = "recorded in " + " ".join(["an", "internal", "audit"])
        with zipfile.ZipFile(distribution / "pkg-0.1.0-py3-none-any.whl", "w") as wheel:
            wheel.writestr("pkg/leaky.py", f'"""{planted}."""\n')
        problems = checks.check_built_distributions(tmp_path)
        assert any("pkg/leaky.py" in problem for problem in problems)
        assert any(_label("non-public", "audit", "claim") in p for p in problems)

    @pytest.mark.parametrize(
        ("member", "allowed"),
        [
            pytest.param(".github/workflows/ci.yml", True, id="canonical-root"),
            pytest.param(".github/workflows/ci.yaml", True, id="canonical-yaml-root"),
            pytest.param(".github//workflows/ci.yml", False, id="repeated-separator"),
            pytest.param("./.github/workflows/ci.yml", False, id="leading-dot"),
            pytest.param(".github/workflows/./ci.yml", False, id="dot-component"),
            pytest.param(".github/workflows/../ci.yml", False, id="dotdot-component"),
            pytest.param("/.github/workflows/ci.yml", False, id="leading-slash"),
            pytest.param(".github/workflows/ci.yml/", False, id="trailing-slash"),
            pytest.param(".github\\workflows\\ci.yml", False, id="backslash-separator"),
            pytest.param("pkg/.github/workflows/ci.yml", False, id="nested-package"),
            pytest.param(".github/workflows/ci.YML", False, id="uppercase-suffix"),
            pytest.param(".github/workflows/ci.Yml", False, id="mixed-case-suffix"),
        ],
    )
    def test_a_wheel_only_allows_action_pins_in_a_root_canonical_workflow(
        self, tmp_path: Path, member: str, allowed: bool
    ) -> None:
        import zipfile

        distribution = tmp_path / "dist"
        _empty_distribution(distribution)
        token = "ab" * 20
        payload = f"steps:\n  - uses: actions/checkout@{token} # v4\n"
        with zipfile.ZipFile(distribution / "pkg-0.1.0-py3-none-any.whl", "w") as wheel:
            wheel.writestr(member, payload)

        problems = checks.check_built_distributions(tmp_path)

        assert (problems == []) is allowed
        if not allowed:
            assert any(member in problem for problem in problems)
            assert any(
                "bare 40-hex commit identifier" in problem
                or "unsafe wheel member" in problem
                or "directory carries data" in problem
                or "not regular" in problem
                for problem in problems
            )

    @pytest.mark.parametrize(
        ("member", "allowed"),
        [
            pytest.param(
                "pkg-0.1.0/.github/workflows/ci.yml", True, id="canonical-under-root"
            ),
            pytest.param(
                "pkg-0.1.0/.github/workflows/ci.yaml",
                True,
                id="canonical-yaml-under-root",
            ),
            pytest.param(
                "pkg-0.1.0/.github//workflows/ci.yml",
                False,
                id="repeated-separator",
            ),
            pytest.param("pkg-0.1.0/./.github/workflows/ci.yml", False, id="leading-dot"),
            pytest.param(
                "pkg-0.1.0/.github/workflows/./ci.yml", False, id="dot-component"
            ),
            pytest.param(
                "pkg-0.1.0/.github/workflows/../ci.yml",
                False,
                id="dotdot-component",
            ),
            pytest.param("/.github/workflows/ci.yml", False, id="leading-slash"),
            pytest.param(
                "pkg-0.1.0/.github/workflows/ci.yml/", False, id="trailing-slash"
            ),
            pytest.param(
                "pkg-0.1.0/.github\\workflows\\ci.yml",
                False,
                id="backslash-separator",
            ),
            pytest.param(
                "pkg-0.1.0/pkg/.github/workflows/ci.yml", False, id="nested-package"
            ),
            pytest.param(
                "pkg-0.1.0/.github/workflows/ci.YML", False, id="uppercase-suffix"
            ),
            pytest.param(
                "pkg-0.1.0/.github/workflows/ci.Yml", False, id="mixed-case-suffix"
            ),
        ],
    )
    def test_an_sdist_only_strips_its_validated_root_for_workflow_pins(
        self, tmp_path: Path, member: str, allowed: bool
    ) -> None:
        import io
        import tarfile

        distribution = tmp_path / "dist"
        _empty_distribution(distribution)
        token = "ab" * 20
        payload = f"steps:\n  - uses: actions/checkout@{token} # v4\n".encode()
        with tarfile.open(distribution / "pkg-0.1.0.tar.gz", "w:gz") as sdist:
            info = tarfile.TarInfo(member)
            info.size = len(payload)
            sdist.addfile(info, io.BytesIO(payload))

        problems = checks.check_built_distributions(tmp_path)

        assert (problems == []) is allowed
        if not allowed:
            assert any(member in problem for problem in problems)
            assert any(
                "bare 40-hex commit identifier" in problem
                or "unsafe sdist member" in problem
                for problem in problems
            )

    @pytest.mark.parametrize(
        "other_member",
        [
            pytest.param("other-root/metadata.txt", id="multiple-roots"),
            pytest.param("../metadata.txt", id="malformed-member"),
        ],
    )
    def test_an_sdist_without_one_safe_root_does_not_exempt_a_workflow_pin(
        self, tmp_path: Path, other_member: str
    ) -> None:
        import io
        import tarfile

        distribution = tmp_path / "dist"
        _empty_distribution(distribution)
        token = "ab" * 20
        member = "pkg-0.1.0/.github/workflows/ci.yml"
        workflow = f"steps:\n  - uses: actions/checkout@{token} # v4\n".encode()
        with tarfile.open(distribution / "pkg-0.1.0.tar.gz", "w:gz") as sdist:
            for name, payload in ((member, workflow), (other_member, b"metadata\n")):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                sdist.addfile(info, io.BytesIO(payload))

        problems = checks.check_built_distributions(tmp_path)

        if other_member.startswith("other-root/"):
            assert any(
                member in problem and "bare 40-hex commit identifier" in problem
                for problem in problems
            )
        assert any(
            "exactly one safe root" in problem or "unsafe sdist member" in problem
            for problem in problems
        )

    @pytest.mark.parametrize(
        "token",
        [
            pytest.param(("ab" * 20).upper(), id="uppercase"),
            pytest.param("aB" * 20, id="mixed-case"),
        ],
    )
    @pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
    def test_distribution_workflow_exemptions_require_lowercase_action_pins(
        self, tmp_path: Path, token: str, archive_kind: str
    ) -> None:
        import io
        import tarfile
        import zipfile

        distribution = tmp_path / "dist"
        _empty_distribution(distribution)
        payload = f"steps:\n  - uses: actions/checkout@{token} # v4\n"
        if archive_kind == "wheel":
            archive = distribution / "pkg-0.1.0-py3-none-any.whl"
            member = ".github/workflows/ci.yml"
            with zipfile.ZipFile(archive, "w") as wheel:
                wheel.writestr(member, payload)
        else:
            archive = distribution / "pkg-0.1.0.tar.gz"
            member = "pkg-0.1.0/.github/workflows/ci.yml"
            encoded = payload.encode()
            with tarfile.open(archive, "w:gz") as sdist:
                info = tarfile.TarInfo(member)
                info.size = len(encoded)
                sdist.addfile(info, io.BytesIO(encoded))

        problems = checks.check_built_distributions(tmp_path)

        assert len(problems) == 1
        assert member in problems[0]
        assert "bare 40-hex commit identifier" in problems[0]

    @pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
    def test_distribution_workflow_rejects_an_unapproved_action_owner(
        self, tmp_path: Path, archive_kind: str
    ) -> None:
        import io
        import tarfile
        import zipfile

        distribution = tmp_path / "dist"
        _empty_distribution(distribution)
        token = "ab" * 20
        payload = f"steps:\n  - uses: UnapprovedOwner/action@{token} # v4\n"
        if archive_kind == "wheel":
            archive = distribution / "pkg-0.1.0-py3-none-any.whl"
            member = ".github/workflows/ci.yml"
            with zipfile.ZipFile(archive, "w") as wheel:
                wheel.writestr(member, payload)
        else:
            archive = distribution / "pkg-0.1.0.tar.gz"
            member = "pkg-0.1.0/.github/workflows/ci.yml"
            encoded = payload.encode()
            with tarfile.open(archive, "w:gz") as sdist:
                info = tarfile.TarInfo(member)
                info.size = len(encoded)
                sdist.addfile(info, io.BytesIO(encoded))

        problems = checks.check_built_distributions(tmp_path)

        assert len(problems) == 1
        assert member in problems[0]
        assert "bare 40-hex commit identifier" in problems[0]

    @pytest.mark.parametrize(
        "token",
        [
            pytest.param("ab" * 20, id="lowercase"),
            pytest.param(("ab" * 20).upper(), id="uppercase"),
            pytest.param("aB" * 20, id="mixed-case"),
        ],
    )
    def test_a_distribution_rejects_commit_identifiers_in_arbitrary_fields(
        self, tmp_path: Path, token: str
    ) -> None:
        import zipfile

        distribution = tmp_path / "dist"
        _empty_distribution(distribution)
        with zipfile.ZipFile(distribution / "pkg-0.1.0-py3-none-any.whl", "w") as wheel:
            wheel.writestr("pkg/metadata.json", f'{{"source": "{token}"}}\n')

        problems = checks.check_built_distributions(tmp_path)

        assert len(problems) == 1
        assert "pkg/metadata.json:1" in problems[0]
        assert "bare 40-hex commit identifier" in problems[0]

    def test_a_nested_scanner_lookalike_inside_a_wheel_is_caught(
        self, tmp_path: Path
    ) -> None:
        import zipfile

        distribution = tmp_path / "dist"
        _empty_distribution(distribution)
        marker = "recorded in " + " ".join(["an", "internal", "audit"])
        member = "pkg/evil/tools/check_public_release.py"
        with zipfile.ZipFile(distribution / "pkg-0.1.0-py3-none-any.whl", "w") as wheel:
            wheel.writestr(member, marker)

        problems = checks.check_built_distributions(tmp_path)

        assert any(member in problem for problem in problems)
        assert any(_label("non-public", "audit", "claim") in p for p in problems)

    def test_cff_member_in_wheel_is_scanned_without_reading_binary_data(
        self, tmp_path: Path
    ) -> None:
        import zipfile

        distribution = tmp_path / "dist"
        _empty_distribution(distribution)
        credential = "sk-ant-" + "a" * 40
        private_marker = "recorded in " + " ".join(["an", "internal", "audit"])
        payload = f"message: {private_marker}\ntoken: {credential}\n"
        with zipfile.ZipFile(distribution / "pkg-0.1.0-py3-none-any.whl", "w") as wheel:
            wheel.writestr("pkg/CITATION.cff", payload)
            wheel.writestr("pkg/null-byte.bin", b"\x00" + payload.encode())
            wheel.writestr("pkg/invalid-utf8.bin", b"\xff" + payload.encode())

        problems = checks.check_built_distributions(tmp_path)

        assert len(problems) == 2
        assert all("pkg/CITATION.cff" in problem for problem in problems)
        assert any(_label("non-public", "audit", "claim") in p for p in problems)
        assert any("Anthropic-style API key" in problem for problem in problems)

    def test_sdist_scans_lockfiles_and_future_text_formats(self, tmp_path: Path) -> None:
        import io
        import tarfile

        distribution = tmp_path / "dist"
        _empty_distribution(distribution)
        credential = "token: sk-ant-" + "a" * 40
        private_path = "see /" + "var/www/private/run.log"
        payload = f"{credential}\n{private_path}\n".encode()
        with tarfile.open(distribution / "pkg-0.1.0.tar.gz", "w:gz") as sdist:
            for member in ("pkg-0.1.0/uv.lock", "pkg-0.1.0/data.future-text"):
                info = tarfile.TarInfo(member)
                info.size = len(payload)
                sdist.addfile(info, io.BytesIO(payload))

        problems = checks.check_built_distributions(tmp_path)

        assert len(problems) == 4
        for member in ("pkg-0.1.0/uv.lock", "pkg-0.1.0/data.future-text"):
            member_problems = [problem for problem in problems if member in problem]
            assert len(member_problems) == 2
            assert any(
                "Anthropic-style API key" in problem for problem in member_problems
            )
            assert any("absolute local path" in problem for problem in member_problems)

    def test_a_planted_credential_inside_an_sdist_is_caught(self, tmp_path: Path) -> None:
        import io
        import tarfile

        distribution = tmp_path / "dist"
        _empty_distribution(distribution)
        payload = ("token: sk-ant-" + "a" * 40 + "\n").encode()
        with tarfile.open(distribution / "pkg-0.1.0.tar.gz", "w:gz") as sdist:
            info = tarfile.TarInfo("pkg-0.1.0/leaked.md")
            info.size = len(payload)
            sdist.addfile(info, io.BytesIO(payload))
        problems = checks.check_built_distributions(tmp_path)
        assert len(problems) == 1
        assert "pkg-0.1.0/leaked.md:1" in problems[0]
        assert "Anthropic-style API key" in problems[0]

    def test_the_synthetic_key_allowlist_still_applies_inside_a_distribution(
        self, tmp_path: Path
    ) -> None:
        import io
        import tarfile

        distribution = tmp_path / "dist"
        _empty_distribution(distribution)
        literal = sorted(checks.ALLOWED_SYNTHETIC_KEY_LITERALS)[0]
        payload = f"key = {literal!r}\n".encode()
        with tarfile.open(distribution / "pkg-0.1.0.tar.gz", "w:gz") as sdist:
            info = tarfile.TarInfo("pkg-0.1.0/tests/trust.py")
            info.size = len(payload)
            sdist.addfile(info, io.BytesIO(payload))
        assert checks.check_built_distributions(tmp_path) == []

    def test_a_nested_scanner_lookalike_inside_an_sdist_is_caught(
        self, tmp_path: Path
    ) -> None:
        import io
        import tarfile

        distribution = tmp_path / "dist"
        _empty_distribution(distribution)
        marker = "recorded in " + " ".join(["an", "internal", "audit"])
        payload = marker.encode()
        member = "pkg-0.1.0/evil/tools/check_public_release.py"
        with tarfile.open(distribution / "pkg-0.1.0.tar.gz", "w:gz") as sdist:
            info = tarfile.TarInfo(member)
            info.size = len(payload)
            sdist.addfile(info, io.BytesIO(payload))

        problems = checks.check_built_distributions(tmp_path)

        assert any(member in problem for problem in problems)
        assert any(_label("non-public", "audit", "claim") in p for p in problems)

    def test_the_scanner_source_is_not_mistaken_for_run_provenance(
        self, tmp_path: Path
    ) -> None:
        import io
        import tarfile

        distribution = tmp_path / "dist"
        _empty_distribution(distribution)
        provenance_source = (
            "PRIVATE_AUDIT_PATTERN = '" + "private" + " " + "audit" + "'\n"
        )
        credential = "token: sk-ant-" + "a" * 40 + "\n"
        private_path = "see /" + "var/www/private/run.log\n"
        payload = (provenance_source + credential + private_path).encode()
        member = "pkg-0.1.0/tools/check_public_release.py"
        with tarfile.open(distribution / "pkg-0.1.0.tar.gz", "w:gz") as sdist:
            info = tarfile.TarInfo(member)
            info.size = len(payload)
            sdist.addfile(info, io.BytesIO(payload))

        problems = checks.check_built_distributions(tmp_path)

        assert len(problems) == 2
        assert all(member in problem for problem in problems)
        assert any("Anthropic-style API key" in problem for problem in problems)
        assert any("absolute local path" in problem for problem in problems)
        assert not any(_label("non-public", "audit", "claim") in p for p in problems)

    def test_an_absent_dist_directory_is_not_a_failure(self, tmp_path: Path) -> None:
        assert checks.check_built_distributions(tmp_path) == []

    def test_an_explicit_absent_distribution_directory_fails_closed(
        self, tmp_path: Path
    ) -> None:
        problems = checks.check_built_distributions(
            REPO_ROOT, distribution_dir=tmp_path / "missing"
        )
        assert any("distribution directory" in problem for problem in problems)

    @pytest.mark.parametrize("explicit", [False, True])
    @pytest.mark.parametrize("suffix", [".whl", ".tar.gz"])
    def test_existing_distribution_requires_exactly_one_archive_of_each_kind(
        self, tmp_path: Path, explicit: bool, suffix: str
    ) -> None:
        distribution = tmp_path / "dist"
        _canonical_distribution(distribution)
        (distribution / f"duplicate{suffix}").write_bytes(b"duplicate")
        problems = checks.check_built_distributions(
            tmp_path if not explicit else REPO_ROOT,
            distribution_dir=distribution if explicit else None,
        )
        assert any("exactly one wheel and one sdist" in problem for problem in problems)

    @pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
    def test_incomplete_distribution_reports_cardinality_and_archive_content(
        self, tmp_path: Path, archive_kind: str
    ) -> None:
        distribution = tmp_path / "dist"
        distribution.mkdir()
        marker = "recorded in " + " ".join(["an", "internal", "audit"])
        payload = f"{marker}\n".encode()
        if archive_kind == "wheel":
            with zipfile.ZipFile(
                distribution / "pkg-0.1.0-py3-none-any.whl", "w"
            ) as wheel:
                wheel.writestr("pkg/leaked.md", payload)
        else:
            with tarfile.open(distribution / "pkg-0.1.0.tar.gz", "w:gz") as sdist:
                member = tarfile.TarInfo("pkg-0.1.0/leaked.md")
                member.size = len(payload)
                sdist.addfile(member, io.BytesIO(payload))

        problems = checks.check_built_distributions(tmp_path)

        assert any("exactly one wheel and one sdist" in problem for problem in problems)
        assert any("leaked.md" in problem for problem in problems)
        assert any(_label("non-public", "audit", "claim") in p for p in problems)

    @pytest.mark.parametrize(
        ("kind", "renamed"),
        [
            ("wheel", "OperateBench-0.1.0-py3-none-any.whl"),
            ("sdist", "OperateBench-0.1.0.tar.gz"),
        ],
    )
    def test_archive_filenames_preserve_reviewed_distribution_identity(
        self, tmp_path: Path, kind: str, renamed: str
    ) -> None:
        distribution = tmp_path / "dist"
        wheel, sdist = _canonical_distribution(distribution)
        (wheel if kind == "wheel" else sdist).rename(distribution / renamed)
        problems = checks.check_built_distributions(
            REPO_ROOT, distribution_dir=distribution
        )
        assert any("archive filename identity differs" in problem for problem in problems)

    def test_distribution_directory_symlink_is_rejected(self, tmp_path: Path) -> None:
        target = tmp_path / "real-dist"
        _canonical_distribution(target)
        linked = tmp_path / "linked-dist"
        linked.symlink_to(target, target_is_directory=True)
        problems = checks.check_built_distributions(REPO_ROOT, distribution_dir=linked)
        assert any("symlink" in problem for problem in problems)

    @pytest.mark.parametrize("suffix", [".whl", ".tar.gz"])
    def test_archive_symlink_is_rejected(self, tmp_path: Path, suffix: str) -> None:
        real = tmp_path / "real"
        wheel, sdist = _canonical_distribution(real)
        attacked = tmp_path / "dist"
        attacked.mkdir()
        for archive in (wheel, sdist):
            destination = attacked / archive.name
            if archive.name.endswith(suffix):
                destination.symlink_to(archive)
            else:
                destination.write_bytes(archive.read_bytes())
        problems = checks.check_built_distributions(REPO_ROOT, distribution_dir=attacked)
        assert any("archive is not a regular file" in problem for problem in problems)

    @pytest.mark.parametrize("suffix", [".whl", ".tar.gz"])
    def test_oversized_archive_is_rejected_before_parser_construction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
    ) -> None:
        distribution = tmp_path / "dist"
        wheel, sdist = _canonical_distribution(distribution)
        oversized = wheel if suffix == ".whl" else sdist
        with oversized.open("r+b") as handle:
            handle.truncate(64 * 1024 * 1024 + 1)

        def parser_must_not_run(*args: object, **kwargs: object) -> None:
            raise AssertionError("archive parser was constructed")

        if suffix == ".whl":
            monkeypatch.setattr(zipfile, "ZipFile", parser_must_not_run)
        else:
            monkeypatch.setattr(tarfile, "open", parser_must_not_run)
        problems = checks.check_built_distributions(
            REPO_ROOT, distribution_dir=distribution
        )
        assert any("compressed size limit" in problem for problem in problems)

    @pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
    def test_huge_archive_member_name_has_bounded_diagnostic(
        self, tmp_path: Path, archive_kind: str
    ) -> None:
        distribution = tmp_path / "dist"
        _empty_distribution(distribution)
        huge_name = "n" * 60_000 + ".py"
        if archive_kind == "wheel":
            archive = distribution / "pkg-0.1.0-py3-none-any.whl"
            with zipfile.ZipFile(archive, "w") as wheel:
                wheel.writestr(huge_name, b"payload")
        else:
            archive = distribution / "pkg-0.1.0.tar.gz"
            with tarfile.open(archive, "w:gz") as sdist:
                member = tarfile.TarInfo(huge_name)
                member.size = 7
                sdist.addfile(member, io.BytesIO(b"payload"))

        problems = checks.check_built_distributions(
            tmp_path, distribution_dir=distribution
        )
        assert problems
        assert any("member name exceeds limit" in problem for problem in problems)
        assert all(len(problem) <= 512 for problem in problems)
        assert all(len(problem.encode("utf-8")) <= 512 for problem in problems)
        assert huge_name not in "".join(problems)

    def test_tar_member_limit_is_enforced_without_getmembers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        distribution = tmp_path / "dist"
        _empty_distribution(distribution)
        archive = distribution / "pkg-0.1.0.tar.gz"
        with tarfile.open(archive, "w:gz") as sdist:
            for index in range(10_001):
                member = tarfile.TarInfo(f"pkg-0.1.0/member-{index:05d}")
                member.size = 0
                sdist.addfile(member, io.BytesIO())

        def getmembers_must_not_run(self: tarfile.TarFile) -> list[tarfile.TarInfo]:
            raise AssertionError("getmembers materialized the archive index")

        monkeypatch.setattr(tarfile.TarFile, "getmembers", getmembers_must_not_run)
        problems = checks.check_built_distributions(
            tmp_path, distribution_dir=distribution
        )
        assert any("sdist exceeds member limit 10000" in problem for problem in problems)

    @pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
    @pytest.mark.parametrize("violation", ["member", "aggregate"])
    def test_declared_size_limits_reject_before_member_payload_read(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        archive_kind: str,
        violation: str,
    ) -> None:
        distribution = tmp_path / "dist"
        _empty_distribution(distribution)
        chunk_size = 16 * 1024 * 1024 + 1 if violation == "member" else 15 * 1024 * 1024
        member_count = 1 if violation == "member" else 9
        chunk = b"\x00" * chunk_size
        if archive_kind == "wheel":
            archive = distribution / "pkg-0.1.0-py3-none-any.whl"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
                wheel.writestr("pkg/benign.txt", b"benign")
                for index in range(member_count):
                    wheel.writestr(f"pkg/large-{index}.bin", chunk)

            def read_must_not_run(*args: object, **kwargs: object) -> bytes:
                raise AssertionError("ZIP member payload was read")

            monkeypatch.setattr(zipfile.ZipFile, "read", read_must_not_run)
        else:
            archive = distribution / "pkg-0.1.0.tar.gz"
            with tarfile.open(archive, "w:gz") as sdist:
                for index in range(member_count):
                    member = tarfile.TarInfo(f"pkg-0.1.0/large-{index}.bin")
                    member.size = chunk_size
                    sdist.addfile(member, io.BytesIO(chunk))

            def extractfile_must_not_run(*args: object, **kwargs: object) -> None:
                raise AssertionError("tar member payload was read")

            monkeypatch.setattr(tarfile.TarFile, "extractfile", extractfile_must_not_run)

        problems = checks.check_built_distributions(
            tmp_path, distribution_dir=distribution
        )
        expected = (
            "member exceeds size limit"
            if violation == "member"
            else "uncompressed size limit"
        )
        assert any(expected in problem for problem in problems)

    def test_zip_member_limit_is_checked_before_payload_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        distribution = tmp_path / "dist"
        _empty_distribution(distribution)
        archive = distribution / "pkg-0.1.0-py3-none-any.whl"
        with zipfile.ZipFile(archive, "w") as wheel:
            for index in range(10_001):
                wheel.writestr(f"pkg/member-{index:05d}", b"")

        def read_must_not_run(*args: object, **kwargs: object) -> bytes:
            raise AssertionError("ZIP member payload was read")

        monkeypatch.setattr(zipfile.ZipFile, "read", read_must_not_run)
        problems = checks.check_built_distributions(
            tmp_path, distribution_dir=distribution
        )
        assert any("wheel exceeds member limit 10000" in problem for problem in problems)

    def test_distribution_findings_are_capped_and_deterministic(
        self, tmp_path: Path
    ) -> None:
        distribution = tmp_path / "dist"
        _empty_distribution(distribution)
        archive = distribution / "pkg-0.1.0-py3-none-any.whl"
        with zipfile.ZipFile(archive, "w") as wheel:
            for index in range(2_000):
                wheel.writestr(f"../unsafe-{index:05d}", b"")

        first = checks.check_built_distributions(tmp_path, distribution_dir=distribution)
        second = checks.check_built_distributions(tmp_path, distribution_dir=distribution)
        assert first == second
        assert len(first) == 257
        assert first[-1] == "additional_distribution_findings_suppressed"
        assert all(len(problem) <= 512 for problem in first)
        assert all(len(problem.encode("utf-8")) <= 512 for problem in first)

    @pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
    def test_malformed_archive_diagnostic_is_fixed_and_bounded(
        self, tmp_path: Path, archive_kind: str
    ) -> None:
        distribution = tmp_path / "dist"
        _empty_distribution(distribution)
        attacker_text = "raw-attacker-archive-content-" + "x" * 2_000
        archive = distribution / (
            "pkg-0.1.0-py3-none-any.whl"
            if archive_kind == "wheel"
            else "pkg-0.1.0.tar.gz"
        )
        archive.write_bytes(attacker_text.encode())

        first = checks.check_built_distributions(tmp_path, distribution_dir=distribution)
        second = checks.check_built_distributions(tmp_path, distribution_dir=distribution)
        assert first == second
        assert any(
            "malformed wheel" in problem or "malformed sdist" in problem
            for problem in first
        )
        assert attacker_text not in "".join(first)
        assert all(len(problem) <= 512 for problem in first)
        assert all(len(problem.encode("utf-8")) <= 512 for problem in first)

    def test_both_public_packages_declare_pep561_typing(self) -> None:
        for package in ("operatebench", "boundarybench"):
            marker = REPO_ROOT / "src" / package / "py.typed"
            assert marker.is_file(), f"{package} has no PEP 561 marker"
            assert marker.read_bytes() == b""

    def test_a_canonical_distribution_matches_the_reviewed_source(
        self, tmp_path: Path
    ) -> None:
        distribution = tmp_path / "dist"
        _canonical_distribution(distribution)
        assert (
            checks.check_built_distributions(REPO_ROOT, distribution_dir=distribution)
            == []
        )

    @pytest.mark.parametrize(
        "member",
        [
            "operatebench_release_probe.pth",
            "OperateBench/__init__.py",
            "operatebench\N{KELVIN SIGN}/__init__.py",
            "operatebench-0.1.0.data/purelib/injected.py",
            "injected.py",
            "attacker-1.0.dist-info/METADATA",
        ],
    )
    def test_wheel_rejects_every_member_outside_the_reviewed_universe(
        self, tmp_path: Path, member: str
    ) -> None:
        distribution = tmp_path / "dist"
        wheel, _ = _canonical_distribution(distribution)
        _rewrite_wheel(wheel, append=(member, b"import pathlib\n"))
        _regenerate_record(wheel)
        problems = checks.check_built_distributions(
            REPO_ROOT, distribution_dir=distribution
        )
        assert any("wheel member universe differs" in problem for problem in problems)

    def test_wheel_rejects_directory_entries_even_when_their_files_are_reviewed(
        self, tmp_path: Path
    ) -> None:
        distribution = tmp_path / "dist"
        wheel, _ = _canonical_distribution(distribution)
        _rewrite_wheel(wheel, append=("operatebench/", b""))
        problems = checks.check_built_distributions(
            REPO_ROOT, distribution_dir=distribution
        )
        assert any("wheel member is not regular" in problem for problem in problems)

    @pytest.mark.parametrize(
        "payload",
        [
            b"not-valid-wheel-metadata\n",
            b"\xff\xfe\x00malformed",
            b"Generator: hatchling 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            b"Wheel-Version: 1.1\nGenerator: hatchling 1.0\n"
            b"Root-Is-Purelib: true\nTag: py3-none-any\n",
            b"Wheel-Version: 1.0\nWheel-Version: 1.0\nGenerator: hatchling 1.0\n"
            b"Root-Is-Purelib: true\nTag: py3-none-any\n",
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            b"Wheel-Version: 1.0\nGenerator: setuptools 80\n"
            b"Root-Is-Purelib: true\nTag: py3-none-any\n",
            b"Wheel-Version: 1.0\nGenerator: hatchling 1.0\n"
            b"Generator: hatchling 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            b"Wheel-Version: 1.0\nGenerator: hatchling 1.0\nTag: py3-none-any\n",
            b"Wheel-Version: 1.0\nGenerator: hatchling 1.0\n"
            b"Root-Is-Purelib: false\nTag: py3-none-any\n",
            b"Wheel-Version: 1.0\nGenerator: hatchling 1.0\n"
            b"Root-Is-Purelib: true\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            b"Wheel-Version: 1.0\nGenerator: hatchling 1.0\nRoot-Is-Purelib: true\n",
            b"Wheel-Version: 1.0\nGenerator: hatchling 1.0\n"
            b"Root-Is-Purelib: true\nTag: cp311-cp311-linux_x86_64\n",
            b"Wheel-Version: 1.0\nGenerator: hatchling 1.0\n"
            b"Root-Is-Purelib: true\nTag: py3-none-any\nTag: py2-none-any\n",
            b"Wheel-Version: 1.0\nGenerator: hatchling 1.0\n"
            b"Root-Is-Purelib: true\nTag: py3-none-any\nBuild: 1\n",
            b"Wheel-Version: 1.0\nGenerator: hatchling 1.31.0\n"
            b"Root-Is-Purelib: true\nTag: py3-none-any\n",
        ],
    )
    def test_wheel_metadata_is_exact_and_bound_to_the_reviewed_filename(
        self, tmp_path: Path, payload: bytes
    ) -> None:
        distribution = tmp_path / "dist"
        wheel, _ = _canonical_distribution(distribution)
        name = "operatebench-0.1.0.dist-info/WHEEL"
        _rewrite_wheel(wheel, replace={name: payload})
        _regenerate_record(wheel)
        problems = checks.check_built_distributions(
            REPO_ROOT, distribution_dir=distribution
        )
        assert any("invalid WHEEL" in problem for problem in problems)

    @pytest.mark.parametrize(
        ("needle", "replacement"),
        [
            (b"Metadata-Version: 2.5\n", b"Metadata-Version: 2.4\n"),
            (
                b"Maintainer-email: Dmitry Khanukov <dmitry@dwelly.group>\n",
                b"",
            ),
            (
                b"Maintainer-email: Dmitry Khanukov <dmitry@dwelly.group>\n",
                b"Maintainer-email: Attacker <attacker@example.com>\n",
            ),
            (
                b"Keywords: agent-benchmark,business-operations,deterministic-replay,"
                b"event-driven,human-in-the-loop,synthetic-fixtures\n",
                b"",
            ),
            (
                b"Keywords: agent-benchmark,business-operations,deterministic-replay,"
                b"event-driven,human-in-the-loop,synthetic-fixtures\n",
                b"Keywords: attacker-controlled\n",
            ),
            (b"Description-Content-Type: text/markdown\n", b""),
            (
                b"Description-Content-Type: text/markdown\n",
                b"Description-Content-Type: text/plain\n",
            ),
        ],
    )
    def test_wheel_metadata_requires_the_complete_reviewed_header_projection(
        self, tmp_path: Path, needle: bytes, replacement: bytes
    ) -> None:
        distribution = tmp_path / "dist"
        wheel, _ = _canonical_distribution(distribution)
        name = "operatebench-0.1.0.dist-info/METADATA"
        with zipfile.ZipFile(wheel) as archive:
            metadata = archive.read(name)
        assert needle in metadata
        _rewrite_wheel(wheel, replace={name: metadata.replace(needle, replacement, 1)})
        _regenerate_record(wheel)
        problems = checks.check_built_distributions(
            REPO_ROOT, distribution_dir=distribution
        )
        assert any("invalid METADATA" in problem for problem in problems)

    @pytest.mark.parametrize("mutation", ["missing", "changed"])
    def test_wheel_licenses_are_exact_reviewed_members(
        self, tmp_path: Path, mutation: str
    ) -> None:
        distribution = tmp_path / "dist"
        wheel, _ = _canonical_distribution(distribution)
        name = "operatebench-0.1.0.dist-info/licenses/LICENSE"
        if mutation == "missing":
            _rewrite_wheel(wheel, remove={name})
        else:
            _rewrite_wheel(wheel, replace={name: b"not the reviewed license\n"})
        _regenerate_record(wheel)
        problems = checks.check_built_distributions(
            REPO_ROOT, distribution_dir=distribution
        )
        expected = (
            "wheel member universe differs" if mutation == "missing" else "bytes differ"
        )
        assert any(expected in problem for problem in problems)

    @pytest.mark.parametrize("body", [b"", b"reviewed body replaced\n"])
    def test_wheel_metadata_body_is_the_reviewed_readme(
        self, tmp_path: Path, body: bytes
    ) -> None:
        distribution = tmp_path / "dist"
        wheel, _ = _canonical_distribution(distribution)
        name = "operatebench-0.1.0.dist-info/METADATA"
        with zipfile.ZipFile(wheel) as archive:
            headers = archive.read(name).split(b"\n\n", 1)[0]
        _rewrite_wheel(wheel, replace={name: headers + b"\n\n" + body})
        _regenerate_record(wheel)
        problems = checks.check_built_distributions(
            REPO_ROOT, distribution_dir=distribution
        )
        assert any("invalid METADATA" in problem for problem in problems)

    def test_sdist_rejects_changed_production_source_bytes(self, tmp_path: Path) -> None:
        distribution = tmp_path / "dist"
        _, sdist = _canonical_distribution(distribution)
        name = "operatebench-0.1.0/src/operatebench/version.py"
        _rewrite_sdist_member(sdist, name, b'__version__ = "999.0"\n')
        problems = checks.check_built_distributions(
            REPO_ROOT, distribution_dir=distribution
        )
        assert any(
            "sdist production source differs" in problem and name in problem
            for problem in problems
        )

    @pytest.mark.parametrize("mutation", ["missing", "altered", "duplicate"])
    def test_wheel_rejects_noncanonical_console_entry_points(
        self, tmp_path: Path, mutation: str
    ) -> None:
        distribution = tmp_path / "dist"
        wheel, _ = _canonical_distribution(distribution)
        name = "operatebench-0.1.0.dist-info/entry_points.txt"
        if mutation == "missing":
            _rewrite_wheel(wheel, remove={name})
        elif mutation == "altered":
            _rewrite_wheel(
                wheel,
                replace={name: b"[console_scripts]\noperatebench = evil:main\n"},
            )
        else:
            with (
                pytest.warns(UserWarning, match="Duplicate name"),
                zipfile.ZipFile(wheel, "a") as archive,
            ):
                archive.writestr(name, b"[console_scripts]\noperatebench = evil:main\n")
        problems = checks.check_built_distributions(
            REPO_ROOT, distribution_dir=distribution
        )
        assert any("entry_points.txt" in problem for problem in problems)

    def test_wheel_console_script_names_are_case_sensitive(self, tmp_path: Path) -> None:
        distribution = tmp_path / "dist"
        wheel, _ = _canonical_distribution(distribution)
        name = "operatebench-0.1.0.dist-info/entry_points.txt"
        _rewrite_wheel(
            wheel,
            replace={
                name: (
                    b"[console_scripts]\n"
                    b"boundarybench = boundarybench.cli:main\n"
                    b"OperateBench = operatebench.cli:main\n"
                )
            },
        )
        _regenerate_record(wheel)
        problems = checks.check_built_distributions(
            REPO_ROOT, distribution_dir=distribution
        )
        assert any("console scripts differ" in problem for problem in problems)

    @pytest.mark.parametrize(
        "payload",
        [
            b"[console_scripts]\noperatebench = operatebench.cli:main\n"
            b"[console_scripts]\nboundarybench = boundarybench.cli:main\n",
            b"[console_scripts]\noperatebench = operatebench.cli:main\n"
            b"operatebench = evil:main\n",
            b"[DEFAULT]\noperatebench = operatebench.cli:main\n"
            b"[console_scripts]\nboundarybench = boundarybench.cli:main\n",
        ],
    )
    def test_wheel_rejects_duplicate_console_script_declarations(
        self, tmp_path: Path, payload: bytes
    ) -> None:
        distribution = tmp_path / "dist"
        wheel, _ = _canonical_distribution(distribution)
        name = "operatebench-0.1.0.dist-info/entry_points.txt"
        _rewrite_wheel(wheel, replace={name: payload})
        _regenerate_record(wheel)
        problems = checks.check_built_distributions(
            REPO_ROOT, distribution_dir=distribution
        )
        assert any("invalid entry_points.txt" in problem for problem in problems)

    @pytest.mark.parametrize("field", ["Name", "Requires-Dist"])
    def test_wheel_metadata_is_bound_to_reviewed_project(
        self, tmp_path: Path, field: str
    ) -> None:
        distribution = tmp_path / "dist"
        wheel, _ = _canonical_distribution(distribution)
        name = "operatebench-0.1.0.dist-info/METADATA"
        with zipfile.ZipFile(wheel) as archive:
            metadata = archive.read(name)
        if field == "Name":
            metadata = metadata.replace(b"Name: operatebench\n", b"Name: attacker\n")
        else:
            metadata += b"Requires-Dist: injected-release-dependency>=99\n"
        _rewrite_wheel(wheel, replace={name: metadata})
        _regenerate_record(wheel)
        problems = checks.check_built_distributions(
            REPO_ROOT, distribution_dir=distribution
        )
        assert any("METADATA" in problem for problem in problems)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("Provides-Extra", "attacker-controlled"),
            ("X-Attacker-Metadata", "present"),
            ("Dynamic", "Classifier"),
            ("Requires-External", "attacker-runtime"),
        ],
    )
    def test_wheel_rejects_headers_outside_the_reviewed_metadata_universe(
        self, tmp_path: Path, field: str, value: str
    ) -> None:
        distribution = tmp_path / "dist"
        wheel, _ = _canonical_distribution(distribution)
        name = "operatebench-0.1.0.dist-info/METADATA"
        with zipfile.ZipFile(wheel) as archive:
            headers, body = archive.read(name).split(b"\n\n", 1)
        mutated = headers + f"\n{field}: {value}\n\n".encode() + body
        _rewrite_wheel(wheel, replace={name: mutated})
        _regenerate_record(wheel)

        assert checks.check_built_distributions(
            REPO_ROOT, distribution_dir=distribution
        ) == [f"dist/{wheel.name}::{name}: invalid METADATA (ValueError)"]

    @pytest.mark.parametrize(
        "extra",
        [b"Name: operatebench\n", b"Version: 0.1.0\n", b" malformed continuation\n"],
    )
    def test_wheel_rejects_malformed_or_duplicate_metadata_identity(
        self, tmp_path: Path, extra: bytes
    ) -> None:
        distribution = tmp_path / "dist"
        wheel, _ = _canonical_distribution(distribution)
        name = "operatebench-0.1.0.dist-info/METADATA"
        with zipfile.ZipFile(wheel) as archive:
            metadata = archive.read(name) + extra
        _rewrite_wheel(wheel, replace={name: metadata})
        _regenerate_record(wheel)
        problems = checks.check_built_distributions(
            REPO_ROOT, distribution_dir=distribution
        )
        assert any("invalid METADATA" in problem for problem in problems)

    @pytest.mark.parametrize(
        "mutation",
        [
            "bad-width",
            "missing-member",
            "extra-member",
            "duplicate-member",
            "blank-integrity",
            "wrong-algorithm",
            "wrong-hash",
            "wrong-size",
            "unsafe-member",
            "missing-self",
            "populated-self",
        ],
    )
    def test_wheel_record_exactly_authenticates_every_member(
        self, tmp_path: Path, mutation: str
    ) -> None:
        distribution = tmp_path / "dist"
        wheel, _ = _canonical_distribution(distribution)
        record_name = "operatebench-0.1.0.dist-info/RECORD"
        with zipfile.ZipFile(wheel) as archive:
            rows = list(csv.reader(io.StringIO(archive.read(record_name).decode())))
        if mutation == "bad-width":
            rows[0].append("extra")
        elif mutation == "missing-member":
            rows.pop(0)
        elif mutation == "extra-member":
            rows.insert(0, ["ghost.py", "sha256=AAAA", "1"])
        elif mutation == "duplicate-member":
            rows.insert(0, rows[0].copy())
        elif mutation == "blank-integrity":
            rows[0][1:] = ["", ""]
        elif mutation == "wrong-algorithm":
            rows[0][1] = rows[0][1].replace("sha256=", "md5=")
        elif mutation == "wrong-hash":
            rows[0][1] = "sha256=" + "A" * 43
        elif mutation == "wrong-size":
            rows[0][2] = str(int(rows[0][2]) + 1)
        elif mutation == "unsafe-member":
            rows[0][0] = "../outside"
        elif mutation == "missing-self":
            rows = [row for row in rows if row[0] != record_name]
        else:
            rows[-1][1:] = ["sha256=" + "A" * 43, "1"]
        raw = io.StringIO(newline="")
        csv.writer(raw, lineterminator="\n").writerows(rows)
        _rewrite_wheel(wheel, replace={record_name: raw.getvalue().encode()})
        problems = checks.check_built_distributions(
            REPO_ROOT, distribution_dir=distribution
        )
        assert any("RECORD" in problem for problem in problems)

    def test_wheel_rejects_nonregular_dist_info_member(self, tmp_path: Path) -> None:
        distribution = tmp_path / "dist"
        wheel, _ = _canonical_distribution(distribution)
        name = "operatebench-0.1.0.dist-info/METADATA"
        replacement = wheel.with_name("symlink.whl")
        with (
            zipfile.ZipFile(wheel) as source,
            zipfile.ZipFile(replacement, "w") as target,
        ):
            for info in source.infolist():
                raw = source.read(info)
                if info.filename == name:
                    info = zipfile.ZipInfo(name)
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                target.writestr(info, raw)
        replacement.replace(wheel)
        problems = checks.check_built_distributions(
            REPO_ROOT, distribution_dir=distribution
        )
        assert any(name in problem and "not regular" in problem for problem in problems)

    def test_wheel_rejects_a_backslash_package_alias(self, tmp_path: Path) -> None:
        distribution = tmp_path / "dist"
        wheel, _ = _canonical_distribution(distribution)
        _rewrite_wheel(wheel, append=(r"operatebench\version.py", b"conflict\n"))
        problems = checks.check_built_distributions(
            REPO_ROOT, distribution_dir=distribution
        )
        assert any("unsafe wheel member name" in problem for problem in problems)

    def test_wheel_rejects_unicode_normalized_member_collisions(
        self, tmp_path: Path
    ) -> None:
        distribution = tmp_path / "dist"
        _empty_distribution(distribution)
        wheel = distribution / "pkg-0.1.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("pkg/caf\N{LATIN SMALL LETTER E WITH ACUTE}.txt", b"one")
            archive.writestr("pkg/cafe\N{COMBINING ACUTE ACCENT}.txt", b"two")
        problems = checks.check_built_distributions(tmp_path)
        assert any("duplicate normalized wheel member" in problem for problem in problems)

    @pytest.mark.parametrize("mutation", ["omitted", "duplicate", "symlink"])
    def test_wheel_rejects_prior_package_integrity_mutants(
        self, tmp_path: Path, mutation: str
    ) -> None:
        distribution = tmp_path / "dist"
        wheel, _ = _canonical_distribution(distribution)
        name = "boundarybench/__init__.py"
        if mutation == "omitted":
            _rewrite_wheel(wheel, remove={name})
        elif mutation == "duplicate":
            with (
                pytest.warns(UserWarning, match="Duplicate name"),
                zipfile.ZipFile(wheel, "a") as archive,
            ):
                archive.writestr(name, b"duplicate\n")
        else:
            replacement = wheel.with_name("symlink.whl")
            link = zipfile.ZipInfo(name)
            link.create_system = 3
            link.external_attr = 0o120777 << 16
            with (
                zipfile.ZipFile(wheel) as source,
                zipfile.ZipFile(replacement, "w") as target,
            ):
                for info in source.infolist():
                    if info.filename != name:
                        target.writestr(info, source.read(info))
                target.writestr(link, b"outside")
            replacement.replace(wheel)
        assert checks.check_built_distributions(REPO_ROOT, distribution_dir=distribution)

    @pytest.mark.parametrize("link_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
    def test_sdist_rejects_duplicate_production_source_links(
        self, tmp_path: Path, link_type: bytes
    ) -> None:
        distribution = tmp_path / "dist"
        _, sdist = _canonical_distribution(distribution)
        replacement = sdist.with_name("linked.tar.gz")
        duplicate_name = "operatebench-0.1.0/src/operatebench/version.py"
        with (
            tarfile.open(sdist, "r:gz") as source,
            tarfile.open(replacement, "w:gz") as target,
        ):
            for member in source.getmembers():
                payload = source.extractfile(member) if member.isfile() else None
                target.addfile(member, payload)
            link = tarfile.TarInfo(duplicate_name)
            link.type = link_type
            link.linkname = "../../outside"
            target.addfile(link)
        replacement.replace(sdist)
        problems = checks.check_built_distributions(
            REPO_ROOT, distribution_dir=distribution
        )
        assert any(duplicate_name in problem for problem in problems)


class TestProviderAllowlistIsEmptyInThePublicBuild:
    def test_the_public_adapter_pre_authorises_no_configuration(self) -> None:
        from boundarybench.providers.anthropic_messages import (
            AUDITED_CONFIGURATIONS,
        )

        assert frozenset() == AUDITED_CONFIGURATIONS

    def test_no_shipped_module_carries_a_configuration_identity_digest(self) -> None:
        # Default-deny is only default-deny if the build ships no identity it
        # could have been seeded with.
        digest = re.compile(r"\b[0-9a-f]{64}\b")
        for path in sorted((REPO_ROOT / "src" / "boundarybench").rglob("*.py")):
            assert not digest.search(path.read_text(encoding="utf-8")), path.name


class TestSyntheticFixture:
    def test_the_shipped_fixture_declares_itself_synthetic_and_non_production(
        self,
    ) -> None:
        assert checks.check_synthetic_fixture(REPO_ROOT) == []

    def test_the_fixture_the_readme_names_is_the_fixture_that_ships(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        assert checks.FIXTURE.as_posix() in readme
        assert (REPO_ROOT / checks.FIXTURE).is_file()


class TestDocumentation:
    def test_every_relative_link_and_anchor_resolves(self) -> None:
        assert checks.check_relative_links(REPO_ROOT) == []

    def test_the_readme_states_the_boundary_compatibility_version(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        assert checks.DISTRIBUTION_VERSION in readme
        assert checks.BOUNDARY_PACKAGE_VERSION in readme

    def test_the_readme_reports_the_dimensions_the_evaluator_actually_reports(
        self,
    ) -> None:
        from operatebench.domains.lettings.maintenance.evaluator import DIMENSIONS

        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        for dimension in DIMENSIONS:
            assert f"`{dimension}`" in readme, dimension

    def test_the_readme_reports_the_agents_that_actually_ship(self) -> None:
        from operatebench.domains.lettings.maintenance.agents import agent_ids

        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        shipped = agent_ids()
        negatives = [name for name in shipped if name != "reference"]
        assert f"{len(negatives)} targeted negative agents" in readme
        for name in shipped:
            assert f"`{name}`" in readme, name

    def test_the_methodology_result_vector_is_the_evaluators_own_list(self) -> None:
        # docs/METHODOLOGY.md prints a table of dimensions and claims the suite
        # holds it to the evaluator's tuple. This is that assertion, in order.
        import re

        from operatebench.domains.lettings.maintenance.evaluator import DIMENSIONS

        text = (REPO_ROOT / "docs" / "METHODOLOGY.md").read_text(encoding="utf-8")
        section = text.split("### What this build actually emits", 1)[1]
        tabled = re.findall(r"^\| `(\w+)` \|", section, re.MULTILINE)
        assert tuple(tabled) == DIMENSIONS

    def test_the_methodology_negative_agent_table_matches_the_registry(self) -> None:
        import re

        from operatebench.domains.lettings.maintenance.agents import NEGATIVE_AGENTS

        text = (REPO_ROOT / "docs" / "METHODOLOGY.md").read_text(encoding="utf-8")
        rows = dict(
            re.findall(r"^\| `(\w+)` \| [^|]+ \| ([^|]+) \|$", text, re.MULTILINE)
        )
        assert set(rows) == {entry.agent_id for entry in NEGATIVE_AGENTS}
        for entry in NEGATIVE_AGENTS:
            declared = {name.strip(" `") for name in rows[entry.agent_id].split(",")}
            assert declared == set(entry.targets), entry.agent_id

    def test_the_documentation_index_reaches_every_shipped_document(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        for document in sorted((REPO_ROOT / "docs").glob("*.md")):
            assert f"docs/{document.name}" in readme, document.name

    def test_the_readme_claims_no_package_index_publication(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8").lower()
        assert "pip install operatebench" not in readme
        assert "uv add operatebench" not in readme

    def test_no_drafting_note_survived_into_the_published_documents(self) -> None:
        for name in ("README.md", "docs/RELATED_WORK.md", "CITATION.cff"):
            text = (REPO_ROOT / name).read_text(encoding="utf-8")
            for marker in ("PLACEHOLDER", "Draft note", "DRAFT —", "TODO", "FIXME"):
                assert marker not in text, f"{name} still carries {marker!r}"


class TestGeneratedArtifacts:
    def test_no_cache_build_or_run_artefact_is_carried_as_content(self) -> None:
        assert checks.check_no_generated_artifacts(REPO_ROOT) == []

    def test_building_the_wheel_does_not_fail_the_gate(self) -> None:
        # dist/ and the tool caches are created by the documented commands. The
        # gate asks whether they can be committed, not whether they exist — a
        # check that failed after `uv build` is one people learn to skip.
        for name in ("dist", ".ruff_cache", ".coverage"):
            assert name in checks.GENERATED_NAMES or name.rstrip("/") in {
                entry.rstrip("/") for entry in checks.REQUIRED_IGNORE_ENTRIES
            }
        assert checks.check_no_generated_artifacts(REPO_ROOT) == []

    def test_an_unignored_generated_artefact_is_caught(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("\n".join(checks.REQUIRED_IGNORE_ENTRIES))
        (tmp_path / "dist").mkdir()
        assert checks.check_no_generated_artifacts(tmp_path) == []

        # Now remove the entry that covers it and the same tree must fail.
        kept = [e for e in checks.REQUIRED_IGNORE_ENTRIES if e != "dist/"]
        (tmp_path / ".gitignore").write_text("\n".join(kept))
        problems = checks.check_no_generated_artifacts(tmp_path)
        assert any("dist/" in problem for problem in problems)
        assert any("can be committed as content" in problem for problem in problems)


class TestScannerScope:
    def test_the_scanner_never_descends_into_history_or_environments(self) -> None:
        for name in (".git", ".venv", "__pycache__", ".mypy_cache", "dist"):
            assert name in checks.SKIPPED_DIRECTORIES

    def test_every_required_file_is_in_the_source_scan(self) -> None:
        candidates = {
            path.relative_to(REPO_ROOT).as_posix()
            for path in checks.candidate_files(REPO_ROOT)
        }
        required = {
            name for name, _ in checks.REQUIRED_FILES + checks.SOURCE_CONTROL_FILES
        }
        assert required <= candidates
        assert ".gitignore" in candidates

    def test_candidate_files_include_arbitrary_regular_files_but_not_links_or_skips(
        self, tmp_path: Path
    ) -> None:
        included = {
            tmp_path / "uv.lock",
            tmp_path / ".gitignore",
            tmp_path / "future.format-never-seen-before",
            tmp_path / "opaque-binary",
        }
        for path in included:
            path.write_bytes(b"candidate")
        skipped = tmp_path / ".git"
        skipped.mkdir()
        (skipped / "hidden.txt").write_text("hidden", encoding="utf-8")
        (tmp_path / "linked.txt").symlink_to(tmp_path / "uv.lock")

        assert set(checks.candidate_files(tmp_path)) == included

    @pytest.mark.parametrize("name", ["uv.lock", ".gitignore", "data.future-text"])
    def test_source_scanner_catches_markers_in_any_text_format(
        self, tmp_path: Path, name: str
    ) -> None:
        planted = tmp_path / name
        planted.write_text(
            "token: sk-ant-" + "a" * 40 + "\nsee /" + "var/www/private/run.log\n",
            encoding="utf-8",
        )

        credential_problems = checks.check_no_credential_shapes(tmp_path)
        private_path_problems = checks.check_no_private_references(tmp_path)

        assert len(credential_problems) == 1
        assert f"{name}:1" in credential_problems[0]
        assert len(private_path_problems) == 1
        assert f"{name}:2" in private_path_problems[0]

    def test_a_file_with_a_null_byte_is_not_decoded(self, tmp_path: Path) -> None:
        binary = tmp_path / "compiled.py"
        binary.write_bytes(b"\x00\x01sk-ant-" + b"a" * 40)
        assert checks._read(binary) is None

    def test_a_file_with_invalid_utf8_is_not_decoded(self, tmp_path: Path) -> None:
        binary = tmp_path / "invalid.data"
        binary.write_bytes(b"\xffsk-ant-" + b"a" * 40)
        assert checks._read(binary) is None

    def test_the_scanner_would_catch_a_planted_credential(self, tmp_path: Path) -> None:
        # The gate is only worth running if it fails when it should.
        planted = tmp_path / "leaked.md"
        planted.write_text("token: sk-ant-" + "a" * 40 + "\n", encoding="utf-8")
        problems = checks.check_no_credential_shapes(tmp_path)
        assert len(problems) == 1
        assert "leaked.md:1" in problems[0]

    def test_the_scanner_would_catch_a_planted_private_path(self, tmp_path: Path) -> None:
        planted = tmp_path / "note.md"
        # Assembled at run time: a literal here would make this file the very
        # thing the scanner is meant to find.
        planted.write_text("see /" + "var/www/private-bench/run.log\n", encoding="utf-8")
        problems = checks.check_no_private_references(tmp_path)
        assert len(problems) == 1
        assert "absolute local path" in problems[0]

    @pytest.mark.parametrize(
        ("name", "body"),
        [
            ("PUBLICATION_MANIFEST.json", '"interface": "{}"\n'),
            ("METHODOLOGY.md", "predecessor {}\n"),
            ("test_claim.py", '"""baseline {}"""\n'),
        ],
    )
    @pytest.mark.parametrize(
        "token",
        [
            pytest.param("ab" * 20, id="lowercase"),
            pytest.param(("ab" * 20).upper(), id="uppercase"),
            pytest.param("aB" * 20, id="mixed-case"),
        ],
    )
    def test_a_bare_commit_identifier_in_public_source_is_caught(
        self, tmp_path: Path, name: str, body: str, token: str
    ) -> None:
        (tmp_path / name).write_text(body.format(token), encoding="utf-8")

        problems = checks.check_no_bare_commit_identifiers(tmp_path)

        assert len(problems) == 1
        assert f"{name}:1" in problems[0]
        assert "bare 40-hex commit identifier" in problems[0]

    @pytest.mark.parametrize("suffix", ["yml", "yaml"])
    def test_a_sha_pinned_action_uses_line_is_the_only_exception(
        self, tmp_path: Path, suffix: str
    ) -> None:
        workflow = tmp_path / ".github" / "workflows" / f"ci.{suffix}"
        workflow.parent.mkdir(parents=True)
        token = "ab" * 20
        workflow.write_text(
            f"steps:\n  - uses: actions/checkout@{token} # v4\n", encoding="utf-8"
        )

        assert checks.check_no_bare_commit_identifiers(tmp_path) == []

    @pytest.mark.parametrize(
        "target",
        [
            pytest.param("actions/checkout", id="checkout"),
            pytest.param("actions/upload-artifact", id="upload-artifact"),
            pytest.param("astral-sh/setup-uv", id="setup-uv"),
            pytest.param("actions/cache/save", id="action-subpath"),
        ],
    )
    def test_current_approved_action_targets_are_accepted(self, target: str) -> None:
        token = "ab" * 20
        text = f"steps:\n  - uses: {target}@{token} # v4\n"

        assert (
            checks.scan_bare_commit_identifiers(
                text,
                where=".github/workflows/ci.yml",
                workflow_path=".github/workflows/ci.yml",
            )
            == []
        )

    @pytest.mark.parametrize(
        "target",
        [
            pytest.param("UnapprovedOwner/action", id="unapproved-owner"),
            pytest.param("Actions/checkout", id="owner-case-mismatch"),
            pytest.param("owner/repo/../path", id="dotdot-segment"),
            pytest.param("owner/repo//path", id="empty-segment"),
            pytest.param("owner/repo/./path", id="dot-segment"),
            pytest.param("owner/repo/", id="trailing-slash"),
            pytest.param("/actions/checkout", id="leading-slash"),
            pytest.param("actions\\checkout", id="backslash"),
            pytest.param("actions", id="missing-repository"),
            pytest.param("https://github.com/actions/checkout", id="github-url-form"),
        ],
    )
    def test_action_pin_exception_rejects_noncanonical_or_unapproved_targets(
        self, target: str
    ) -> None:
        token = "ab" * 20
        text = f"steps:\n  - uses: {target}@{token} # v4\n"

        problems = checks.scan_bare_commit_identifiers(
            text,
            where=".github/workflows/ci.yml",
            workflow_path=".github/workflows/ci.yml",
        )

        assert len(problems) == 1
        assert "bare 40-hex commit identifier" in problems[0]

    @pytest.mark.parametrize(
        ("workflow_path", "allowed"),
        [
            pytest.param(".github/workflows/ci.yml", True, id="canonical-yml"),
            pytest.param(".github/workflows/ci.yaml", True, id="canonical-yaml"),
            pytest.param(".github//workflows/ci.yml", False, id="repeated-separator"),
            pytest.param("./.github/workflows/ci.yml", False, id="leading-dot"),
            pytest.param(".github/workflows/./ci.yml", False, id="dot-component"),
            pytest.param(".github/workflows/../ci.yml", False, id="dotdot-component"),
            pytest.param("/.github/workflows/ci.yml", False, id="leading-slash"),
            pytest.param(".github/workflows/ci.yml/", False, id="trailing-slash"),
            pytest.param(".github\\workflows\\ci.yml", False, id="backslash-separator"),
            pytest.param("pkg/.github/workflows/ci.yml", False, id="nested-package"),
            pytest.param(".github/workflows/ci.YML", False, id="uppercase-suffix"),
            pytest.param(".github/workflows/ci.Yml", False, id="mixed-case-suffix"),
        ],
    )
    def test_action_pin_exception_requires_a_raw_canonical_workflow_path(
        self, workflow_path: str, allowed: bool
    ) -> None:
        token = "ab" * 20
        text = f"steps:\n  - uses: actions/checkout@{token} # v4\n"

        problems = checks.scan_bare_commit_identifiers(
            text, where=workflow_path, workflow_path=workflow_path
        )

        assert (problems == []) is allowed
        if not allowed:
            assert "bare 40-hex commit identifier" in problems[0]

    @pytest.mark.parametrize(
        "token",
        [
            pytest.param(("ab" * 20).upper(), id="uppercase"),
            pytest.param("aB" * 20, id="mixed-case"),
        ],
    )
    def test_a_noncanonical_sha_pinned_action_is_rejected(
        self, tmp_path: Path, token: str
    ) -> None:
        workflow = tmp_path / ".github" / "workflows" / "ci.yml"
        workflow.parent.mkdir(parents=True)
        workflow.write_text(
            f"steps:\n  - uses: actions/checkout@{token} # v4\n", encoding="utf-8"
        )

        problems = checks.check_no_bare_commit_identifiers(tmp_path)

        assert len(problems) == 1
        assert "ci.yml:2" in problems[0]

    @pytest.mark.parametrize(
        "line",
        [
            "# predecessor {}",
            "env: {}",
            "run: echo {}",
            "uses: owner/action@tag # {}",
        ],
    )
    @pytest.mark.parametrize(
        "token",
        [
            pytest.param("ab" * 20, id="lowercase"),
            pytest.param(("ab" * 20).upper(), id="uppercase"),
            pytest.param("aB" * 20, id="mixed-case"),
        ],
    )
    def test_a_workflow_does_not_allow_commit_tokens_outside_action_pins(
        self, tmp_path: Path, line: str, token: str
    ) -> None:
        workflow = tmp_path / ".github" / "workflows" / "ci.yml"
        workflow.parent.mkdir(parents=True)
        workflow.write_text(line.format(token) + "\n", encoding="utf-8")

        problems = checks.check_no_bare_commit_identifiers(tmp_path)

        assert len(problems) == 1
        assert "ci.yml:1" in problems[0]

    def test_the_scanner_would_catch_a_non_public_document_reference(
        self, tmp_path: Path
    ) -> None:
        planted = tmp_path / "note.md"
        name = "-".join(["private", "evidence"]) + ".md"
        planted.write_text(f"see {name}\n", encoding="utf-8")
        problems = checks.check_no_internal_documents(tmp_path)
        assert len(problems) == 1
        assert "non-public document reference" in problems[0]

    def test_the_scanner_would_catch_a_broken_relative_link(self, tmp_path: Path) -> None:
        planted = tmp_path / "note.md"
        planted.write_text("[gone](does-not-exist.md)\n", encoding="utf-8")
        problems = checks.check_relative_links(tmp_path)
        assert len(problems) == 1
        assert "does not resolve" in problems[0]

    def test_the_scanner_would_catch_a_broken_anchor(self, tmp_path: Path) -> None:
        (tmp_path / "target.md").write_text("# Real heading\n", encoding="utf-8")
        (tmp_path / "note.md").write_text("[x](target.md#missing)\n", encoding="utf-8")
        problems = checks.check_relative_links(tmp_path)
        assert len(problems) == 1
        assert "anchor" in problems[0]


class TestTheGateItself:
    def test_every_check_passes_on_the_candidate_tree(self) -> None:
        results = checks.run_all(REPO_ROOT)
        failed = {name: problems for name, problems in results.items() if problems}
        assert failed == {}

    def test_the_command_reports_success(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert checks.main([str(REPO_ROOT)]) == 0
        assert f"all {len(checks.ALL_CHECKS)} passed" in capsys.readouterr().out

    def test_the_command_reports_failure_when_a_check_fails(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # An empty directory fails almost everything; the point is the exit code.
        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname='x'\n", encoding="utf-8"
        )
        (tmp_path / ".gitignore").write_text("", encoding="utf-8")
        assert checks.main([str(tmp_path)]) == 1
        assert "FAILED" in capsys.readouterr().out
