"""Deterministic public-release checks for the OperateBench candidate tree.

Run as a command::

    uv run python -m tools.check_public_release

or assert as tests (``tests/test_public_release.py``). Both call the same
functions, so the script and the suite cannot drift apart.

Every check answers one question about the candidate and returns any problems as
strings. No check phones anywhere, imports a provider, or consults a service.

The working-tree checks skip ``.git`` (history is a separate gate), virtual
environments, dependency directories, generated caches and ``dist``. They attempt
every other regular non-symlink file and classify text by content: a NUL byte or
invalid UTF-8 makes data binary. The built-distribution check applies the same
classification to every regular member in wheel and sdist archives already
present under ``dist``; an absent ``dist`` directory is not an error.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal, cast

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Directories the scanner never descends into. ``.git`` is excluded on purpose:
#: history is checked by the export gate, not by a working-tree text scan.
SKIPPED_DIRECTORIES: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        ".tox",
        ".nox",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".coverage_cache",
        "htmlcov",
        "dist",
        "build",
        "mutants",
        ".idea",
        ".vscode",
        ".claude",
    }
)

DISTRIBUTION_NAME = "operatebench"
DISTRIBUTION_VERSION = "0.1.0"
#: The runtime-and-evaluator identity recorded as ``engine_version`` in every
#: artefact. Deliberately not derived from the distribution version: the two
#: answer different questions and move on different schedules (docs/VERSIONING.md).
ENGINE_VERSION = "0.12.0"
#: The artefact contract this build writes, recorded as ``artifact_version`` in
#: every artefact. A fourth identity, pinned the same way and for the same
#: reason: contract 3 requires every accepted effect to record the canonical
#: identities it established, and a record shape that changed while the number
#: did not is a document a reader has to guess at (docs/VERSIONING.md).
ARTIFACT_VERSION = 8
BOUNDARY_PACKAGE_VERSION = "0.0.15"

FIXTURE = Path("examples/operatebench/maintenance_v0_1.yaml")

#: Files a public release is not a public release without.
REQUIRED_FILES: tuple[tuple[str, str], ...] = (
    ("LICENSE", "Apache License"),
    ("LICENSE-DATA", "CC BY 4.0"),
    ("NOTICE", "PROSPIRE TECHNOLOGIES LTD"),
    ("README.md", "OperateBench"),
    ("CITATION.cff", "cff-version"),
    ("CONTRIBUTING.md", "Contributing"),
    ("CODE_OF_CONDUCT.md", "Code of Conduct"),
    ("SECURITY.md", "Security"),
    ("REUSE.toml", "SPDX-License-Identifier"),
    ("LICENSES/Apache-2.0.txt", "Apache License"),
    ("LICENSES/CC-BY-4.0.txt", "Creative Commons Attribution 4.0"),
    ("LICENSES/LicenseRef-DCO-1.1.txt", "Developer Certificate of Origin"),
    ("DCO", "Developer Certificate of Origin"),
    ("uv.lock", "version = 1"),
    ("docs/METHODOLOGY.md", "Methodology"),
    ("docs/RELATED_WORK.md", "Related work"),
    ("docs/VERSIONING.md", "Versioning"),
    ("docs/CONTRIBUTING_OPERATIONS.md", "Contributing an operation"),
    ("docs/ARCHITECTURE_RFC.md", "Architecture RFC"),
    ("docs/HISTORY.md", "Historical Continuity"),
    (".github/workflows/ci.yml", "uv sync"),
)

SOURCE_CONTROL_FILES: tuple[tuple[str, str], ...] = (
    ("PUBLIC_RELEASE_CHECKLIST.md", "Public release checklist"),
    ("PUBLIC_CANDIDATE_VERIFICATION.md", "Public candidate verification"),
    ("PUBLICATION_MANIFEST.json", "manifest_schema_version"),
)

Surface = Literal["source", "sdist"]
VALID_SURFACES: frozenset[str] = frozenset({"source", "sdist"})


def _validated_surface(surface: str) -> Surface:
    if surface not in VALID_SURFACES:
        raise ValueError(f"invalid scanner surface: {surface!r}")
    return cast(Surface, surface)


#: Shapes that mean "somebody pasted a real credential". Deliberately narrow:
#: a pattern that matches every long hex string flags every content digest in
#: the tree and teaches the reader to ignore the gate.
CREDENTIAL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Anthropic-style API key", r"sk-ant-[A-Za-z0-9_-]{20,}"),
    ("OpenAI-style API key", r"sk-(?!ant-)[A-Za-z0-9]{32,}"),
    # Project and service-account families: complete prefix plus 32..512 ASCII
    # letters/digits/underscores/hyphens, bounded on both sides. This is a
    # conservative detection envelope, not provider validation or a promise to
    # recognize every future credential format. Keep the legacy rule independent.
    (
        "OpenAI project/service-account API key",
        r"(?<![A-Za-z0-9_-])sk-(?:proj|svcacct)-[A-Za-z0-9_-]{32,512}"
        r"(?![A-Za-z0-9_-])",
    ),
    ("AWS access key id", r"AKIA[0-9A-Z]{16}"),
    ("GitHub token", r"gh[pousr]_[A-Za-z0-9]{30,}"),
    ("Slack token", r"xox[abprs]-[A-Za-z0-9-]{16,}"),
    ("Google API key", r"AIza[0-9A-Za-z_-]{35}"),
    ("PEM private key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("Bearer credential literal", r"[Bb]earer\s+[A-Za-z0-9_\-]{24,}\."),
)

#: Historical synthetic key literals from the Boundary Track's provider-trust
#: tests. They exist to prove a credential is *refused*, *redacted* or *never
#: written*, so they are the one thing a credential scan must not remove. Each
#: entry is an exact literal, not a prefix: an allowlist that accepts a prefix
#: would accept a real key that happens to start the same way.
ALLOWED_SYNTHETIC_KEY_LITERALS: frozenset[str] = frozenset(
    {
        "sk-ant-not-a-real-key",
        "sk-ant-not-a-real-key-0123456789",
        "sk-ant-should-never-appear",
        "sk-ant-api03-not-a-real-key",
    }
)

#: Absolute local paths and private references. The published tree must name no
#: filesystem outside itself and no repository except its own public identity.
PRIVATE_REFERENCE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("absolute local path", r"(?<![\w.-])/(?:var/www|home|Users|root|srv|opt/dwelly)/"),
    ("internal hostname", r"\b[\w.-]+\.(?:internal|local|corp|lan|intranet)\b"),
    ("private network endpoint", r"\bhttps?://(?:10|127|192\.168)\.\d{1,3}\."),
)

#: A bare Git commit identity is internal provenance, not a content digest. Match
#: every hexadecimal case representation. The sole publication-safe form is a
#: canonical lowercase supply-chain pin in a GitHub Actions workflow.
BARE_COMMIT_IDENTIFIER = re.compile(r"(?<![0-9A-Za-z_-])[0-9A-Fa-f]{40}(?![0-9A-Za-z_-])")
PINNED_ACTION_USES_LINE = re.compile(
    r"^\s*(?:-\s+)?uses:\s+"
    r"(?P<target>\S+)@(?P<sha>[0-9a-f]{40})"
    r"(?:\s+#\s+v[0-9]+(?:\.[0-9]+)*)?\s*$"
)
ACTION_TARGET_SEGMENT = re.compile(r"[A-Za-z0-9_.-]+")

#: GitHub owners trusted to execute pinned code in the current workflows. This
#: is intentionally independent from the public-identity/citation allowlist.
ALLOWED_GITHUB_ACTION_OWNERS: frozenset[str] = frozenset({"actions", "astral-sh"})

#: The only GitHub owners this tree is allowed to name: its own public identity,
#: plus the owners of the works cited in docs/RELATED_WORK.md.
ALLOWED_GITHUB_OWNERS: frozenset[str] = frozenset(
    {
        "DwellyOrg",
        "rdi-berkeley",
        "sierra-research",
        "apple",
        "StonyBrookNLP",
        "ServiceNow",
        "SalesforceAIResearch",
        "gitleaks",
        "astral-sh",
        "actions",
    }
)

#: Explicitly non-public document references do not belong in a public tree.
#: The pattern uses visibility words, not names of any particular work product.
INTERNAL_DOCUMENT_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "non-public document reference",
        r"(?i)\b[\w./-]*(?:private|internal|non[-_]?public)[-_][\w./-]*\.md\b",
    ),
)

#: Generic provenance policy. Public source may describe behavior and methodology,
#: but must not assert non-public validation or point readers to inaccessible
#: execution evidence. Rules are intentionally phrased as broad policy classes,
#: without encoding project-specific milestones or work-product names.
PRIVATE_PROVENANCE_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "non-public audit claim",
        r"(?i)\b(?:an?\s+)?(?:private|internal|non-public)\s+"
        r"(?:\w+\s+){0,2}audit(?:ed|s)?\b|"
        r"\b(?:run|execution|result|evidence)\s+"
        r"(?:was|were|has been|had been)\s+audited\b",
    ),
    (
        "claimed live/real provider execution",
        r"(?i)\b(?:evidence|results?|findings?|measurements?)\s+"
        r"(?:came|come|derive[ds]?|(?:was|were)\s+obtained)\s+from\s+"
        r"(?:a|an|one|the|this|that|our|their)\s+(?:real|live)\s+"
        r"provider\s+runs?\b|"
        r"\b(?:a|an|one|the|this|that|our|their)\s+(?:real|live)\s+"
        r"provider\s+runs?\b",
    ),
    (
        "non-public production-provider validation claim",
        r"(?i)\b(?:confidential|restricted|proprietary)\s+"
        r"(?:\w+\s+){0,2}(?:assessment|evaluation|validation)\b"
        r"[^.!?\n]{0,240}\b(?:production[- ]provider|provider)\s*[- ]?"
        r"(?:executions?|calls?|runs?)\b",
    ),
    (
        "external/private path",
        r"(?i)\b(?:an?\s+|the\s+)?(?:external|private|non-public)\s+"
        r"(?:filesystem\s+)?(?:path|directory|folder|location)\b|"
        r"\b(?:path|directory|folder|location)\s+outside\s+(?:this|the)\s+"
        r"(?:repository|repo|tree|project)\b",
    ),
    (
        "non-public run configuration/evidence reference",
        r"(?i)\b(?:the\s+|an?\s+)?(?:private|internal|non-public|external)\s+"
        r"(?:run\s+)?(?:configuration|evidence)(?:\s+(?:reference|record|"
        r"identifier|identity|id|path|artifact|artefact|report|log))?\b",
    ),
)

#: Prose about the *shipped thing itself* that a public reader cannot check and
#: that was never true of what they downloaded. Two classes, and both shipped
#: once inside the installed package's own docstring.
#:
#: **Internal stage labels.** A number or letter naming a step in a private plan
#: — "the P1 executable slice", "curating it is Pass B" — tells a reader nothing
#: they can act on and leaks the shape of work they are not part of. The
#: patterns are the *label forms*, not the words: ``Stage``, ``phase`` and
#: ``pass`` all have legitimate behavioural meanings in this tree (a staged
#: execution, a test that passes), so a rule that matched those would be a rule
#: people learn to work around.
#:
#: **False distribution topology.** This tree builds one distribution containing
#: two importable packages. Prose saying it builds two — "the two distributions
#: are built from the same wheel" — is mechanically false about the artefact a
#: user installs, and a reader who believes it will look for a second package
#: index entry that does not exist.
PUBLIC_SCOPE_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "internal stage label",
        r"\bP\d+\s+(?:executable\s+)?"
        r"(?:slice|pass|phase|milestone|tranche|increment|workstream|deliverable)\b"
        r"|\bPass\s+[A-Z]\b"
        r"|(?i:\b(?:for|in|during)\s+this\s+pass\b)",
    ),
    (
        "false distribution topology",
        r"(?i:\b(?:the\s+)?two\s+distributions\b)"
        r"|(?i:\bboth\s+distributions\b)"
        r"|(?i:\bdistributions\s+are\s+built\s+from\s+the\s+same\b)",
    ),
)

#: The files that state the release policy itself. They enumerate the classes of
#: private content by name — that is what they are *for* — so the provenance scan
#: exempts them the same way and for the same reason it exempts this module. The
#: exemption is by exact path, never by directory: a doc that merely discusses
#: the topic gets no relief.
PROVENANCE_POLICY_DOCUMENTS: frozenset[str] = frozenset(
    {
        "tools/check_public_release.py",
        "PUBLICATION_MANIFEST.json",
        "PUBLIC_CANDIDATE_VERIFICATION.md",
    }
)

#: Generated artefacts. The question is never "is this on disk" — running the
#: tools *creates* dist/, .ruff_cache/ and .coverage, and a gate that fails
#: because you built the wheel is a gate people learn to skip. The question is
#: whether anything generated could be *tracked*, so each class must be named in
#: .gitignore and every generated path on disk must be matched by one of them.
REQUIRED_IGNORE_ENTRIES: tuple[str, ...] = (
    ".venv/",
    "__pycache__/",
    "*.pyc",
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    "*.egg-info/",
    "dist/",
    "build/",
    ".coverage",
    "htmlcov/",
    "mutants/",
)

#: Names that are generated output, never candidate content. Each must be
#: covered by .gitignore wherever it appears in the tree.
GENERATED_NAMES: tuple[str, ...] = (
    "dist",
    "build",
    "htmlcov",
    "mutants",
    ".coverage",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".venv",
    "__pycache__",
)

#: Suffixes that are generated output wherever they appear.
GENERATED_SUFFIXES: tuple[str, ...] = (".pyc", ".pyo", ".egg-info")


# -- tree walking -------------------------------------------------------------


def candidate_files(root: Path = REPO_ROOT) -> list[Path]:
    """Every regular file that may be part of the publication candidate.

    Symlinks and device nodes are skipped rather than followed: a scanner that
    opens whatever a name points at is reading something other than the tree it
    was asked about. Callers use :func:`_read` to distinguish UTF-8 text from
    binary content, never a filename suffix or stem allowlist.
    """

    found: list[Path] = []

    def walk(directory: Path) -> Iterator[Path]:
        for entry in sorted(directory.iterdir()):
            if entry.name in SKIPPED_DIRECTORIES:
                continue
            if entry.is_symlink():
                continue
            if entry.is_dir():
                yield from walk(entry)
            elif entry.is_file():
                yield entry

    found.extend(walk(root))
    return found


def _decode_text(data: bytes) -> str | None:
    """UTF-8 text, or ``None`` for NUL-bearing or invalid UTF-8 binary data."""
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _read(path: Path) -> str | None:
    """Text of a candidate file, or ``None`` if it is unreadable or binary."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return _decode_text(data)


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


# -- checks -------------------------------------------------------------------


def check_distribution_metadata(root: Path = REPO_ROOT) -> list[str]:
    """The distribution names itself, versions itself and exposes both CLIs."""
    problems: list[str] = []
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return ["pyproject.toml is missing; there is no distribution to check"]
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    tool = data.get("tool", {})

    build_system = data.get("build-system", {})
    expected_build_requires = ["hatchling==1.32.0"]
    if build_system.get("requires") != expected_build_requires:
        problems.append(
            f"build-system.requires is {build_system.get('requires')!r}, expected "
            f"exactly {expected_build_requires!r}"
        )

    if project.get("name") != DISTRIBUTION_NAME:
        problems.append(
            f"distribution name is {project.get('name')!r}, expected "
            f"{DISTRIBUTION_NAME!r}"
        )
    if "version" not in project.get("dynamic", []):
        problems.append(
            "the distribution version is written out in pyproject.toml; it must be "
            "dynamic so it cannot drift from the module that states it"
        )
    if "version" in project:
        problems.append("a static project.version duplicates src/operatebench/version.py")

    version_path = tool.get("hatch", {}).get("version", {}).get("path")
    if version_path != "src/operatebench/version.py":
        problems.append(
            f"the build reads the version from {version_path!r}; the distribution "
            "version must come from src/operatebench/version.py"
        )

    scripts = project.get("scripts", {})
    for name, target in (
        ("operatebench", "operatebench.cli:main"),
        ("boundarybench", "boundarybench.cli:main"),
    ):
        if scripts.get(name) != target:
            problems.append(f"console script {name!r} is {scripts.get(name)!r}")

    packages = (
        tool.get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
        .get("packages", [])
    )
    if set(packages) != {"src/operatebench", "src/boundarybench"}:
        problems.append(f"the wheel carries {sorted(packages)}, expected both packages")

    if project.get("license") != "Apache-2.0":
        problems.append(
            f"project.license is {project.get('license')!r}; the code licence is the "
            "SPDX expression Apache-2.0"
        )
    for required in ("LICENSE", "NOTICE"):
        if required not in project.get("license-files", []):
            problems.append(f"{required} is not listed in project.license-files")

    description = project.get("description", "")
    if "OperateBench" not in description:
        problems.append(
            f"project.description does not describe OperateBench: {description!r}"
        )

    if "ruff" not in " ".join(data.get("dependency-groups", {}).get("dev", [])):
        problems.append("ruff is not in the dev dependency group, but CI lints with it")

    classifiers = project.get("classifiers", [])
    if not any(
        "Apache Software License" in c or c.startswith("License ::") for c in classifiers
    ):
        # PEP 639 retires the licence classifier; either form is acceptable, but
        # the Python versions the matrix claims must be declared.
        pass
    for version in ("3.11", "3.14"):
        if f"Programming Language :: Python :: {version}" not in classifiers:
            problems.append(f"no classifier declares support for Python {version}")

    if not project.get("urls"):
        problems.append(
            "project.urls is empty; a published distribution states where it lives"
        )
    if not project.get("authors"):
        problems.append("project.authors is empty")
    return problems


def check_package_versions(
    root: Path = REPO_ROOT, *, surface: Surface = "source"
) -> list[str]:
    """Every version identity states itself, and none of them is guessed.

    Distribution, engine and artefact contract are pinned separately and by
    value. Requiring the first two to be *equal* was the defect this replaces:
    it made an engine change look like a packaging change, so a runtime and
    evaluator that had both moved still published themselves as 0.1.0. The
    artefact contract is here for the same reason one identity over — the record
    grew three fields, and a contract number that did not move would leave two
    different document shapes both calling themselves version 1.

    The manifest cross-check is surface-aware. On the source surface the
    published record must be readable and must carry a ``release`` mapping: a
    manifest that is present but unparseable, not an object, or carrying no
    release block cannot be allowed to *skip* the comparison, because skipping
    it silently is indistinguishable from passing it.
    """
    surface = _validated_surface(surface)
    problems: list[str] = []

    operate = _module_assignments(root / "src" / "operatebench" / "version.py")
    if operate.get("__version__") != DISTRIBUTION_VERSION:
        problems.append(
            f"operatebench.version.__version__ is {operate.get('__version__')!r}, "
            f"expected the distribution version {DISTRIBUTION_VERSION!r}"
        )
    if operate.get("OPERATEBENCH_VERSION") != ENGINE_VERSION:
        problems.append(
            "operatebench.version.OPERATEBENCH_VERSION is "
            f"{operate.get('OPERATEBENCH_VERSION')!r}, expected the engine version "
            f"{ENGINE_VERSION!r}"
        )

    contract = _module_int_assignments(root / "src" / "operatebench" / "artifact.py")
    if "ARTIFACT_VERSION" not in contract:
        problems.append(
            "operatebench.artifact.ARTIFACT_VERSION is not an integer literal this "
            "scanner can read; the artefact contract version is an identity a "
            "candidate publishes, so it has to be pinnable without importing the "
            "package"
        )
    elif contract["ARTIFACT_VERSION"] != ARTIFACT_VERSION:
        problems.append(
            f"operatebench.artifact.ARTIFACT_VERSION is "
            f"{contract['ARTIFACT_VERSION']!r}, expected the artefact contract "
            f"version {ARTIFACT_VERSION!r}"
        )

    release, unreadable = _publication_manifest_release(root, surface=surface)
    problems.extend(unreadable)
    if release is not None:
        recorded = release.get("engine_version")
        if recorded != ENGINE_VERSION:
            problems.append(
                f"PUBLICATION_MANIFEST.json records engine_version {recorded!r}; the "
                f"engine this candidate ships is {ENGINE_VERSION!r}"
            )
        published = release.get("artifact_version")
        # A boolean is an int in Python and is not a contract version anywhere.
        if not isinstance(published, int) or isinstance(published, bool):
            problems.append(
                f"PUBLICATION_MANIFEST.json records a "
                f"{type(published).__name__} for release.artifact_version, not the "
                "integer the artefact contract is numbered with"
            )
        elif published != ARTIFACT_VERSION:
            problems.append(
                f"PUBLICATION_MANIFEST.json records artifact_version {published!r}; "
                f"the artefact contract this candidate writes is {ARTIFACT_VERSION!r}"
            )
        distribution = release.get("distribution")
        if not isinstance(distribution, dict):
            problems.append(
                f"PUBLICATION_MANIFEST.json records a {type(distribution).__name__} "
                "for release.distribution, not the mapping that carries the "
                "distribution version"
            )
        elif distribution.get("version") != DISTRIBUTION_VERSION:
            problems.append(
                f"PUBLICATION_MANIFEST.json records distribution version "
                f"{distribution.get('version')!r}; this candidate ships "
                f"{DISTRIBUTION_VERSION!r}"
            )

    boundary = _module_assignments(root / "src" / "boundarybench" / "version.py")
    if boundary.get("__version__") != BOUNDARY_PACKAGE_VERSION:
        problems.append(
            f"boundarybench.__version__ is {boundary.get('__version__')!r}; the "
            f"Boundary Track keeps its historical {BOUNDARY_PACKAGE_VERSION!r}"
        )
    return problems


def _publication_manifest_release(
    root: Path, *, surface: Surface = "source"
) -> tuple[dict[str, Any] | None, list[str]]:
    """The manifest's ``release`` block, and why it could not be read.

    There is exactly one reason a candidate may answer nothing here: the sdist
    does not ship ``PUBLICATION_MANIFEST.json``, so a scanner run from inside an
    extracted sdist has no published record to compare the literals against.
    Absence *there* is the design, and it stays silent — reporting it would turn
    one deliberately omitted file into two identity defects.

    Every other way of reading nothing is a problem the source surface names.
    Returning a bare ``None`` for a manifest that is present but unparseable,
    not an object, or carrying no ``release`` mapping made "the cross-check
    passed" and "the cross-check never ran" the same observable outcome: a
    single stray character in the file silently retired the engine/distribution
    comparison while the gate still printed ``ok``. The required-files check is
    not a substitute, because it only asks whether the text contains
    ``manifest_schema_version`` — which invalid JSON can carry perfectly well.
    """
    surface = _validated_surface(surface)
    path = root / "PUBLICATION_MANIFEST.json"
    if not path.is_file():
        if surface == "sdist":
            return None, []
        return None, [
            "PUBLICATION_MANIFEST.json is missing; the source surface publishes the "
            "engine and distribution identities and there is nothing to compare "
            "the code's literals against"
        ]
    text = _read(path)
    if text is None:
        return None, [
            "PUBLICATION_MANIFEST.json is not readable UTF-8 text; the engine and "
            "distribution cross-check cannot be skipped by making the record "
            "unreadable"
        ]
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as error:
        return None, [
            f"PUBLICATION_MANIFEST.json does not parse as JSON ({error.msg} at line "
            f"{error.lineno}); a manifest that cannot be read is not a manifest that "
            "agrees"
        ]
    if not isinstance(loaded, dict):
        return None, [
            f"PUBLICATION_MANIFEST.json is a {type(loaded).__name__} at its top "
            "level, not the object that carries the release record"
        ]
    release = loaded.get("release")
    if not isinstance(release, dict):
        found = "nothing" if "release" not in loaded else type(release).__name__
        return None, [
            f"PUBLICATION_MANIFEST.json carries {found} where the release mapping "
            "belongs; the engine and distribution identities are published there"
        ]
    return cast("dict[str, Any]", release), []


def _module_assignments(path: Path) -> dict[str, str]:
    """Top-level ``NAME = "literal"`` assignments, read without importing."""
    if not path.is_file():
        return {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            literal = node.value.value
        elif isinstance(node.value, ast.Name) and node.value.id in values:
            literal = values[node.value.id]
        else:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                values[target.id] = literal
    return values


def _module_int_assignments(path: Path) -> dict[str, int]:
    """Top-level ``NAME = 2`` assignments, read without importing.

    Separate from :func:`_module_assignments` rather than folded into it: the
    two answer questions about different kinds of identity, and a scanner that
    returned "whatever literal was there" would report a version string as a
    contract number. A name assigned anything but an integer literal — a
    computed expression, a call, a lookup — is simply absent here, which is what
    makes "this cannot be pinned" a statable problem rather than a silent pass.
    """
    if not path.is_file():
        return {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, int] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant):
            continue
        literal = node.value.value
        if not isinstance(literal, int) or isinstance(literal, bool):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                values[target.id] = literal
    return values


def check_required_files(
    root: Path = REPO_ROOT, *, surface: Surface = "source"
) -> list[str]:
    """Required files for the explicitly selected source or sdist surface."""
    surface = _validated_surface(surface)
    problems: list[str] = []
    if surface == "sdist":
        required = REQUIRED_FILES
    else:
        required = REQUIRED_FILES + SOURCE_CONTROL_FILES
    for name, marker in required:
        path = root / name
        if not path.is_file():
            problems.append(f"missing required file: {name}")
            continue
        text = _read(path)
        if text is None or marker not in text:
            problems.append(
                f"{name} does not read as itself (expected to contain {marker!r})"
            )
    if (root / "PUBLICATION_MANIFEST.template.json").exists():
        problems.append(
            "PUBLICATION_MANIFEST.template.json is still present; the candidate "
            "carries the manifest itself, not its template"
        )
    return problems


def check_no_credential_shapes(root: Path = REPO_ROOT) -> list[str]:
    """No credential-shaped literal outside the allowlisted synthetic ones."""
    problems: list[str] = []
    compiled = [(label, re.compile(pattern)) for label, pattern in CREDENTIAL_PATTERNS]
    for path in candidate_files(root):
        text = _read(path)
        if text is None:
            continue
        for label, pattern in compiled:
            for match in pattern.finditer(text):
                literal = match.group(0)
                if literal in ALLOWED_SYNTHETIC_KEY_LITERALS:
                    continue
                line = text.count("\n", 0, match.start()) + 1
                problems.append(
                    f"{_relative(path, root)}:{line}: {label} literal {literal!r}"
                )
    return problems


def check_no_private_references(root: Path = REPO_ROOT) -> list[str]:
    """No absolute local path, internal hostname or foreign repository."""
    problems: list[str] = []
    compiled = [
        (label, re.compile(pattern)) for label, pattern in PRIVATE_REFERENCE_PATTERNS
    ]
    owner = re.compile(r"github\.com[/:]([A-Za-z0-9_.-]+)/")
    for path in candidate_files(root):
        text = _read(path)
        if text is None:
            continue
        relative = _relative(path, root)
        if relative == "tools/check_public_release.py":
            # This module states the patterns it forbids; matching itself would
            # make the gate unrunnable rather than strict.
            continue
        for label, pattern in compiled:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                problems.append(f"{relative}:{line}: {label} {match.group(0)!r}")
        for match in owner.finditer(text):
            if match.group(1) not in ALLOWED_GITHUB_OWNERS:
                line = text.count("\n", 0, match.start()) + 1
                problems.append(
                    f"{relative}:{line}: repository owner {match.group(1)!r} is not "
                    "an approved public identity or a cited source"
                )
    return problems


def _is_workflow_path(relative: str) -> bool:
    if relative.startswith("/") or "\\" in relative:
        return False
    parts = relative.split("/")
    return (
        len(parts) == 3
        and all(parts)
        and parts[:2] == [".github", "workflows"]
        and parts[2] not in {".", ".."}
        and parts[2].endswith((".yml", ".yaml"))
        and "/".join(parts) == relative
    )


def _is_allowed_pinned_action_uses_line(line: str) -> bool:
    match = PINNED_ACTION_USES_LINE.fullmatch(line)
    if match is None:
        return False
    segments = match.group("target").split("/")
    return (
        len(segments) >= 2
        and segments[0] in ALLOWED_GITHUB_ACTION_OWNERS
        and all(
            segment not in {"", ".", ".."}
            and ACTION_TARGET_SEGMENT.fullmatch(segment) is not None
            for segment in segments
        )
    )


def scan_bare_commit_identifiers(
    text: str, *, where: str, workflow_path: str | None = None
) -> list[str]:
    """Reject commit tokens except canonical approved-owner SHA-pinned Actions."""
    problems: list[str] = []
    workflow = workflow_path is not None and _is_workflow_path(workflow_path)
    for match in BARE_COMMIT_IDENTIFIER.finditer(text):
        line_number = text.count("\n", 0, match.start()) + 1
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        if line_end == -1:
            line_end = len(text)
        line_text = text[line_start:line_end]
        if workflow and _is_allowed_pinned_action_uses_line(line_text):
            continue
        problems.append(f"{where}:{line_number}: bare 40-hex commit identifier")
    return problems


def check_no_bare_commit_identifiers(root: Path = REPO_ROOT) -> list[str]:
    """No internal commit identity outside a pinned GitHub Action reference."""
    problems: list[str] = []
    for path in candidate_files(root):
        text = _read(path)
        if text is None:
            continue
        relative = _relative(path, root)
        problems.extend(
            scan_bare_commit_identifiers(
                text,
                where=relative,
                workflow_path=relative,
            )
        )
    return problems


def check_no_internal_documents(root: Path = REPO_ROOT) -> list[str]:
    """No explicitly non-public document, present or cited."""
    problems: list[str] = []
    compiled = [
        (label, re.compile(pattern)) for label, pattern in INTERNAL_DOCUMENT_PATTERNS
    ]
    for path in candidate_files(root):
        relative = _relative(path, root)
        if relative == "tools/check_public_release.py":
            continue
        for label, pattern in compiled:
            if pattern.search(path.name):
                problems.append(f"{relative}: is a {label}")
        text = _read(path)
        if text is None:
            continue
        for label, pattern in compiled:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                problems.append(f"{relative}:{line}: names {label} {match.group(0)!r}")
    return problems


def scan_private_provenance(text: str, *, where: str) -> list[str]:
    """Every private-provenance marker in one blob of text, as located problems.

    Split out from the tree walk because the same rules have to be asked of two
    different things: the working tree, and the wheel and sdist built from it.
    """
    problems: list[str] = []
    for label, pattern in _PROVENANCE_RULES:
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            found = " ".join(match.group(0).split())
            if len(found) > 80:
                found = found[:77] + "..."
            problems.append(f"{where}:{line}: {label} {found!r}")
    return problems


_PROVENANCE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (label, re.compile(pattern)) for label, pattern in PRIVATE_PROVENANCE_PATTERNS
)


def scan_public_scope(text: str, *, where: str) -> list[str]:
    """Every internal-stage label and topology contradiction in one blob of text.

    Split out for the same reason :func:`scan_private_provenance` is: the tree,
    the wheel and the sdist are three different selections of the same content,
    and the prose a user actually installs is the one in the wheel.
    """
    problems: list[str] = []
    for label, pattern in _PUBLIC_SCOPE_RULES:
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            found = " ".join(match.group(0).split())
            problems.append(f"{where}:{line}: {label} {found!r}")
    return problems


_PUBLIC_SCOPE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (label, re.compile(pattern)) for label, pattern in PUBLIC_SCOPE_PATTERNS
)


def check_public_scope_prose(root: Path = REPO_ROOT) -> list[str]:
    """No internal stage label, and no prose contradicting what is shipped.

    Exempt by exact path, and only the module that *states* the rules: this
    file necessarily contains every label form it forbids, and a scanner that
    flagged itself would be a gate nobody could run rather than a strict one.
    The exemption is never by directory — a document that merely discusses the
    topic gets no relief, and neither does a test that plants one of these
    strings, which is why the suite builds them rather than spelling them out.
    """
    problems: list[str] = []
    for path in candidate_files(root):
        relative = _relative(path, root)
        if relative == "tools/check_public_release.py":
            continue
        text = _read(path)
        if text is None:
            continue
        problems.extend(scan_public_scope(text, where=relative))
    return problems


def check_no_private_provenance(root: Path = REPO_ROOT) -> list[str]:
    """No inaccessible validation or execution provenance.

    Public source may explain a contract or methodology. It may not claim
    non-public validation or point to execution evidence readers cannot access.
    """
    problems: list[str] = []
    for path in candidate_files(root):
        relative = _relative(path, root)
        if relative in PROVENANCE_POLICY_DOCUMENTS:
            continue
        text = _read(path)
        if text is None:
            continue
        problems.extend(scan_private_provenance(text, where=relative))
    return problems


def check_built_distributions(
    root: Path = REPO_ROOT, *, distribution_dir: Path | None = None
) -> list[str]:
    """Validate built archives against the reviewed source without extracting them.

    The implicit ``root/dist`` remains optional for source-only invocations.  An
    explicitly supplied directory is a release boundary and therefore fails
    closed when it is absent or is not a directory.
    """
    import base64
    import configparser
    import csv
    import hashlib
    import io
    import os
    import stat
    import tarfile
    import unicodedata
    import zipfile

    max_members = 10_000
    max_member_size = 16 * 1024 * 1024
    max_archive_size = 128 * 1024 * 1024
    max_compressed_size = 64 * 1024 * 1024
    supported_zip_compression = {
        zipfile.ZIP_STORED,
        zipfile.ZIP_DEFLATED,
        zipfile.ZIP_BZIP2,
        zipfile.ZIP_LZMA,
    }
    max_findings = 256
    max_diagnostic_size = 512
    suppressed_code = "additional_distribution_findings_suppressed"

    def bounded_diagnostic(message: str) -> str:
        if (
            len(message) <= max_diagnostic_size
            and len(message.encode("utf-8")) <= max_diagnostic_size
        ):
            return message
        suffix = "...[truncated]"
        budget = max_diagnostic_size - len(suffix)
        prefix = message[:budget]
        while len(prefix.encode("utf-8")) > budget:
            prefix = prefix[:-1]
        return prefix + suffix

    class FindingList(list[str]):
        def append(self, message: str) -> None:
            if len(self) < max_findings:
                super().append(bounded_diagnostic(message))
            elif len(self) == max_findings:
                super().append(suppressed_code)

        def extend(self, messages: Iterable[str]) -> None:
            for message in messages:
                self.append(message)

    problems = FindingList()
    explicit = distribution_dir is not None
    distribution = root / "dist" if distribution_dir is None else distribution_dir
    try:
        distribution_mode = distribution.lstat().st_mode
    except FileNotFoundError:
        if explicit:
            problems.append(
                f"distribution directory is absent or not a directory: {distribution}"
            )
        return problems
    except OSError as error:
        problems.append(f"distribution directory cannot be inspected: {error.strerror}")
        return problems
    if stat.S_ISLNK(distribution_mode):
        problems.append(f"distribution directory is a symlink: {distribution}")
        return problems
    if not stat.S_ISDIR(distribution_mode):
        problems.append(
            f"distribution directory is absent or not a directory: {distribution}"
        )
        return problems

    def name_exceeds_limit(name: str) -> bool:
        return len(name) > 1_024 or len(name.encode("utf-8")) > 1_024

    def member_label(name: str) -> str:
        prefix = "".join(
            character if character.isprintable() else ascii(character)[1:-1]
            for character in name[:80]
        )
        return prefix if len(name) <= 80 else f"{prefix}...[truncated]"

    def safe_name(name: str, *, directory: bool = False) -> str | None:
        if name_exceeds_limit(name):
            return None
        candidate = name[:-1] if directory and name.endswith("/") else name
        if (
            not candidate
            or candidate.startswith("/")
            or "\\" in candidate
            or "\x00" in candidate
            or (not directory and name.endswith("/"))
        ):
            return None
        parts = candidate.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            return None
        return unicodedata.normalize("NFC", "/".join(parts))

    def source_oracle() -> dict[str, bytes] | None:
        if not (root / "pyproject.toml").is_file():
            return None
        expected: dict[str, bytes] = {}
        excluded_directories = {
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
        }

        def visit(directory: Path) -> None:
            try:
                entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
            except OSError as error:
                problems.append(
                    f"source oracle cannot read {directory}: {error.strerror}"
                )
                return
            for entry in entries:
                path = Path(entry.path)
                try:
                    mode = path.lstat().st_mode
                except OSError as error:
                    problems.append(f"source oracle cannot stat {path}: {error.strerror}")
                    continue
                if stat.S_ISLNK(mode):
                    problems.append(
                        f"source oracle rejects symlink: {path.relative_to(root)}"
                    )
                elif stat.S_ISDIR(mode):
                    if path.name not in excluded_directories:
                        visit(path)
                elif stat.S_ISREG(mode):
                    if path.suffix not in {".pyc", ".pyo"}:
                        expected[path.relative_to(root / "src").as_posix()] = (
                            path.read_bytes()
                        )
                else:
                    problems.append(
                        f"source oracle rejects non-regular entry: "
                        f"{path.relative_to(root)}"
                    )

        for package in ("operatebench", "boundarybench"):
            package_root = root / "src" / package
            if not package_root.is_dir() or package_root.is_symlink():
                problems.append(
                    f"source oracle package is missing or unsafe: src/{package}"
                )
                continue
            visit(package_root)
        return expected

    expected_source = source_oracle()

    def inspect(
        archive: str,
        member: str,
        data: bytes,
        *,
        workflow_path: str | None,
        provenance_exempt: bool = False,
    ) -> None:
        if member.endswith("/tests/matched_control_fixtures.py"):
            problems.append(
                f"dist/{archive}::{member}: test-only helper shipped in distribution"
            )
        text = _decode_text(data)
        if text is None:
            return
        where = f"dist/{archive}::{member}"
        if not provenance_exempt:
            problems.extend(scan_private_provenance(text, where=where))
            problems.extend(scan_public_scope(text, where=where))
        problems.extend(
            scan_bare_commit_identifiers(text, where=where, workflow_path=workflow_path)
        )
        for label, pattern in PRIVATE_REFERENCE_PATTERNS:
            for match in re.finditer(pattern, text):
                line = text.count("\n", 0, match.start()) + 1
                problems.append(f"{where}:{line}: {label} {match.group(0)!r}")
        for label, pattern in CREDENTIAL_PATTERNS:
            for match in re.finditer(pattern, text):
                if match.group(0) in ALLOWED_SYNTHETIC_KEY_LITERALS:
                    continue
                line = text.count("\n", 0, match.start()) + 1
                problems.append(f"{where}:{line}: {label} literal {match.group(0)!r}")

    wheels = sorted(distribution.glob("*.whl"))
    sdists = sorted(distribution.glob("*.tar.gz"))
    exact_cardinality = len(wheels) == 1 and len(sdists) == 1
    if not exact_cardinality:
        problems.append(
            "distribution directory must contain exactly one wheel and one sdist "
            f"(found {len(wheels)} wheel(s), {len(sdists)} sdist(s))"
        )

    if exact_cardinality and expected_source is not None:
        expected_archive_names = {
            f"{DISTRIBUTION_NAME}-{DISTRIBUTION_VERSION}-py3-none-any.whl",
            f"{DISTRIBUTION_NAME}-{DISTRIBUTION_VERSION}.tar.gz",
        }
        actual_archive_names = {wheels[0].name, sdists[0].name}
        if actual_archive_names != expected_archive_names:
            problems.append(
                "archive filename identity differs from reviewed distribution "
                f"(found={sorted(actual_archive_names)!r})"
            )

    unsafe_archives: set[Path] = set()
    for archive in (*wheels, *sdists):
        try:
            archive_stat = archive.lstat()
        except OSError:
            archive_stat = None
        if archive_stat is None or not stat.S_ISREG(archive_stat.st_mode):
            problems.append(
                f"dist/{archive.name}: archive is not a regular file (symlinks rejected)"
            )
            unsafe_archives.add(archive)
        elif archive_stat.st_size > max_compressed_size:
            problems.append(f"dist/{archive.name}: archive exceeds compressed size limit")
            unsafe_archives.add(archive)

    for wheel in wheels:
        if wheel in unsafe_archives:
            continue
        try:
            with zipfile.ZipFile(wheel) as wheel_archive:
                entries: list[Any] = wheel_archive.infolist()
                if not entries:
                    problems.append(f"dist/{wheel.name}: wheel is empty")
                    continue
                if len(entries) > max_members:
                    problems.append(
                        f"dist/{wheel.name}: wheel exceeds member limit {max_members}"
                    )
                    continue
                aggregate = sum(entry.file_size for entry in entries)
                if aggregate > max_archive_size:
                    problems.append(
                        f"dist/{wheel.name}: wheel exceeds uncompressed size limit"
                    )
                    continue
                oversized_member = False
                for entry in entries:
                    if entry.file_size > max_member_size:
                        problems.append(
                            f"dist/{wheel.name}::{member_label(entry.filename)}: "
                            "member exceeds size limit"
                        )
                        oversized_member = True
                if oversized_member:
                    continue
                names: dict[str, zipfile.ZipInfo] = {}
                data: dict[str, bytes] = {}
                for entry in entries:
                    if name_exceeds_limit(entry.filename):
                        problems.append(
                            f"dist/{wheel.name}::{member_label(entry.filename)}: "
                            "member name exceeds limit"
                        )
                        continue
                    normalized = safe_name(entry.filename, directory=entry.is_dir())
                    if normalized is None:
                        problems.append(
                            f"dist/{wheel.name}::{member_label(entry.filename)}: "
                            f"unsafe wheel member name"
                        )
                        continue
                    if normalized in names:
                        problems.append(
                            f"dist/{wheel.name}::{member_label(entry.filename)}: "
                            f"duplicate normalized wheel member"
                        )
                        continue
                    names[normalized] = entry
                    if entry.flag_bits & 1:
                        problems.append(
                            f"dist/{wheel.name}::{member_label(entry.filename)}: "
                            "encrypted wheel member"
                        )
                        continue
                    if entry.compress_type not in supported_zip_compression:
                        problems.append(
                            f"dist/{wheel.name}::{member_label(entry.filename)}: "
                            f"unsupported ZIP compression"
                        )
                        continue
                    if entry.file_size > max_member_size:
                        problems.append(
                            f"dist/{wheel.name}::{member_label(entry.filename)}: "
                            f"member exceeds size limit"
                        )
                        continue
                    mode = entry.external_attr >> 16 if entry.create_system == 3 else 0
                    file_type = stat.S_IFMT(mode)
                    if entry.is_dir() or file_type not in {0, stat.S_IFREG}:
                        problems.append(
                            f"dist/{wheel.name}::{member_label(entry.filename)}: "
                            f"wheel member is not regular"
                        )
                        continue
                    try:
                        raw = wheel_archive.read(entry)
                    except (RuntimeError, NotImplementedError, OSError) as error:
                        problems.append(
                            f"dist/{wheel.name}::{member_label(entry.filename)}: "
                            f"unreadable wheel member ({type(error).__name__})"
                        )
                        continue
                    data[normalized] = raw
                    inspect(
                        wheel.name,
                        normalized,
                        raw,
                        workflow_path=normalized,
                    )

                if expected_source is None:
                    continue
                package_data = {
                    name: raw
                    for name, raw in data.items()
                    if name.startswith(("operatebench/", "boundarybench/"))
                }
                if set(package_data) != set(expected_source):
                    missing = sorted(set(expected_source) - set(package_data))
                    extra = sorted(set(package_data) - set(expected_source))
                    problems.append(
                        f"dist/{wheel.name}: wheel production membership differs "
                        f"(missing={missing[:5]!r}, extra={extra[:5]!r})"
                    )
                for name in sorted(set(package_data) & set(expected_source)):
                    if package_data[name] != expected_source[name]:
                        problems.append(
                            f"dist/{wheel.name}::{name}: wheel production source differs"
                        )
                if any("tests/fixtures/" in name for name in names):
                    problems.append(f"dist/{wheel.name}: wheel carries test fixtures")

                normalized_distribution = DISTRIBUTION_NAME.replace("-", "_")
                expected_dist_info = (
                    f"{normalized_distribution}-{DISTRIBUTION_VERSION}.dist-info"
                )
                pyproject = tomllib.loads(
                    (root / "pyproject.toml").read_text(encoding="utf-8")
                )
                project = pyproject["project"]
                license_members = {
                    f"{expected_dist_info}/licenses/{license_name}": (
                        root / license_name
                    ).read_bytes()
                    for license_name in project["license-files"]
                }
                mandatory = {"METADATA", "WHEEL", "RECORD", "entry_points.txt"}
                expected_wheel_members = (
                    set(expected_source)
                    | set(license_members)
                    | {f"{expected_dist_info}/{basename}" for basename in mandatory}
                )
                if set(names) != expected_wheel_members:
                    missing = sorted(expected_wheel_members - set(names))
                    extra = sorted(set(names) - expected_wheel_members)
                    problems.append(
                        f"dist/{wheel.name}: wheel member universe differs "
                        f"(missing={missing[:5]!r}, extra={extra[:5]!r})"
                    )
                for name in sorted(set(data) & set(license_members)):
                    if data[name] != license_members[name]:
                        problems.append(
                            f"dist/{wheel.name}::{name}: reviewed license bytes differ"
                        )
                roots = {
                    name.split("/", 1)[0]
                    for name in names
                    if name.split("/", 1)[0].endswith(".dist-info")
                }
                if roots != {expected_dist_info}:
                    problems.append(
                        f"dist/{wheel.name}: dist-info root is {sorted(roots)!r}, "
                        f"expected {[expected_dist_info]!r}"
                    )
                for basename in sorted(mandatory):
                    member_name = f"{expected_dist_info}/{basename}"
                    if member_name not in data:
                        problems.append(
                            f"dist/{wheel.name}: mandatory {member_name} "
                            f"is missing or not regular"
                        )

                record_name = f"{expected_dist_info}/RECORD"
                record_valid = record_name in data
                if record_valid:
                    try:
                        rows = list(
                            csv.reader(
                                io.StringIO(data[record_name].decode("utf-8")),
                                strict=True,
                            )
                        )
                    except (UnicodeDecodeError, csv.Error) as error:
                        problems.append(
                            f"dist/{wheel.name}::{record_name}: invalid RECORD "
                            f"({type(error).__name__})"
                        )
                        record_valid = False
                    else:
                        recorded: dict[str, tuple[str, str]] = {}
                        for row in rows:
                            if len(row) != 3:
                                problems.append(
                                    f"dist/{wheel.name}::{record_name}: RECORD row "
                                    f"does not have exactly three columns"
                                )
                                record_valid = False
                                continue
                            record_member, digest_field, size_field = row
                            normalized = safe_name(record_member)
                            if normalized is None or normalized != record_member:
                                problems.append(
                                    f"dist/{wheel.name}::{record_name}: "
                                    f"unsafe RECORD member name"
                                )
                                record_valid = False
                                continue
                            if record_member in recorded:
                                problems.append(
                                    f"dist/{wheel.name}::{record_name}: duplicate RECORD "
                                    f"row for {record_member}"
                                )
                                record_valid = False
                                continue
                            recorded[record_member] = (digest_field, size_field)

                        if set(recorded) != set(data):
                            missing = sorted(set(data) - set(recorded))
                            extra = sorted(set(recorded) - set(data))
                            problems.append(
                                f"dist/{wheel.name}::{record_name}: RECORD member set "
                                f"differs (missing={missing[:5]!r}, extra={extra[:5]!r})"
                            )
                            record_valid = False

                        for member_name, (digest_field, size_field) in recorded.items():
                            if member_name == record_name:
                                if digest_field or size_field:
                                    problems.append(
                                        f"dist/{wheel.name}::{record_name}: "
                                        f"RECORD self-row "
                                        f"hash and size must be empty"
                                    )
                                    record_valid = False
                                continue
                            member_raw = data.get(member_name)
                            if member_raw is None:
                                continue
                            expected_digest = (
                                base64.urlsafe_b64encode(
                                    hashlib.sha256(member_raw).digest()
                                )
                                .rstrip(b"=")
                                .decode("ascii")
                            )
                            if digest_field != f"sha256={expected_digest}":
                                problems.append(
                                    f"dist/{wheel.name}::{record_name}: {member_name} "
                                    f"has an invalid sha256 digest"
                                )
                                record_valid = False
                            if size_field != str(len(member_raw)):
                                problems.append(
                                    f"dist/{wheel.name}::{record_name}: {member_name} "
                                    f"has an invalid byte size"
                                )
                                record_valid = False

                # RECORD authenticates these bytes before any metadata is trusted.
                if record_valid:
                    wheel_metadata_name = f"{expected_dist_info}/WHEEL"
                    if wheel_metadata_name in data:
                        from email import policy
                        from email.parser import BytesParser

                        try:
                            wheel_message = BytesParser(
                                policy=policy.default.clone(raise_on_defect=True)
                            ).parsebytes(data[wheel_metadata_name])
                            expected_wheel_headers = {
                                "wheel-version",
                                "generator",
                                "root-is-purelib",
                                "tag",
                            }
                            if {
                                field.lower() for field in wheel_message
                            } != expected_wheel_headers:
                                raise ValueError("WHEEL header set differs")
                            for field in (
                                "Wheel-Version",
                                "Generator",
                                "Root-Is-Purelib",
                                "Tag",
                            ):
                                if len(wheel_message.get_all(field, [])) != 1:
                                    raise ValueError(
                                        f"WHEEL {field} is missing or duplicated"
                                    )
                            if str(wheel_message["Wheel-Version"]) != "1.0":
                                raise ValueError("Wheel-Version differs")
                            if str(wheel_message["Generator"]) != "hatchling 1.32.0":
                                raise ValueError("Generator differs from pinned backend")
                            if str(wheel_message["Root-Is-Purelib"]) != "true":
                                raise ValueError("Root-Is-Purelib differs")
                            if wheel_message.get_all("Tag", []) != ["py3-none-any"]:
                                raise ValueError("WHEEL Tag differs from filename")
                            if wheel_message.get_payload() not in ("", "\n"):
                                raise ValueError("WHEEL carries an unexpected body")
                        except (UnicodeError, ValueError) as error:
                            problems.append(
                                f"dist/{wheel.name}::{wheel_metadata_name}: invalid "
                                f"WHEEL ({type(error).__name__})"
                            )

                    entry_name = f"{expected_dist_info}/entry_points.txt"
                    if entry_name in data:
                        parser = configparser.ConfigParser(
                            interpolation=None, strict=True, delimiters=("=",)
                        )
                        cast(Any, parser).optionxform = str
                        try:
                            parser.read_string(data[entry_name].decode("utf-8"))
                            if (
                                parser.sections() != ["console_scripts"]
                                or parser.defaults()
                            ):
                                raise configparser.Error(
                                    "entry_points.txt must contain one "
                                    "console_scripts section"
                                )
                            actual_scripts = dict(parser.items("console_scripts"))
                        except (
                            UnicodeDecodeError,
                            configparser.Error,
                            KeyError,
                        ) as error:
                            problems.append(
                                f"dist/{wheel.name}::{entry_name}: invalid "
                                f"entry_points.txt ({type(error).__name__})"
                            )
                        else:
                            expected_scripts = pyproject["project"]["scripts"]
                            if actual_scripts != expected_scripts:
                                problems.append(
                                    f"dist/{wheel.name}::{entry_name}: console scripts "
                                    f"differ from pyproject.toml"
                                )

                    metadata_name = f"{expected_dist_info}/METADATA"
                    if metadata_name in data:
                        from email import policy
                        from email.parser import BytesParser

                        from packaging.requirements import InvalidRequirement, Requirement

                        try:
                            message = BytesParser(
                                policy=policy.default.clone(raise_on_defect=True)
                            ).parsebytes(data[metadata_name])
                            singleton_fields = (
                                "Metadata-Version",
                                "Name",
                                "Version",
                                "Summary",
                                "Requires-Python",
                                "License-Expression",
                                "Author-email",
                                "Maintainer-email",
                                "Keywords",
                                "Description-Content-Type",
                            )
                            project = tomllib.loads(
                                (root / "pyproject.toml").read_text(encoding="utf-8")
                            )["project"]
                            expected_field_names = Counter(
                                field.lower() for field in singleton_fields
                            )
                            expected_field_names.update(
                                {
                                    "license-file": len(project["license-files"]),
                                    "classifier": len(project["classifiers"]),
                                    "project-url": len(project["urls"]),
                                    "requires-dist": len(project["dependencies"]),
                                }
                            )
                            if Counter(field.lower() for field in message) != (
                                expected_field_names
                            ):
                                raise ValueError("METADATA header multiset differs")
                            if any(
                                len(message.get_all(field, [])) != 1
                                for field in singleton_fields
                            ):
                                raise ValueError("missing or duplicate singleton field")
                            expected_singletons = {
                                "Metadata-Version": "2.5",
                                "Name": DISTRIBUTION_NAME,
                                "Version": DISTRIBUTION_VERSION,
                                "Summary": project["description"],
                                "Requires-Python": project["requires-python"],
                                "License-Expression": project["license"],
                                "Author-email": ", ".join(
                                    f"{author['name']} <{author['email']}>"
                                    for author in project["authors"]
                                ),
                                "Maintainer-email": ", ".join(
                                    f"{maintainer['name']} <{maintainer['email']}>"
                                    for maintainer in project["maintainers"]
                                ),
                                "Description-Content-Type": "text/markdown",
                            }
                            if any(
                                str(message[field]) != expected
                                for field, expected in expected_singletons.items()
                            ):
                                raise ValueError("singleton identity differs")
                            if str(message["Keywords"]).split(",") != project["keywords"]:
                                raise ValueError("keyword projection differs")
                            separator = b"\n\n"
                            if separator not in data[metadata_name]:
                                raise ValueError("METADATA has no header/body separator")
                            metadata_body = data[metadata_name].split(separator, 1)[1]
                            if metadata_body != (root / project["readme"]).read_bytes():
                                raise ValueError("long description differs from README")
                            if (
                                message.get_all("Classifier", [])
                                != project["classifiers"]
                            ):
                                raise ValueError("classifier set differs")
                            expected_urls = {
                                f"{label}, {url}"
                                for label, url in project["urls"].items()
                            }
                            if set(message.get_all("Project-URL", [])) != expected_urls:
                                raise ValueError("project URL set differs")
                            if set(message.get_all("License-File", [])) != set(
                                project["license-files"]
                            ):
                                raise ValueError("license file set differs")
                            actual_requirements = {
                                Requirement(value)
                                for value in message.get_all("Requires-Dist", [])
                            }
                            expected_requirements = {
                                Requirement(value) for value in project["dependencies"]
                            }
                            if actual_requirements != expected_requirements:
                                raise ValueError("dependency requirement set differs")
                        except (
                            InvalidRequirement,
                            KeyError,
                            TypeError,
                            UnicodeError,
                            ValueError,
                        ) as error:
                            problems.append(
                                f"dist/{wheel.name}::{metadata_name}: invalid METADATA "
                                f"({type(error).__name__})"
                            )
        except (zipfile.BadZipFile, OSError, EOFError) as error:
            problems.append(
                f"dist/{wheel.name}: malformed wheel ({type(error).__name__})"
            )

    for sdist in sdists:
        if sdist in unsafe_archives:
            continue
        try:
            with sdist.open("rb") as archive_handle:
                sdist_names: set[str] = set()
                sdist_roots: set[str] = set()
                member_count = 0
                aggregate = 0
                reject_before_payload = False
                with tarfile.open(
                    fileobj=archive_handle, mode="r|gz"
                ) as metadata_archive:
                    for entry in metadata_archive:
                        member_count += 1
                        if member_count > max_members:
                            problems.append(
                                f"dist/{sdist.name}: sdist exceeds member limit "
                                f"{max_members}"
                            )
                            reject_before_payload = True
                            break
                        if name_exceeds_limit(entry.name):
                            problems.append(
                                f"dist/{sdist.name}::{member_label(entry.name)}: "
                                "member name exceeds limit"
                            )
                            continue
                        normalized = safe_name(entry.name, directory=entry.isdir())
                        if normalized is None:
                            problems.append(
                                f"dist/{sdist.name}::{member_label(entry.name)}: "
                                "unsafe sdist member name"
                            )
                            continue
                        sdist_roots.add(normalized.split("/", 1)[0])
                        if normalized in sdist_names:
                            problems.append(
                                f"dist/{sdist.name}::{member_label(entry.name)}: "
                                "duplicate normalized sdist member"
                            )
                            continue
                        sdist_names.add(normalized)
                        if not (entry.isfile() or entry.isdir()):
                            problems.append(
                                f"dist/{sdist.name}::{member_label(entry.name)}: sdist "
                                "member is not a safe directory or regular file"
                            )
                            continue
                        if entry.isfile():
                            aggregate += entry.size
                            if entry.size > max_member_size:
                                problems.append(
                                    f"dist/{sdist.name}::{member_label(entry.name)}: "
                                    "member exceeds size limit"
                                )
                                reject_before_payload = True
                            if aggregate > max_archive_size:
                                reject_before_payload = True

                if member_count == 0:
                    problems.append(f"dist/{sdist.name}: sdist is empty")
                    continue
                if aggregate > max_archive_size:
                    problems.append(
                        f"dist/{sdist.name}: sdist exceeds uncompressed size limit"
                    )
                if reject_before_payload:
                    continue

                archive_handle.seek(0)
                sdist_data: dict[str, bytes] = {}
                with tarfile.open(fileobj=archive_handle, mode="r|gz") as payload_archive:
                    for entry in payload_archive:
                        normalized = safe_name(entry.name, directory=entry.isdir())
                        if normalized is None or not entry.isfile():
                            continue
                        handle = payload_archive.extractfile(entry)
                        if handle is None:
                            problems.append(
                                f"dist/{sdist.name}::{member_label(entry.name)}: "
                                "regular member is unreadable"
                            )
                            continue
                        raw = handle.read(max_member_size + 1)
                        if len(raw) != entry.size:
                            problems.append(
                                f"dist/{sdist.name}::{member_label(entry.name)}: "
                                "member size does not match payload"
                            )
                            continue
                        sdist_data[normalized] = raw

                archive_root = next(iter(sdist_roots)) if len(sdist_roots) == 1 else None
                if archive_root is None:
                    problems.append(
                        f"dist/{sdist.name}: sdist must contain exactly one safe root"
                    )
                canonical_scanner = (
                    f"{archive_root}/tools/check_public_release.py"
                    if archive_root
                    else None
                )
                for name, raw in sdist_data.items():
                    relative = (
                        name.removeprefix(f"{archive_root}/") if archive_root else None
                    )
                    inspect(
                        sdist.name,
                        name,
                        raw,
                        workflow_path=relative,
                        provenance_exempt=name == canonical_scanner,
                    )

                if expected_source is None or archive_root is None:
                    continue
                prefix = f"{archive_root}/src/"
                production_data = {
                    name.removeprefix(prefix): raw
                    for name, raw in sdist_data.items()
                    if name.startswith(
                        (f"{prefix}operatebench/", f"{prefix}boundarybench/")
                    )
                }
                if set(production_data) != set(expected_source):
                    missing = sorted(set(expected_source) - set(production_data))
                    extra = sorted(set(production_data) - set(expected_source))
                    problems.append(
                        f"dist/{sdist.name}: sdist production membership differs "
                        f"(missing={missing[:5]!r}, extra={extra[:5]!r})"
                    )
                for name in sorted(set(production_data) & set(expected_source)):
                    member_name = f"{prefix}{name}"
                    if production_data[name] != expected_source[name]:
                        problems.append(
                            f"dist/{sdist.name}::{member_name}: sdist production "
                            f"source differs"
                        )
                for relative in (
                    "pyproject.toml",
                    "uv.lock",
                    ".github/workflows/ci.yml",
                ):
                    member_name = f"{archive_root}/{relative}"
                    source_path = root / relative
                    if member_name not in sdist_data:
                        problems.append(
                            f"dist/{sdist.name}: provenance member {member_name} "
                            f"is missing"
                        )
                    elif sdist_data[member_name] != source_path.read_bytes():
                        problems.append(
                            f"dist/{sdist.name}::{member_name}: provenance bytes differ"
                        )
        except (tarfile.TarError, OSError, EOFError) as error:
            problems.append(
                f"dist/{sdist.name}: malformed sdist ({type(error).__name__})"
            )

    return problems


def check_no_distribution_test_helpers(
    root: Path = REPO_ROOT, *, surface: Surface = "source"
) -> list[str]:
    """The source keeps test helpers; an extracted sdist must not carry them."""
    surface = _validated_surface(surface)
    if surface == "source":
        return []
    relative = Path("tests/matched_control_fixtures.py")
    if (root / relative).is_file():
        return [f"{relative.as_posix()}: test-only helper shipped in sdist"]
    return []


def check_synthetic_fixture(root: Path = REPO_ROOT) -> list[str]:
    """The shipped executable fixture declares itself synthetic and non-production."""
    problems: list[str] = []
    path = root / FIXTURE
    if not path.is_file():
        return [f"missing the shipped operation fixture: {FIXTURE.as_posix()}"]
    text = path.read_text(encoding="utf-8")
    if "privacy_status: SYNTHETIC_ONLY" not in text:
        problems.append(f"{FIXTURE.as_posix()}: privacy_status is not SYNTHETIC_ONLY")
    if "disclaimer:" not in text:
        problems.append(f"{FIXTURE.as_posix()}: carries no disclaimer")
    if "data_provenance:" not in text:
        problems.append(f"{FIXTURE.as_posix()}: carries no data_provenance statement")
    lowered = text.lower()
    if "not dwelly policy" not in lowered:
        problems.append(
            f"{FIXTURE.as_posix()}: does not state that it is not Dwelly policy"
        )
    if "fully synthetic" not in lowered:
        problems.append(
            f"{FIXTURE.as_posix()}: does not state that it is fully synthetic"
        )
    return problems


def check_relative_links(
    root: Path = REPO_ROOT, *, surface: Surface = "source"
) -> list[str]:
    """Every relative Markdown link and anchor resolves inside the tree."""
    surface = _validated_surface(surface)
    problems: list[str] = []
    omitted_sdist_controls = {(root / name).resolve() for name, _ in SOURCE_CONTROL_FILES}
    link = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
    heading = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.MULTILINE)
    for path in candidate_files(root):
        if path.suffix != ".md":
            continue
        text = _read(path)
        if text is None:
            continue
        for match in link.finditer(text):
            target = match.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            file_part, _, anchor = target.partition("#")
            line = text.count("\n", 0, match.start()) + 1
            resolved = (path.parent / file_part).resolve() if file_part else path
            if not resolved.exists():
                if surface == "sdist" and resolved in omitted_sdist_controls:
                    continue
                problems.append(
                    f"{_relative(path, root)}:{line}: relative link {target!r} does "
                    "not resolve"
                )
                continue
            if anchor and resolved.suffix == ".md":
                body = _read(resolved) or ""
                slugs = {_slug(found) for found in heading.findall(body)}
                if anchor not in slugs:
                    problems.append(
                        f"{_relative(path, root)}:{line}: anchor {'#' + anchor!r} is "
                        f"not a heading in {file_part or path.name}"
                    )
    return problems


def _slug(heading: str) -> str:
    """GitHub's heading slug, for the subset of Markdown this tree uses."""
    text = re.sub(r"`|\*|_", "", heading).strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s+", "-", text)


def check_no_generated_artifacts(root: Path = REPO_ROOT) -> list[str]:
    """Nothing generated could be tracked: every class stays ignored, everywhere.

    Deliberately not "no generated file exists". Running the documented commands
    creates ``dist/``, ``.ruff_cache/`` and ``.coverage``; failing because the
    reviewer built the wheel would teach them to skip the gate. What matters is
    that none of it can be *committed*, so the test is coverage by .gitignore.
    """
    problems: list[str] = []
    ignore_file = root / ".gitignore"
    if not ignore_file.is_file():
        return [".gitignore is missing; generated output has nothing keeping it out"]
    entries = {
        line.strip()
        for line in ignore_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for required in REQUIRED_IGNORE_ENTRIES:
        if required not in entries:
            problems.append(f".gitignore does not ignore {required!r}")

    ignored_names = {entry.rstrip("/") for entry in entries}

    def is_ignored(name: str) -> bool:
        if name in ignored_names:
            return True
        return any(
            pattern.startswith("*") and name.endswith(pattern.lstrip("*"))
            for pattern in ignored_names
        )

    for path in _every_entry(root):
        name = path.name
        generated = name in GENERATED_NAMES or name.endswith(GENERATED_SUFFIXES)
        if generated and not is_ignored(name):
            problems.append(
                f"{_relative(path, root)}: generated artefact is not covered by "
                ".gitignore, so it can be committed as content"
            )
    return problems


def _every_entry(root: Path) -> Iterator[Path]:
    """Every path in the tree, including the ones the text scanner skips.

    The generated-artefact check is the one question that has to be asked *of*
    the skipped directories rather than inside them: it wants to know that they
    are ignored, so it must be able to see that they exist. It never opens them.
    """
    for entry in sorted(root.iterdir()):
        if entry.name == ".git" or entry.is_symlink():
            continue
        yield entry
        if entry.is_dir() and entry.name not in SKIPPED_DIRECTORIES:
            yield from _every_entry(entry)


ALL_CHECKS: tuple[tuple[str, object], ...] = (
    ("distribution metadata", check_distribution_metadata),
    ("package versions", check_package_versions),
    ("required licensing and documentation files", check_required_files),
    ("credential shapes", check_no_credential_shapes),
    ("private paths and repository references", check_no_private_references),
    ("bare internal commit identifiers", check_no_bare_commit_identifiers),
    ("non-public document references", check_no_internal_documents),
    ("non-public validation and execution provenance", check_no_private_provenance),
    ("internal stage labels and shipped topology", check_public_scope_prose),
    ("built wheel and sdist contents", check_built_distributions),
    ("distribution test-only helpers", check_no_distribution_test_helpers),
    ("synthetic-only executable fixture", check_synthetic_fixture),
    ("relative links and anchors", check_relative_links),
    ("generated artefacts", check_no_generated_artifacts),
)


def run_all(
    root: Path = REPO_ROOT, *, surface: Surface = "source"
) -> dict[str, list[str]]:
    """Every check, keyed by name. An empty list is a pass."""
    surface = _validated_surface(surface)
    results: dict[str, list[str]] = {}
    for name, check in ALL_CHECKS:
        if (
            check is check_required_files
            or check is check_relative_links
            or check is check_package_versions
            or check is check_no_distribution_test_helpers
        ):
            results[name] = check(root, surface=surface)
        else:
            results[name] = check(root)  # type: ignore[operator]
    return results


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=REPO_ROOT)
    parser.add_argument("--surface", choices=sorted(VALID_SURFACES), default="source")
    arguments = parser.parse_args(list(argv) if argv is not None else [])
    root = arguments.root.resolve()
    results = run_all(root, surface=arguments.surface)
    failures = 0
    for name, problems in results.items():
        if problems:
            failures += 1
            print(f"FAILED  {name}")
            for problem in problems:
                print(f"    - {problem}")
        else:
            print(f"ok      {name}")
    if failures:
        print(f"\npublic-release checks: {failures} of {len(results)} failed")
        return 1
    print(f"\npublic-release checks: all {len(results)} passed")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
