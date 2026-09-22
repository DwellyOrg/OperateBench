"""What the shipped package says about itself, on every surface it ships from.

The defect this file exists to stop: the installed package's own docstring
described a topology that does not exist and named steps in a plan no reader is
part of. It called itself a numbered slice of an internal plan, claimed a second
distribution that has never existed — one repository and one wheel building both
of them — and deferred curating the public export to a lettered stage. One
distribution ships, containing two importable packages; the rest was internal
vocabulary that travelled into a wheel.

Three obligations follow, and each is asserted here.

**The prose is true, and it is checked where a user meets it.** Not only in
``src/``: the same docstring is asserted inside the built wheel and the built
sdist, because the artefact a user installs is the thing that makes the claim.

**The gate fails closed on the classes, not on this instance.** The scanner
gained a check for internal stage labels and for prose contradicting the shipped
topology, and it is exercised by planting each class in a tree, in a wheel and
in an sdist — a check only ever run against a passing repository is not evidence
that it can fail.

**It does not fire on legitimate vocabulary.** ``Stage``, ``phase``, ``pass``
and ``P0``/``P1`` all mean something behavioural in this tree — a staged
execution, a policy arm, a test that passes. A gate that flagged those would be
one people learn to route around, so the honest negatives are asserted too.

Every forbidden string in this file is *built* rather than written out. The
scanner walks the whole tree including this file, and a test that spelled the
labels it forbids would fail the gate it is testing.
"""

from __future__ import annotations

import io
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from tools import check_public_release as checks

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_INIT = ROOT / "src" / "operatebench" / "__init__.py"

#: The forbidden forms, assembled word by word so this file never contains one
#: contiguously and therefore never fails the gate it is testing.
STAGE_SLICE = " ".join(("P1", "executable", "slice"))
STAGE_PASS = " ".join(("Pass", "B"))
STAGE_FOR_THIS = " ".join(("for", "this", "pass"))
FALSE_TOPOLOGY = " ".join(
    ("the", "two", "distributions", "are", "built", "from", "the", "same", "wheel")
)
#: The bare topology claim, on its own, for the source assertion below.
TWO_DISTRIBUTIONS = " ".join(("two", "distributions"))

#: Vocabulary that must keep working. Each is a real sentence shape from this
#: tree: a staged execution, the Boundary Track's policy arms, an ordinary use
#: of the verb, and the honest topology statement.
HONEST_PROSE = (
    "The Lifecycle arms Stage 1 is matched against.",
    "P0 is the habitual arm; P1 never reaches the quote.",
    "A future change that made this pass by widening the reader would fail.",
    "Distribution operatebench installs two importable packages from one wheel.",
    "Phase two of the visit is authored, not inferred.",
    "The suite must pass before the gate reports success.",
)


def _plant(tmp_path: Path, text: str) -> Path:
    tree = tmp_path / "tree"
    (tree / "docs").mkdir(parents=True)
    (tree / "docs" / "note.md").write_text(text, encoding="utf-8")
    return tree


def _wheel_bytes(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return next(dist.glob("*.whl"))


def _sdist_bytes(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--sdist", "--out-dir", str(dist)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return next(dist.glob("*.tar.gz"))


# --------------------------------------------------------------- the source


class TestTheShippedDescription:
    def test_the_package_docstring_carries_no_internal_stage_label(self) -> None:
        text = PACKAGE_INIT.read_text(encoding="utf-8")
        assert STAGE_SLICE not in text
        assert STAGE_PASS not in text
        assert STAGE_FOR_THIS not in text

    def test_the_package_docstring_does_not_claim_two_distributions(self) -> None:
        text = PACKAGE_INIT.read_text(encoding="utf-8")
        assert FALSE_TOPOLOGY not in text
        assert TWO_DISTRIBUTIONS not in text.lower()

    def test_it_states_the_topology_that_is_actually_shipped(self) -> None:
        flowed = " ".join(PACKAGE_INIT.read_text(encoding="utf-8").split())
        assert (
            "Distribution ``operatebench`` installs two importable packages from "
            "one wheel" in flowed
        )
        assert "historical Boundary Track package" in flowed

    def test_it_still_states_the_narrow_primitive_reuse(self) -> None:
        flowed = " ".join(PACKAGE_INIT.read_text(encoding="utf-8").split())
        assert "boundarybench.jsonsafe" in flowed
        assert "canonical UTF-8 JSON" in flowed
        assert "Neither package imports the other's domain semantics." in flowed

    def test_the_whole_candidate_tree_is_clean(self) -> None:
        assert checks.check_public_scope_prose(ROOT) == []

    def test_the_check_is_part_of_the_gate(self) -> None:
        results = checks.run_all(ROOT)
        assert "internal stage labels and shipped topology" in results
        assert results["internal stage labels and shipped topology"] == []


# ------------------------------------------------------------- the gate fails


class TestTheGateFailsClosed:
    @pytest.mark.parametrize(
        ("label", "planted"),
        [
            ("stage slice", f"This package is the {STAGE_SLICE} of the RFC."),
            ("lettered pass", f"Curating the export is {STAGE_PASS}."),
            ("stage-scoped export", f"The export is small {STAGE_FOR_THIS}."),
            ("false topology", f"Note: {FALSE_TOPOLOGY}."),
        ],
    )
    def test_a_planted_label_or_topology_claim_is_caught(
        self, tmp_path: Path, label: str, planted: str
    ) -> None:
        problems = checks.check_public_scope_prose(_plant(tmp_path, planted))
        assert problems, f"{label} was not caught"
        assert all("docs/note.md" in problem for problem in problems)

    @pytest.mark.parametrize("honest", HONEST_PROSE)
    def test_legitimate_behavioural_vocabulary_is_not_a_finding(
        self, tmp_path: Path, honest: str
    ) -> None:
        assert checks.check_public_scope_prose(_plant(tmp_path, honest)) == []

    def test_the_scanner_source_does_not_trigger_on_its_own_rules(self) -> None:
        """The module states every label form it forbids. That is what it is for.

        Asserted two ways: the exemption exists, and it is the *only* thing
        keeping the module clean — so an exemption that quietly widened to a
        directory would fail here rather than pass silently.
        """
        scanner = ROOT / "tools" / "check_public_release.py"
        text = scanner.read_text(encoding="utf-8")
        assert checks.scan_public_scope(text, where="tools/check_public_release.py")
        assert checks.check_public_scope_prose(ROOT) == []

    def test_a_test_file_planting_a_label_would_still_be_caught(
        self, tmp_path: Path
    ) -> None:
        # The exemption is one exact path. A file under tests/ that spelled a
        # forbidden label out gets no relief, which is why this suite builds
        # them from parts.
        tree = tmp_path / "tree"
        (tree / "tests").mkdir(parents=True)
        (tree / "tests" / "test_planted.py").write_text(
            f'LABEL = "{STAGE_PASS}"\n', encoding="utf-8"
        )
        problems = checks.check_public_scope_prose(tree)
        assert any("tests/test_planted.py" in problem for problem in problems)


# ---------------------------------------------------------- the built artefacts


class TestTheBuiltDistributionsCarryTheSameProse:
    def test_the_wheel_ships_the_truthful_description(self, tmp_path: Path) -> None:
        with zipfile.ZipFile(_wheel_bytes(tmp_path)) as wheel:
            member = next(
                name for name in wheel.namelist() if name == "operatebench/__init__.py"
            )
            text = wheel.read(member).decode("utf-8")
        assert STAGE_SLICE not in text
        assert STAGE_PASS not in text
        assert FALSE_TOPOLOGY not in text
        assert "installs two importable packages from one wheel" in " ".join(text.split())
        assert checks.scan_public_scope(text, where="wheel") == []

    def test_no_wheel_member_carries_a_label_or_topology_claim(
        self, tmp_path: Path
    ) -> None:
        problems: list[str] = []
        with zipfile.ZipFile(_wheel_bytes(tmp_path)) as wheel:
            for info in wheel.infolist():
                if info.is_dir():
                    continue
                text = checks._decode_text(wheel.read(info))
                if text is None:
                    continue
                problems.extend(checks.scan_public_scope(text, where=info.filename))
        assert problems == []

    def test_no_sdist_member_carries_a_label_or_topology_claim(
        self, tmp_path: Path
    ) -> None:
        problems: list[str] = []
        with tarfile.open(_sdist_bytes(tmp_path)) as sdist:
            for entry in sdist.getmembers():
                if not entry.isfile():
                    continue
                handle = sdist.extractfile(entry)
                if handle is None:
                    continue
                text = checks._decode_text(handle.read())
                if text is None:
                    continue
                if entry.name.endswith("/tools/check_public_release.py"):
                    # The canonical scanner, which states the rules. Exempt for
                    # the same reason and by the same exact-path rule the tree
                    # walk uses.
                    continue
                problems.extend(checks.scan_public_scope(text, where=entry.name))
        assert problems == []

    def test_the_built_distribution_check_catches_a_planted_label(
        self, tmp_path: Path
    ) -> None:
        """The wheel path of the gate, exercised on a wheel that is not clean."""
        root = tmp_path / "candidate"
        dist = root / "dist"
        dist.mkdir(parents=True)
        with zipfile.ZipFile(dist / "operatebench-0.0.0-py3-none-any.whl", "w") as wheel:
            wheel.writestr(
                "operatebench/__init__.py",
                f'"""This package is the {STAGE_SLICE}."""\n',
            )
        problems = checks.check_built_distributions(root)
        assert any("internal stage label" in problem for problem in problems)

    def test_the_built_distribution_check_catches_a_planted_topology_claim(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "candidate"
        dist = root / "dist"
        dist.mkdir(parents=True)
        payload = f"{FALSE_TOPOLOGY}\n".encode()
        archive = dist / "operatebench-0.0.0.tar.gz"
        with tarfile.open(archive, "w:gz") as sdist:
            info = tarfile.TarInfo("operatebench-0.0.0/README.md")
            info.size = len(payload)
            sdist.addfile(info, io.BytesIO(payload))
        problems = checks.check_built_distributions(root)
        assert any("false distribution topology" in problem for problem in problems)

    def test_the_extracted_sdist_scanner_still_passes_on_this_candidate(
        self, tmp_path: Path
    ) -> None:
        extracted = tmp_path / "extracted"
        extracted.mkdir()
        with tarfile.open(_sdist_bytes(tmp_path)) as sdist:
            sdist.extractall(extracted, filter="data")
        root = next(extracted.iterdir())
        result = subprocess.run(
            [sys.executable, "-m", "tools.check_public_release", "--surface", "sdist"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "internal stage labels and shipped topology" in result.stdout
