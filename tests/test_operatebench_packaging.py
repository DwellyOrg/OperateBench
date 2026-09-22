"""Dependency direction, packaging and Boundary Track compatibility.

Three things are guarded here that no behavioural test would catch.

**Dependency direction.** Core must not learn what a quote is. The check is a
source scan rather than a convention: an import of a domain pack from anywhere
under ``operatebench.core`` fails the suite.

**Both packages ship.** ``boundarybench`` is the Boundary Track and stays
exactly where it was; ``operatebench`` is added beside it. The wheel has to
carry both, and both console scripts have to exist, or one track silently stops
being installable.

**The negative-control oracle ships as package data.** The acceptance gate reads
its expectations from a manifest inside the package, so a wheel that dropped the
YAML would leave ``check-maintenance`` unable to grade anything — a packaging
mistake that no in-tree test run would ever notice.
"""

from __future__ import annotations

import ast
import subprocess
import tarfile
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
FIXTURE = REPO_ROOT / "examples" / "operatebench" / "maintenance_v0_1.yaml"
CORE = REPO_ROOT / "src" / "operatebench" / "core"


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


class TestDependencyDirection:
    def test_core_imports_no_domain_pack(self) -> None:
        offenders: dict[str, set[str]] = {}
        for path in sorted(CORE.glob("*.py")):
            domain_imports = {
                module
                for module in _imported_modules(path)
                if module.startswith("operatebench.domains")
            }
            if domain_imports:
                offenders[path.name] = domain_imports
        assert offenders == {}

    def test_core_imports_only_core_and_the_shared_json_primitive(self) -> None:
        # Stronger than "no domain imports": Core may reach sideways only for the
        # one serialisation primitive both tracks share.
        allowed_outside = {"boundarybench.jsonsafe"}
        for path in sorted(CORE.glob("*.py")):
            for module in _imported_modules(path):
                if module.startswith("operatebench"):
                    assert module.startswith("operatebench.core") or module in {
                        "operatebench.jsonsafe",
                        "operatebench.version",
                    }, f"{path.name} imports {module}"
                elif module.startswith("boundarybench"):
                    assert module in allowed_outside, f"{path.name} imports {module}"

    def test_the_domain_pack_implements_the_core_protocols(self) -> None:
        from operatebench.core.protocol import OperationAgent, OperationDomain
        from operatebench.domains.lettings.maintenance.agents import ReferenceAgent
        from operatebench.domains.lettings.maintenance.operation import (
            MaintenanceOperation,
        )
        from operatebench.domains.lettings.maintenance.spec import load_spec

        spec = load_spec(FIXTURE)
        domain: OperationDomain = MaintenanceOperation(spec, "V1")
        agent: OperationAgent = ReferenceAgent()
        assert domain.build_plan().starts_at
        assert agent.agent_id == "reference"


class TestPackaging:
    def test_both_console_scripts_are_declared(self) -> None:
        scripts = _pyproject()["project"]["scripts"]
        assert scripts["boundarybench"] == "boundarybench.cli:main"
        assert scripts["operatebench"] == "operatebench.cli:main"

    def test_the_wheel_carries_both_packages(self) -> None:
        packages = _pyproject()["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
        assert set(packages) == {"src/boundarybench", "src/operatebench"}

    def test_the_shipped_fixture_is_packaged_with_the_examples(self) -> None:
        assert FIXTURE.exists()

    def test_the_negative_control_oracle_lives_under_a_wheel_package(self) -> None:
        # `packages = ["src/operatebench", ...]` carries every file under those
        # directories, data included. That is why the manifest is inside the
        # package rather than in examples/, which the wheel does not carry.
        from operatebench.domains.lettings.maintenance.oracle import (
            oracle_manifest_path,
        )

        packages = _pyproject()["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
        path = oracle_manifest_path().resolve()
        assert path.is_file()
        assert any(
            (REPO_ROOT / package).resolve() in path.parents for package in packages
        ), path

    def test_the_sdist_carries_the_oracle_with_the_source(self) -> None:
        include = _pyproject()["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
        assert "/src" in include

    def test_the_actual_sdist_excludes_the_test_only_matched_grammar(
        self, tmp_path: Path
    ) -> None:
        subprocess.run(
            ["uv", "build", "--sdist", "--out-dir", str(tmp_path)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        [sdist] = tmp_path.glob("*.tar.gz")
        with tarfile.open(sdist) as archive:
            members = {member.name for member in archive.getmembers()}
        assert not any(
            member.endswith("/tests/matched_control_fixtures.py") for member in members
        )

    def test_the_oracle_is_readable_through_the_package_loader(self) -> None:
        # Read the way an installed wheel would, not by walking the repository.
        from operatebench.domains.lettings.maintenance.oracle import (
            negative_control_oracle,
        )

        oracle = negative_control_oracle()
        assert len(oracle.controls) == 8
        assert oracle.oracle_digest_sha256

    def test_both_entry_points_are_importable(self) -> None:
        import boundarybench.cli as boundary_cli
        import operatebench.cli as operate_cli

        assert callable(boundary_cli.main)
        assert callable(operate_cli.main)


class TestBoundaryTrackUnaffected:
    def test_projection_is_operatebench_owned_and_boundary_is_an_exact_reexport(
        self,
    ) -> None:
        import boundarybench.freezing as boundary_freezing
        import operatebench.jsonsafe as operatebench_jsonsafe

        assert boundary_freezing.to_json is operatebench_jsonsafe.to_json
        assert boundary_freezing.to_json.__module__ == "operatebench.jsonsafe"
        assert (
            boundary_freezing.to_json_preserving_tuples
            is operatebench_jsonsafe.to_json_preserving_tuples
        )
        assert (
            boundary_freezing.to_json_preserving_tuples.__module__
            == "operatebench.jsonsafe"
        )

    def test_the_boundary_public_api_still_exports_what_it_did(self) -> None:
        import boundarybench

        for name in (
            "compile_cube",
            "load_card",
            "evaluate",
            "check_solvers",
            "check_suite",
            "__version__",
        ):
            assert hasattr(boundarybench, name)

    def test_the_boundary_cli_still_validates_its_own_example(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from boundarybench.cli import main as boundary_main

        card = REPO_ROOT / "examples" / "maintenance_authority.yaml"
        code = boundary_main(["validate", str(card)])
        assert code == 0
        assert "OK" in capsys.readouterr().out

    def test_core_reuses_the_operatebench_owned_json_safety_primitive(self) -> None:
        # Stated rather than duplicated: canonical UTF-8 JSON is one primitive,
        # and Core reaches its OperateBench owner rather than a historical shim.
        import operatebench.core.ledger as ledger

        imports = _imported_modules(Path(ledger.__file__))
        assert "operatebench.jsonsafe" in imports
        assert not any(module.startswith("boundarybench") for module in imports)
