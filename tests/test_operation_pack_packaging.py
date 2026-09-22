"""SDK and scaffold resources survive installation in the wheel package."""

from __future__ import annotations

import subprocess
import zipfile
from importlib.resources import files
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sdk_public_surface_imports() -> None:
    from operatebench import sdk

    assert sdk.PACK_API_VERSION == 2
    assert callable(sdk.scaffold_operation)


def test_every_scaffold_template_is_packaged() -> None:
    root = files("operatebench.resources.operation_pack_template")
    names = {entry.name for entry in root.iterdir()}
    assert {
        "README.md.tmpl",
        "__init__.py.tmpl",
        "agents.py.tmpl",
        "evaluator.py.tmpl",
        "negative_control_oracle.yaml.tmpl",
        "operation.py.tmpl",
        "operation.yaml.tmpl",
        "pack.py.tmpl",
        "spec.py.tmpl",
        "test_pack.py.tmpl",
    } <= names


def test_built_wheel_contains_scaffold_resources(tmp_path: Path) -> None:
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    [wheel] = tmp_path.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    template_root = "operatebench/resources/operation_pack_template"
    expected = {
        f"{template_root}/{name}"
        for name in (
            "README.md.tmpl",
            "__init__.py.tmpl",
            "agents.py.tmpl",
            "evaluator.py.tmpl",
            "negative_control_oracle.yaml.tmpl",
            "operation.py.tmpl",
            "operation.yaml.tmpl",
            "pack.py.tmpl",
            "spec.py.tmpl",
            "test_pack.py.tmpl",
        )
    }
    assert expected <= names
    assert "operatebench/resources/operation_pack_registry.yaml" in names
    assert "operatebench/contributions/prospire/OWNER.yaml" in names
    assert (
        "operatebench/contributions/prospire/property_compliance/negative_control_oracle.yaml"
        in names
    )
    assert not any("domains/lettings/property_compliance" in name for name in names)
