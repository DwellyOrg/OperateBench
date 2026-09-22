"""The scaffold is deterministic, inert and safe around existing paths."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from operatebench.sdk import OperationScaffoldError, scaffold_operation

PACK_ID = "commerce.return_refund.synthetic"
OPERATION_ID = "commerce_return_refund_synthetic_v1"


def _create(destination: Path) -> tuple[Path, ...]:
    return scaffold_operation(
        pack_id=PACK_ID,
        operation_id=OPERATION_ID,
        owner_id="prospire",
        owner_display_name="PROSPIRE TECHNOLOGIES LTD",
        contribution_kind="maintainer",
        destination=destination,
    )


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.iterdir())
        if path.is_file()
    }


def test_scaffold_has_exact_flat_tree_and_compiling_python(tmp_path: Path) -> None:
    destination = tmp_path / "return_refund"
    paths = _create(destination)
    assert {path.name for path in paths} == {
        "README.md",
        "__init__.py",
        "agents.py",
        "evaluator.py",
        "negative_control_oracle.yaml",
        "operation.py",
        "operation.yaml",
        "pack.py",
        "spec.py",
        "test_pack.py",
    }
    for path in destination.glob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    payload = yaml.safe_load((destination / "operation.yaml").read_text())
    assert payload["operation_type"] == PACK_ID


def test_same_identity_produces_byte_identical_tree(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _create(left)
    _create(right)
    assert _tree(left) == _tree(right)


def test_two_fictional_partner_owners_are_path_and_byte_isolated(tmp_path: Path) -> None:
    northstar = tmp_path / "northstar" / "sample"
    riverside = tmp_path / "riverside" / "sample"
    northstar.parent.mkdir()
    riverside.parent.mkdir()
    for destination, owner, display in (
        (northstar, "northstar", "Northstar Fictional Company"),
        (riverside, "riverside", "Riverside Fictional Company"),
    ):
        scaffold_operation(
            pack_id=f"partner.{owner}.sample",
            operation_id=f"{owner}_sample_v1",
            owner_id=owner,
            owner_display_name=display,
            contribution_kind="partner",
            destination=destination,
        )
    before = {
        path.relative_to(riverside).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in riverside.iterdir()
        if path.is_file()
    }
    (northstar / "README.md").write_text("changed only northstar\n", encoding="utf-8")
    after = {
        path.relative_to(riverside).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in riverside.iterdir()
        if path.is_file()
    }
    assert before == after
    assert set(northstar.iterdir()).isdisjoint(riverside.iterdir())
    for destination in (northstar, riverside):
        result = subprocess.run(
            ["pytest", "-q", str(destination / "test_pack.py")],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 1
        assert "TODO: replace this" in result.stdout
        assert "passed" not in result.stdout


def test_partner_scaffold_is_inert_and_never_edits_central_registry(
    tmp_path: Path,
) -> None:
    registry = Path("src/operatebench/sdk/builtins.py")
    before = hashlib.sha256(registry.read_bytes()).hexdigest()
    destination = tmp_path / "northstar_sample"
    scaffold_operation(
        pack_id="partner.northstar.sample",
        operation_id="northstar_sample_v1",
        owner_id="northstar",
        owner_display_name="Northstar Fictional Company",
        contribution_kind="partner",
        destination=destination,
    )
    assert hashlib.sha256(registry.read_bytes()).hexdigest() == before
    readme = (destination / "README.md").read_text(encoding="utf-8")
    assert "unregistered incubator scaffold" in readme
    assert "Do not edit `operatebench/sdk/builtins.py`" in readme
    assert "does not make" in readme and "model evidence" in readme
    pack = (destination / "pack.py").read_text(encoding="utf-8")
    assert 'default_spec="examples/contributions/northstar/sample/operation.yaml"' in pack


def test_yaml_like_ids_remain_text(tmp_path: Path) -> None:
    destination = tmp_path / "yaml_scalars"
    scaffold_operation(
        pack_id="null",
        operation_id="true",
        owner_id="prospire",
        owner_display_name="PROSPIRE TECHNOLOGIES LTD",
        contribution_kind="maintainer",
        destination=destination,
    )
    fixture = yaml.safe_load((destination / "operation.yaml").read_text())
    oracle = yaml.safe_load((destination / "negative_control_oracle.yaml").read_text())
    assert fixture["operation_id"] == "true"
    assert fixture["operation_type"] == "null"
    assert oracle["operation_type"] == "null"


def test_existing_destination_is_never_changed(tmp_path: Path) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("mine", encoding="utf-8")
    with pytest.raises(OperationScaffoldError, match="already exists"):
        _create(destination)
    assert sentinel.read_text(encoding="utf-8") == "mine"


def test_destination_must_be_an_importable_package_name(tmp_path: Path) -> None:
    destination = tmp_path / "return-refund"
    with pytest.raises(OperationScaffoldError, match="Python package"):
        _create(destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "pack_id",
    ("../escape", "/absolute", ".", "ReturnRefund", "réfund.synthetic", "a" * 129),
)
def test_unsafe_pack_ids_are_refused_before_write(tmp_path: Path, pack_id: str) -> None:
    destination = tmp_path / "target"
    with pytest.raises(OperationScaffoldError):
        scaffold_operation(
            pack_id=pack_id,
            operation_id=OPERATION_ID,
            owner_id="prospire",
            owner_display_name="PROSPIRE TECHNOLOGIES LTD",
            contribution_kind="maintainer",
            destination=destination,
        )
    assert not destination.exists()


def test_cross_kind_identity_collision_is_refused_before_resources_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "absent" / "sample"
    identity = "partner.northstar.sample"

    def fail_if_resources_are_read(_package: str) -> None:
        pytest.fail("identity collision must win before template resource lookup")

    monkeypatch.setattr("operatebench.sdk.scaffold.files", fail_if_resources_are_read)
    with pytest.raises(OperationScaffoldError, match="must be distinct"):
        scaffold_operation(
            pack_id=identity,
            operation_id=identity,
            owner_id="northstar",
            owner_display_name="Northstar Fictional Company",
            contribution_kind="partner",
            destination=destination,
        )
    assert not destination.parent.exists()


def test_del_in_owner_display_name_is_refused_before_write(tmp_path: Path) -> None:
    destination = tmp_path / "target"
    with pytest.raises(OperationScaffoldError, match="safe text"):
        scaffold_operation(
            pack_id=PACK_ID,
            operation_id=OPERATION_ID,
            owner_id="prospire",
            owner_display_name="PROSPIRE\x7f TECHNOLOGIES LTD",
            contribution_kind="maintainer",
            destination=destination,
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    "value", ["unsafe\u202ename", "unsafe\u2067name", "zero\u200bwidth"]
)
def test_unicode_format_owner_display_is_refused_before_write(
    tmp_path: Path, value: str
) -> None:
    destination = tmp_path / "target"
    with pytest.raises(OperationScaffoldError, match="safe text"):
        scaffold_operation(
            pack_id=PACK_ID,
            operation_id=OPERATION_ID,
            owner_id="prospire",
            owner_display_name=value,
            contribution_kind="maintainer",
            destination=destination,
        )
    assert not destination.exists()


def test_ordinary_international_owner_display_is_allowed(tmp_path: Path) -> None:
    destination = tmp_path / "target"
    display = "Étoile 技術株式会社 № ٢"
    scaffold_operation(
        pack_id=PACK_ID,
        operation_id=OPERATION_ID,
        owner_id="prospire",
        owner_display_name=display,
        contribution_kind="maintainer",
        destination=destination,
    )
    assert destination.is_dir()
    assert display in (destination / "pack.py").read_text()


def test_non_nfc_owner_display_is_refused_before_write(tmp_path: Path) -> None:
    destination = tmp_path / "target"
    with pytest.raises(OperationScaffoldError, match="safe text"):
        scaffold_operation(
            pack_id=PACK_ID,
            operation_id=OPERATION_ID,
            owner_id="prospire",
            owner_display_name="E\u0301toile",
            contribution_kind="maintainer",
            destination=destination,
        )
    assert not destination.exists()


def test_symlink_ancestor_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(OperationScaffoldError, match="symlink"):
        _create(link / "target")
    assert not (real / "target").exists()


def test_dangling_symlink_ancestor_is_refused_as_a_symlink(tmp_path: Path) -> None:
    link = tmp_path / "missing-link"
    try:
        link.symlink_to(tmp_path / "absent", target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(OperationScaffoldError, match="symlink"):
        _create(link / "target")


def test_target_symlink_replacement_cannot_redirect_generated_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "target"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_open = os.open
    replaced = False

    def replacing_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if (
            not replaced
            and os.fspath(path) == destination.name
            and dir_fd is not None
            and flags & os.O_DIRECTORY
        ):
            replaced = True
            destination.rmdir()
            try:
                destination.symlink_to(outside, target_is_directory=True)
            except OSError:
                pytest.skip("symlinks unavailable")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replacing_open)
    with pytest.raises(OperationScaffoldError, match="replacement is refused"):
        _create(destination)
    assert destination.is_symlink()
    assert not tuple(outside.iterdir())


def test_target_replacement_after_open_stays_bound_to_original_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "target"
    moved = tmp_path / "moved_target"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_open = os.open
    replaced = False

    def replacing_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if (
            not replaced
            and os.fspath(path) == "README.md"
            and dir_fd is not None
            and flags & os.O_EXCL
        ):
            replaced = True
            destination.rename(moved)
            try:
                destination.symlink_to(outside, target_is_directory=True)
            except OSError:
                pytest.skip("symlinks unavailable")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replacing_open)
    with pytest.raises(OperationScaffoldError, match="was replaced"):
        _create(destination)
    assert destination.is_symlink()
    assert not tuple(outside.iterdir())
    assert moved.is_dir()
    assert not tuple(moved.iterdir())


def test_ancestor_replacement_is_detected_after_anchored_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container = tmp_path / "container"
    parent = container / "parent"
    parent.mkdir(parents=True)
    moved = tmp_path / "moved"
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = parent / "target"
    real_mkdir = os.mkdir
    replaced = False

    def replacing_mkdir(
        path: str | os.PathLike[str],
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal replaced
        if not replaced and os.fspath(path) == destination.name and dir_fd is not None:
            replaced = True
            container.rename(moved)
            try:
                container.symlink_to(outside, target_is_directory=True)
            except OSError:
                pytest.skip("symlinks unavailable")
        real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", replacing_mkdir)
    with pytest.raises(OperationScaffoldError, match=r"symlinks.*refused"):
        _create(destination)
    assert container.is_symlink()
    assert not (outside / "target").exists()
    assert not (moved / "parent" / "target").exists()


def test_platform_without_secure_directory_fds_is_refused_before_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "target"
    monkeypatch.setattr(
        "operatebench.sdk.scaffold._supports_secure_directory_fds",
        lambda: False,
    )
    with pytest.raises(OperationScaffoldError, match="directory-descriptor"):
        _create(destination)
    assert not destination.exists()


def test_racing_file_is_preserved_when_partial_scaffold_is_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "target"
    contested = destination / "__init__.py"
    real_open = os.open

    def racing_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if os.fspath(path) == contested.name and dir_fd is not None and flags & os.O_EXCL:
            contested.write_text("belongs to another writer\n", encoding="utf-8")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", racing_open)
    with pytest.raises(OperationScaffoldError, match="appeared"):
        _create(destination)
    assert contested.read_text(encoding="utf-8") == "belongs to another writer\n"
    assert set(destination.iterdir()) == {contested}


def test_parent_must_exist_and_no_partial_target_is_left(tmp_path: Path) -> None:
    destination = tmp_path / "absent" / "target"
    with pytest.raises(OperationScaffoldError, match="parent directory"):
        _create(destination)
    assert not destination.exists()


def test_second_invocation_does_not_mutate_first_tree(tmp_path: Path) -> None:
    destination = tmp_path / "target"
    _create(destination)
    before = _tree(destination)
    with pytest.raises(OperationScaffoldError):
        _create(destination)
    assert _tree(destination) == before
