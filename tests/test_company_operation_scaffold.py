# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Repository-aware, owner-scoped company scaffold contract."""

from __future__ import annotations

import hashlib
import os
import py_compile
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import operatebench._contribution_structure as structure
import operatebench.sdk.company_scaffold as company
from operatebench.cli import EXIT_ERROR, EXIT_OK, main
from operatebench.sdk import OperationScaffoldError, scaffold_company_operation
from tools.check_contribution_isolation import check_contributions

ROOT = Path(__file__).resolve().parents[1]


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    for relative in (
        "src/operatebench/contributions",
        "src/operatebench/resources",
        "tests/contributions",
        "examples/contributions",
        "docs/contributions",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        ROOT / "src/operatebench/resources/operation_pack_registry.yaml",
        root / "src/operatebench/resources/operation_pack_registry.yaml",
    )
    (root / "src/operatebench/contributions/__init__.py").write_text(
        '"""Static contribution namespaces."""\n', encoding="utf-8"
    )
    (root / "tests/contributions/__init__.py").write_text("", encoding="utf-8")
    return root


def _tracked_repository(tmp_path: Path) -> Path:
    root = tmp_path / "tracked-repository"
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", "src", "tests", "examples", "docs", "tools"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout.split(b"\0")
    for encoded in tracked:
        if not encoded:
            continue
        relative = Path(os.fsdecode(encoded))
        if "__pycache__" in relative.parts or relative.suffix == ".pyc":
            continue
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    return root


def _create(root: Path, operation_name: str = "invoice_review"):
    return scaffold_company_operation(
        repository_root=root,
        owner_id="northstar",
        owner_display_name="Northstar Fictional Company",
        pack_id=f"partner.northstar.{operation_name}.synthetic",
        operation_id=f"northstar_{operation_name}_synthetic_v1",
        operation_name=operation_name,
    )


def _tree_snapshot(root: Path) -> dict[str, bytes | None]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None
        for path in sorted(root.rglob("*"))
        if not path.is_symlink()
    }


def test_public_tree_aggregate_budget_counts_across_directories_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = tmp_path / "public"
    tree.mkdir()
    for dirname in ("a", "b"):
        child = tree / dirname
        child.mkdir()
        (child / "one.txt").write_text("one", encoding="utf-8")
    cache = tree / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-311.pyc").write_bytes(b"cache")
    monkeypatch.setattr(structure, "DIRECTORY_ENTRY_LIMIT", 4)
    monkeypatch.setattr(structure, "PUBLIC_TREE_ENTRY_LIMIT", 6)
    assert len(structure.validate_public_tree(tree, allow_canonical_cache=True)) == 2
    monkeypatch.setattr(structure, "PUBLIC_TREE_ENTRY_LIMIT", 5)
    with pytest.raises(structure.PublicTreeLimitError, match="aggregate"):
        structure.validate_public_tree(tree, allow_canonical_cache=True)


def test_public_tree_depth_budget_exact_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = tmp_path / "public"
    leaf = tree
    for name in ("a", "b"):
        leaf = leaf / name
        leaf.mkdir(parents=True)
    (leaf / "value.txt").write_text("value", encoding="utf-8")
    monkeypatch.setattr(structure, "PUBLIC_TREE_DEPTH_LIMIT", 2)
    assert structure.validate_public_tree(tree) == (leaf / "value.txt",)
    monkeypatch.setattr(structure, "PUBLIC_TREE_DEPTH_LIMIT", 1)
    with pytest.raises(structure.PublicTreeLimitError, match="deep"):
        structure.validate_public_tree(tree)


def test_public_tree_aggregate_failure_does_not_request_an_extra_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = tmp_path / "public"
    tree.mkdir()
    requested = 0

    def traced_iterdir(path: Path):
        nonlocal requested
        assert path == tree
        requested += 1
        yield tree / "limit-plus-one.txt"
        pytest.fail("validator requested an entry after aggregate failure")

    monkeypatch.setattr(Path, "iterdir", traced_iterdir)
    monkeypatch.setattr(structure, "PUBLIC_TREE_ENTRY_LIMIT", 0)
    with pytest.raises(structure.PublicTreeLimitError, match="aggregate"):
        structure.validate_public_tree(tree)
    assert requested == 1


def test_root_replacement_after_render_fails_prewrite_and_closes_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path)
    substitute = _repository(tmp_path / "substitute-parent")
    original = tmp_path / "validated-original"
    before_root = _tree_snapshot(root)
    before_substitute = _tree_snapshot(substitute)
    fd_root = Path("/proc/self/fd")
    before_fds = len(tuple(fd_root.iterdir())) if fd_root.is_dir() else None
    render = company._rendered_files

    def swap_after_render(**kwargs: object) -> dict[str, bytes]:
        rendered = render(**kwargs)
        root.rename(original)
        substitute.rename(root)
        return rendered

    monkeypatch.setattr(company, "_rendered_files", swap_after_render)
    with pytest.raises(OperationScaffoldError, match="repository_root"):
        _create(root)
    assert _tree_snapshot(original) == before_root
    assert _tree_snapshot(root) == before_substitute
    if before_fds is not None:
        assert len(tuple(fd_root.iterdir())) == before_fds


@pytest.mark.parametrize(
    "phase", ["before_first_mkdir", "between_mkdirs", "before_first_file", "final"]
)
def test_root_replacement_during_transaction_never_splits_or_redirects_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    root = _repository(tmp_path)
    substitute = _repository(tmp_path / "substitute-parent")
    original = tmp_path / "validated-original"
    before_root = _tree_snapshot(root)
    before_substitute = _tree_snapshot(substitute)
    swapped = False

    def swap() -> None:
        nonlocal swapped
        root.rename(original)
        substitute.rename(root)
        swapped = True

    real_mkdir = company.os.mkdir
    mkdirs = 0

    def mkdir(path: object, *args: object, **kwargs: object) -> None:
        nonlocal mkdirs
        mkdirs += 1
        if phase == "before_first_mkdir" and mkdirs == 1:
            swap()
        real_mkdir(path, *args, **kwargs)
        if phase == "between_mkdirs" and mkdirs == 1:
            swap()

    real_write = company._write_file
    writes = 0

    def write(parent_fd: int, name: str, content: bytes) -> tuple[int, int]:
        nonlocal writes
        writes += 1
        if phase == "before_first_file" and writes == 1:
            swap()
        return real_write(parent_fd, name, content)

    real_fchmod = company.os.fchmod
    directory_chmods = 0

    def fchmod(fd: int, mode: int) -> None:
        nonlocal directory_chmods
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            directory_chmods += 1
            if phase == "final" and directory_chmods == 7:
                swap()
        real_fchmod(fd, mode)

    monkeypatch.setattr(company.os, "mkdir", mkdir)
    monkeypatch.setattr(company, "_write_file", write)
    monkeypatch.setattr(company.os, "fchmod", fchmod)
    with pytest.raises(OperationScaffoldError, match="repository_root"):
        _create(root)
    assert swapped
    assert _tree_snapshot(original) == before_root
    assert _tree_snapshot(root) == before_substitute


@pytest.mark.parametrize(
    "owner_relative",
    [
        "src/operatebench/contributions/northstar",
        "tests/contributions/northstar",
        "examples/contributions/northstar",
        "docs/contributions/northstar",
    ],
)
@pytest.mark.parametrize("phase", ["after_render", "before_first_write", "final"])
def test_existing_owner_root_replacement_is_anchored_and_rolled_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_relative: str,
    phase: str,
) -> None:
    root = _repository(tmp_path)
    _create(root)
    owner_root = root / owner_relative
    moved = owner_root.with_name(owner_root.name + "-validated-original")
    substitute = tmp_path / "foreign-owner"
    substitute.mkdir()
    sentinel = substitute / "sentinel.bin"
    sentinel.write_bytes(b"foreign owner bytes\n")
    substitute_identity = (substitute.stat().st_dev, substitute.stat().st_ino)
    before_owner = _tree_snapshot(owner_root)
    before_substitute = _tree_snapshot(substitute)
    fd_root = Path("/proc/self/fd")
    before_fds = len(tuple(fd_root.iterdir())) if fd_root.is_dir() else None
    swapped = False

    def swap() -> None:
        nonlocal swapped
        owner_root.rename(moved)
        substitute.rename(owner_root)
        swapped = True

    real_render = company._rendered_files

    def render(**kwargs: object) -> dict[str, bytes]:
        rendered = real_render(**kwargs)
        if phase == "after_render":
            swap()
        return rendered

    real_write = company._write_file

    def write(*args: object, **kwargs: object) -> tuple[int, int]:
        if phase == "before_first_write" and not swapped:
            swap()
        return real_write(*args, **kwargs)  # type: ignore[arg-type]

    real_scan = company._scan_authority

    def scan(*args: object, **kwargs: object) -> bool:
        if phase == "final" and kwargs.get("claim_requested") is False and not swapped:
            swap()
        return real_scan(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(company, "_rendered_files", render)
    monkeypatch.setattr(company, "_write_file", write)
    monkeypatch.setattr(company, "_scan_authority", scan)
    with pytest.raises(OperationScaffoldError):
        _create(root, "account_recovery")
    assert swapped
    assert _tree_snapshot(moved) == before_owner
    assert _tree_snapshot(owner_root) == before_substitute
    assert (owner_root.stat().st_dev, owner_root.stat().st_ino) == substitute_identity
    assert (owner_root / sentinel.name).read_bytes() == b"foreign owner bytes\n"
    for relative in (
        "src/operatebench/contributions/northstar/account_recovery",
        "tests/contributions/northstar/account_recovery",
        "examples/contributions/northstar/account_recovery",
        "docs/contributions/northstar/account_recovery.md",
    ):
        assert not (root / relative).exists()
    if before_fds is not None:
        assert len(tuple(fd_root.iterdir())) == before_fds


@pytest.mark.parametrize("manifest_name", ["OWNER.yaml", "CONTRIBUTION.yaml"])
def test_same_size_authority_change_at_final_is_refused_and_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest_name: str
) -> None:
    root = _repository(tmp_path)
    _create(root)
    owner = root / "src/operatebench/contributions/northstar"
    target = (
        owner / "OWNER.yaml"
        if manifest_name == "OWNER.yaml"
        else owner / "invoice_review/CONTRIBUTION.yaml"
    )
    before = target.read_bytes()
    after = (
        before.replace(b"Northstar", b"Southstar", 1)
        if manifest_name == "OWNER.yaml"
        else before.replace(
            b"northstar_invoice_review_synthetic_v1",
            b"northstar_invoice_review_synthetic_v2",
            1,
        )
    )
    assert len(after) == len(before) and after != before
    fd_root = Path("/proc/self/fd")
    before_fds = len(tuple(fd_root.iterdir())) if fd_root.is_dir() else None
    real_scan = company._scan_authority
    mutated = False

    def scan(*args: object, **kwargs: object) -> bool:
        nonlocal mutated
        if kwargs.get("claim_requested") is False and not mutated:
            target.write_bytes(after)
            mutated = True
        return real_scan(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(company, "_scan_authority", scan)
    with pytest.raises(OperationScaffoldError):
        _create(root, "account_recovery")
    assert mutated and target.read_bytes() == after
    for relative in (
        "src/operatebench/contributions/northstar/account_recovery",
        "tests/contributions/northstar/account_recovery",
        "examples/contributions/northstar/account_recovery",
        "docs/contributions/northstar/account_recovery.md",
    ):
        assert not (root / relative).exists()
    if before_fds is not None:
        assert len(tuple(fd_root.iterdir())) == before_fds


def test_entry_enumeration_stops_on_exact_cap_plus_one() -> None:
    yielded = 0

    class TracedPath:
        def iterdir(self):
            nonlocal yielded
            for index in range(4098):
                if index == 4097:
                    pytest.fail("enumerator requested an item after the cap-plus-one")
                yielded += 1
                yield Path(f"entry-{index:04d}")

        def __str__(self) -> str:
            return "traced-directory"

    with pytest.raises(OperationScaffoldError, match="too many entries"):
        company._entries(TracedPath(), "traced directory")  # type: ignore[arg-type]
    assert yielded == 4097


def test_checker_entry_enumeration_stops_on_same_exact_cap_plus_one() -> None:
    from tools import check_contribution_isolation as isolation

    yielded = 0

    class TracedPath:
        def iterdir(self):
            nonlocal yielded
            for index in range(4098):
                if index == 4097:
                    pytest.fail("checker requested an item after the cap-plus-one")
                yielded += 1
                yield Path(f"entry-{index:04d}")

        def __str__(self) -> str:
            return "traced-directory"

    with pytest.raises(isolation.ContributionError, match="too many entries"):
        isolation._enumerate(TracedPath(), "traced directory")  # type: ignore[arg-type]
    assert yielded == 4097


def test_literal_tree_is_iterative_bounded_and_cycle_safe() -> None:
    deeply_nested: object = None
    for _ in range(2000):
        deeply_nested = [deeply_nested]
    assert company._is_literal_tree(deeply_nested)

    cycle: list[object] = []
    cycle.append(cycle)
    assert not company._is_literal_tree(cycle)
    assert not company._is_literal_tree([None] * company._LITERAL_NODE_LIMIT)


def test_fresh_owner_creates_exact_repository_tree_and_metadata(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    result = _create(root)
    assert result.created_paths == (
        "docs/contributions/northstar/invoice_review.md",
        "examples/contributions/northstar/invoice_review/operation.yaml",
        "src/operatebench/contributions/northstar/OWNER.yaml",
        "src/operatebench/contributions/northstar/__init__.py",
        "src/operatebench/contributions/northstar/invoice_review/CONTRIBUTION.yaml",
        "src/operatebench/contributions/northstar/invoice_review/README.md",
        "src/operatebench/contributions/northstar/invoice_review/__init__.py",
        "src/operatebench/contributions/northstar/invoice_review/agents.py",
        "src/operatebench/contributions/northstar/invoice_review/evaluator.py",
        "src/operatebench/contributions/northstar/invoice_review/negative_control_oracle.yaml",
        "src/operatebench/contributions/northstar/invoice_review/operation.py",
        "src/operatebench/contributions/northstar/invoice_review/pack.py",
        "src/operatebench/contributions/northstar/invoice_review/spec.py",
        "tests/contributions/northstar/__init__.py",
        "tests/contributions/northstar/invoice_review/__init__.py",
        "tests/contributions/northstar/invoice_review/test_pack.py",
    )
    assert result.reused_owner_declaration is None
    assert not result.registered and not result.runnable and not result.evidence_eligible
    manifest = yaml.safe_load(
        (
            root
            / "src/operatebench/contributions/northstar/invoice_review/CONTRIBUTION.yaml"
        ).read_text()
    )
    assert manifest == {
        "schema_version": 1,
        "owner_id": "northstar",
        "pack_id": "partner.northstar.invoice_review.synthetic",
        "operation_id": "northstar_invoice_review_synthetic_v1",
        "operation_type": "partner.northstar.invoice_review.synthetic",
        "aliases": [],
        "module": "operatebench.contributions.northstar.invoice_review",
        "source_path": "src/operatebench/contributions/northstar/invoice_review",
        "test_path": "tests/contributions/northstar/invoice_review/test_pack.py",
        "fixture_path": "examples/contributions/northstar/invoice_review/operation.yaml",
        "documentation_path": "docs/contributions/northstar/invoice_review.md",
    }


def test_exact_existing_owner_is_reused_byte_for_byte(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _create(root)
    owner = root / "src/operatebench/contributions/northstar/OWNER.yaml"
    before = owner.read_bytes()
    result = _create(root, "account_recovery")
    assert owner.read_bytes() == before
    assert result.reused_owner_declaration == (
        "src/operatebench/contributions/northstar/OWNER.yaml"
    )


def test_real_source_layout_cli_tolerates_shared_caches_across_operations(
    tmp_path: Path,
) -> None:
    root = _tracked_repository(tmp_path)
    environment = os.environ.copy()
    environment.pop("PYTHONDONTWRITEBYTECODE", None)
    environment.pop("PYTHONPYCACHEPREFIX", None)
    environment["PYTHONPATH"] = str(root / "src")
    compiled_test_init = subprocess.run(
        [
            sys.executable,
            "-c",
            "import py_compile; "
            "py_compile.compile('tests/contributions/__init__.py', doraise=True)",
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert compiled_test_init.returncode == 0, compiled_test_init.stderr

    def invoke(operation_name: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "operatebench.cli",
                "init-company-operation",
                "--repository-root",
                str(root),
                "--owner-id",
                "northstar",
                "--owner-display-name",
                "Northstar Fictional Company",
                "--operation-name",
                operation_name,
                "--pack-id",
                f"partner.northstar.{operation_name}.synthetic",
                "--operation-id",
                f"northstar_{operation_name}_synthetic_v1",
                "--json",
            ],
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

    first = invoke("invoice_review")
    assert first.returncode == EXIT_OK, first.stderr
    first_result = yaml.safe_load(first.stdout)
    assert len(first_result["created_paths"]) == 16
    for relative in (
        "src/operatebench/contributions/__pycache__",
        "tests/contributions/__pycache__",
    ):
        assert (root / relative).is_dir()

    second = invoke("account_recovery")
    assert second.returncode == EXIT_OK, second.stderr
    second_result = yaml.safe_load(second.stdout)
    assert len(second_result["created_paths"]) == 13
    assert check_contributions(root) == (
        "northstar:northstar_account_recovery_synthetic_v1",
        "northstar:northstar_invoice_review_synthetic_v1",
        "prospire:lettings_property_compliance_synthetic_v1",
    )


@pytest.mark.parametrize(
    "relative",
    [
        "src/operatebench/contributions",
        "tests/contributions",
        "examples/contributions",
        "docs/contributions",
    ],
)
@pytest.mark.parametrize("replacement", ["missing", "symlink"])
def test_canonical_root_open_failures_are_public_sdk_errors(
    tmp_path: Path, relative: str, replacement: str
) -> None:
    root = _repository(tmp_path)
    canonical = root / relative
    shutil.rmtree(canonical)
    if replacement == "symlink":
        target = tmp_path / f"replacement-{relative.replace('/', '-')}"
        target.mkdir()
        canonical.symlink_to(target, target_is_directory=True)
    with pytest.raises(
        OperationScaffoldError, match="canonical repository root"
    ) as caught:
        _create(root)
    assert isinstance(caught.value.__cause__, OSError)


def test_generated_test_packages_prevent_duplicate_basename_collection(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    _create(root)
    scaffold_company_operation(
        repository_root=root,
        owner_id="riverside",
        owner_display_name="Riverside Fictional Company",
        pack_id="partner.riverside.account_review.synthetic",
        operation_id="riverside_account_review_synthetic_v1",
        operation_name="account_review",
    )
    (root / "src/operatebench/__init__.py").write_text(
        "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n",
        encoding="utf-8",
    )
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/contributions", "--collect-only", "-q"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": f"{root / 'src'}:{ROOT / 'src'}"},
    )
    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert "import file mismatch" not in collected.stdout + collected.stderr


def test_repository_root_rejects_symlink_before_dotdot_without_writes(
    tmp_path: Path,
) -> None:
    first = _repository(tmp_path / "a")
    second = _repository(tmp_path / "b")
    (first / "link").symlink_to(second / "docs", target_is_directory=True)
    before_first = _tree_snapshot(first)
    before_second = _tree_snapshot(second)
    with pytest.raises(OperationScaffoldError, match="lexical/realpath divergence"):
        _create(first / "link" / "..")
    assert _tree_snapshot(first) == before_first
    assert _tree_snapshot(second) == before_second


def test_final_verification_rejects_external_hardlink_and_retains_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path)
    outside = tmp_path / "outside-link"
    real_write = company._write_file
    linked = False

    def write_and_link(parent_fd: int, name: str, content: bytes) -> tuple[int, int]:
        nonlocal linked
        identity = real_write(parent_fd, name, content)
        if not linked:
            parent = Path(f"/proc/self/fd/{parent_fd}").resolve(strict=True)
            os.link(parent / name, outside)
            linked = True
        return identity

    monkeypatch.setattr(company, "_write_file", write_and_link)
    with pytest.raises(OperationScaffoldError, match="hardlinked"):
        _create(root)
    assert linked and outside.is_file() and outside.stat().st_nlink == 2


@pytest.mark.parametrize(
    ("relative", "body"),
    [
        (
            "tests/contributions/northstar/invoice_review/test_replay.py",
            '"""Extra test."""\n',
        ),
        (
            "examples/contributions/northstar/invoice_review/extra_fixture.yaml",
            "synthetic: true\n",
        ),
        ("docs/contributions/northstar/nested/notes.md", "Notes.\n"),
    ],
)
def test_checker_accepted_public_extras_do_not_block_scaffolding(
    tmp_path: Path, relative: str, body: str
) -> None:
    root = _repository(tmp_path)
    _create(root)
    extra = root / relative
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text(body, encoding="utf-8")
    assert check_contributions(root)
    scaffold_company_operation(
        repository_root=root,
        owner_id="riverside",
        owner_display_name="Riverside Fictional Company",
        pack_id="partner.riverside.account_review.synthetic",
        operation_id="riverside_account_review_synthetic_v1",
        operation_name="account_review",
    )


@pytest.mark.parametrize(
    ("relative", "shape", "accepted"),
    [
        ("examples/contributions/README.md", "file", False),
        ("docs/contributions/README.md", "file", False),
        ("examples/contributions/northstar/invoice_review/rows.csv", "file", False),
        ("docs/contributions/northstar/diagram.png", "file", False),
        (
            "examples/contributions/northstar/invoice_review/nested/notes.txt",
            "file",
            True,
        ),
        ("docs/contributions/northstar/nested/data.json", "file", True),
        ("docs/contributions/northstar/.hidden.md", "file", False),
        ("docs/contributions/northstar/link.md", "symlink", False),
        ("docs/contributions/northstar/pipe.txt", "fifo", False),
        ("docs/contributions/northstar/__pycache__/junk.pyc", "file", False),
        ("docs/contributions/__pycache__/module.cpython-311.pyc", "file", False),
        ("examples/contributions/__pycache__/module.cpython-311.pyc", "file", False),
    ],
)
def test_public_tree_policy_has_zero_scaffold_checker_divergence(
    tmp_path: Path, relative: str, shape: str, accepted: bool
) -> None:
    root = _repository(tmp_path)
    _create(root)
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if shape == "symlink":
        target.symlink_to(root / "docs/contributions/northstar/invoice_review.md")
    elif shape == "fifo":
        os.mkfifo(target)
    else:
        target.write_text("finite beta text resource\n", encoding="utf-8")

    checker_accepted = True
    try:
        check_contributions(root)
    except Exception:
        checker_accepted = False
    scaffold_accepted = True
    try:
        _create(root, "account_recovery")
    except OperationScaffoldError:
        scaffold_accepted = False
    assert checker_accepted == scaffold_accepted == accepted


def test_repeat_and_partial_state_fail_without_byte_changes(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _create(root)
    before = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).digest()
        for p in root.rglob("*")
        if p.is_file()
    }
    with pytest.raises(OperationScaffoldError, match=r"already exists|conflict|claimed"):
        _create(root)
    after = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).digest()
        for p in root.rglob("*")
        if p.is_file()
    }
    assert after == before


def test_partial_target_directory_is_refused_before_write(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    partial = root / "tests/contributions/northstar"
    partial.mkdir()
    sentinel = partial / "keep.txt"
    sentinel.write_text("pre-existing", encoding="utf-8")
    with pytest.raises(OperationScaffoldError, match=r"partial|ambiguous"):
        _create(root)
    assert sentinel.read_text(encoding="utf-8") == "pre-existing"
    assert not (root / "src/operatebench/contributions/northstar").exists()


def test_registered_and_unregistered_cross_kind_collision_precedes_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path)
    registry_path = root / "src/operatebench/resources/operation_pack_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["operations"][1]["aliases"].append("northstar_registered_collision_v1")
    registry["operations"][1]["aliases"].sort()
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    writes = 0

    def forbidden(*args: object, **kwargs: object) -> None:
        nonlocal writes
        writes += 1
        pytest.fail("collision must be resolved in PRE-WRITE")

    monkeypatch.setattr("operatebench.sdk.company_scaffold._write_file", forbidden)
    with pytest.raises(
        OperationScaffoldError,
        match=r"operation identity .* is alias .* and operation_id",
    ):
        scaffold_company_operation(
            repository_root=root,
            owner_id="northstar",
            owner_display_name="Northstar Fictional Company",
            pack_id="partner.northstar.sample.synthetic",
            operation_id="northstar_registered_collision_v1",
            operation_name="sample",
        )
    assert writes == 0


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("owner_display_name", "Different Company"),
        ("contribution_kind", "maintainer"),
        ("code_license", "MIT"),
        ("content_license", "Apache-2.0"),
    ],
)
def test_existing_owner_authority_mismatch_is_prewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str,
) -> None:
    root = _repository(tmp_path)
    _create(root)
    owner_path = root / "src/operatebench/contributions/northstar/OWNER.yaml"
    owner = yaml.safe_load(owner_path.read_text(encoding="utf-8"))
    owner[field] = replacement
    owner_path.write_text(yaml.safe_dump(owner, sort_keys=False), encoding="utf-8")
    before = owner_path.read_bytes()
    monkeypatch.setattr(
        "operatebench.sdk.company_scaffold._write_file",
        lambda *args, **kwargs: pytest.fail("authority mismatch must be PRE-WRITE"),
    )
    with pytest.raises(OperationScaffoldError, match="does not exactly match"):
        _create(root, "account_recovery")
    assert owner_path.read_bytes() == before


def test_duplicate_key_registry_and_missing_templates_are_prewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path)
    registry = root / "src/operatebench/resources/operation_pack_registry.yaml"
    registry.write_text("schema_version: 1\nschema_version: 1\noperations: []\n")
    monkeypatch.setattr(
        "operatebench.sdk.company_scaffold._write_file",
        lambda *args, **kwargs: pytest.fail("manifest failure must be PRE-WRITE"),
    )
    with pytest.raises(OperationScaffoldError, match="duplicate manifest key"):
        _create(root)
    assert not (root / "src/operatebench/contributions/northstar").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_module", "x" * 300, "bounded text"),
        ("source_module", "operatebench.domains.a\x01b.pack", "unsafe text"),
        ("source_module", "operatebench.domains.a\x7fb.pack", "unsafe text"),
        ("source_path", "x" * 300, "bounded text"),
        ("source_path", "src/operatebench/domains/a\x01b", "unsafe text"),
        ("source_path", "src/operatebench/domains/a\x7fb", "unsafe text"),
    ],
)
def test_reserved_source_text_contract_rejects_in_checker_and_scaffold_prewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    root = _repository(tmp_path)
    registry_path = root / "src/operatebench/resources/operation_pack_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["operations"][0][field] = value
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    with pytest.raises(Exception, match=message):
        check_contributions(root)
    monkeypatch.setattr(
        company, "_write_file", lambda *a, **k: pytest.fail("must reject pre-write")
    )
    monkeypatch.setattr(
        company, "files", lambda *a, **k: pytest.fail("must reject before templates")
    )
    with pytest.raises(OperationScaffoldError, match=message):
        _create(root)


def test_unsupported_secure_fd_boundary_precedes_resources_and_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path)
    monkeypatch.setattr(
        "operatebench.sdk.company_scaffold._supports_secure_directory_fds",
        lambda: False,
    )
    monkeypatch.setattr(
        "operatebench.sdk.company_scaffold.files",
        lambda *args: pytest.fail("unsupported platform must precede resources"),
    )
    with pytest.raises(OperationScaffoldError, match="directory-fd"):
        _create(root)
    assert not (root / "src/operatebench/contributions/northstar").exists()


def test_injected_write_failure_rolls_back_new_owner_across_all_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path)
    import operatebench.sdk.company_scaffold as company

    real = company._write_file
    calls = 0

    def fail_second(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected")
        return real(*args, **kwargs)

    monkeypatch.setattr(company, "_write_file", fail_second)
    with pytest.raises(OSError, match="injected"):
        _create(root)
    assert not (root / "src/operatebench/contributions/northstar").exists()
    assert not (root / "tests/contributions/northstar").exists()
    assert not (root / "examples/contributions/northstar").exists()
    assert not (root / "docs/contributions/northstar").exists()


@pytest.mark.parametrize(
    "mutation", ["before_child_open", "after_child_open", "parent_path"]
)
def test_directory_substitution_never_deletes_foreign_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root = _repository(tmp_path)
    fd_root = Path("/proc/self/fd")
    before_fds = len(list(fd_root.iterdir())) if fd_root.is_dir() else None
    public = (
        root / "docs/contributions"
        if mutation == "parent_path"
        else root / "docs/contributions/northstar"
    )
    moved = public.with_name(public.name + "-invocation-created-moved")
    foreign = b"foreign replacement sentinel\n"
    replacement_identity: list[tuple[int, int]] = []
    real_open = company.os.open
    swapped = False

    def swap() -> None:
        nonlocal swapped
        public.rename(moved)
        public.mkdir()
        replacement_identity.append((public.stat().st_dev, public.stat().st_ino))
        if mutation == "parent_path":
            (public / "sentinel.txt").write_bytes(foreign)
        swapped = True

    def opening(path: object, flags: int, *args: object, **kwargs: object) -> int:
        dir_fd = kwargs.get("dir_fd")
        if (
            mutation == "before_child_open"
            and not swapped
            and path == "northstar"
            and dir_fd
        ):
            parent = Path(f"/proc/self/fd/{dir_fd}").resolve(strict=True)
            if parent == root / "docs/contributions":
                swap()
        return real_open(path, flags, *args, **kwargs)

    real_write = company._write_file
    writes = 0

    def fail_later(*args: object, **kwargs: object):
        nonlocal writes
        writes += 1
        if mutation in {"after_child_open", "parent_path"} and not swapped:
            swap()
        if writes == 2:
            raise OSError("original later failure")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(company.os, "open", opening)
    monkeypatch.setattr(company, "_write_file", fail_later)
    expected = {
        "before_child_open": "descriptor proof",
        "after_child_open": "original later failure",
        "parent_path": "validated repository directory changed",
    }[mutation]
    with pytest.raises((OSError, OperationScaffoldError), match=expected):
        _create(root)
    assert swapped
    assert public.is_dir()
    assert (public.stat().st_dev, public.stat().st_ino) == replacement_identity[0]
    if mutation == "parent_path":
        assert (public / "sentinel.txt").read_bytes() == foreign
    assert moved.is_dir()
    if before_fds is not None:
        assert len(list(fd_root.iterdir())) == before_fds


def test_partner_operation_id_must_begin_with_owner_prefix_prewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path)
    monkeypatch.setattr(
        company, "_write_file", lambda *a, **k: pytest.fail("must reject pre-write")
    )
    with pytest.raises(OperationScaffoldError, match="operation_id must start"):
        scaffold_company_operation(
            repository_root=root,
            owner_id="northstar",
            owner_display_name="Northstar Fictional Company",
            pack_id="partner.northstar.invoice.synthetic",
            operation_id="riverside_invoice_synthetic_v1",
            operation_name="invoice_review",
        )


def test_initial_created_file_fstat_failure_is_in_rollback_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path)
    import operatebench.sdk.company_scaffold as company

    real_fstat = company.os.fstat
    tripped = False
    fd_root = Path("/proc/self/fd")
    before_fds = len(list(fd_root.iterdir())) if fd_root.is_dir() else None

    def failing_fstat(fd: int):
        nonlocal tripped
        try:
            target = (fd_root / str(fd)).resolve(strict=True)
        except OSError:
            return real_fstat(fd)
        if (
            not tripped
            and root in target.parents
            and target.is_file()
            and target.name not in {"operation_pack_registry.yaml", "__init__.py"}
        ):
            tripped = True
            raise OSError("injected initial file fstat failure")
        return real_fstat(fd)

    monkeypatch.setattr(company.os, "fstat", failing_fstat)
    with pytest.raises(OSError, match="initial file fstat"):
        _create(root)
    assert tripped
    for relative in (
        "src/operatebench/contributions/northstar",
        "tests/contributions/northstar",
        "examples/contributions/northstar",
        "docs/contributions/northstar",
    ):
        assert not (root / relative).exists()
    if before_fds is not None:
        assert len(list(fd_root.iterdir())) == before_fds


@pytest.mark.parametrize(
    "stage", ["initial_fstat", "write", "flush", "fsync", "fchmod", "final_fstat"]
)
def test_created_file_stage_failures_rollback_and_preserve_existing_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    root = _repository(tmp_path)
    _create(root)
    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    import operatebench.sdk.company_scaffold as company

    fd_root = Path("/proc/self/fd")
    before_fds = len(list(fd_root.iterdir())) if fd_root.is_dir() else None
    real_fstat = company.os.fstat
    file_fstats = 0
    tripped = False
    expected_created_file = (
        root / "docs/contributions/northstar/account_recovery.md"
    ).resolve()
    descriptor_events: list[str] = []
    failure_fstat: int | None = None
    events_at_failure: tuple[str, ...] | None = None

    def generated_file(fd: int) -> bool:
        try:
            target = (fd_root / str(fd)).resolve(strict=True)
        except OSError:
            return False
        return target == expected_created_file and target.is_file()

    if stage in {"initial_fstat", "final_fstat"}:
        real_fdopen = company.os.fdopen
        real_fsync = company.os.fsync
        real_fchmod = company.os.fchmod

        class TracingHandle:
            def __init__(self, handle):
                self.handle = handle

            def __enter__(self):
                self.handle.__enter__()
                return self

            def __exit__(self, *args: object):
                return self.handle.__exit__(*args)

            def write(self, content: bytes):
                descriptor_events.append("write")
                return self.handle.write(content)

            def flush(self):
                descriptor_events.append("flush")
                return self.handle.flush()

        def tracing_fdopen(fd: int, *args: object, **kwargs: object):
            handle = real_fdopen(fd, *args, **kwargs)
            return TracingHandle(handle) if generated_file(fd) else handle

        def tracing_fsync(fd: int):
            if generated_file(fd):
                descriptor_events.append("fsync")
            return real_fsync(fd)

        def tracing_fchmod(fd: int, mode: int):
            if generated_file(fd):
                descriptor_events.append("fchmod")
            return real_fchmod(fd, mode)

        def failing_fstat(fd: int):
            nonlocal file_fstats, tripped, failure_fstat, events_at_failure
            if generated_file(fd):
                file_fstats += 1
                descriptor_events.append(f"fstat:{file_fstats}")
                expected = 1 if stage == "initial_fstat" else 2
                if file_fstats == expected:
                    tripped = True
                    failure_fstat = file_fstats
                    events_at_failure = tuple(descriptor_events)
                    raise OSError(f"injected {stage}")
            return real_fstat(fd)

        monkeypatch.setattr(company.os, "fdopen", tracing_fdopen)
        monkeypatch.setattr(company.os, "fsync", tracing_fsync)
        monkeypatch.setattr(company.os, "fchmod", tracing_fchmod)
        monkeypatch.setattr(company.os, "fstat", failing_fstat)
    elif stage in {"write", "flush"}:
        real_fdopen = company.os.fdopen

        class FailingHandle:
            def __init__(self, handle):
                self.handle = handle

            def __enter__(self):
                self.handle.__enter__()
                return self

            def __exit__(self, *args: object):
                return self.handle.__exit__(*args)

            def write(self, content: bytes):
                nonlocal tripped
                if stage == "write" and not tripped:
                    tripped = True
                    raise OSError("injected write")
                return self.handle.write(content)

            def flush(self):
                nonlocal tripped
                if stage == "flush" and not tripped:
                    tripped = True
                    raise OSError("injected flush")
                return self.handle.flush()

        def failing_fdopen(*args: object, **kwargs: object):
            return FailingHandle(real_fdopen(*args, **kwargs))

        monkeypatch.setattr(company.os, "fdopen", failing_fdopen)
    else:
        operation = getattr(company.os, stage)

        def failing_operation(fd: int, *args: object):
            nonlocal tripped
            if generated_file(fd) and not tripped:
                tripped = True
                raise OSError(f"injected {stage}")
            return operation(fd, *args)

        monkeypatch.setattr(company.os, stage, failing_operation)

    with pytest.raises(OSError, match=f"injected {stage}"):
        _create(root, "account_recovery")
    assert tripped
    if stage == "initial_fstat":
        assert failure_fstat == 1
        assert events_at_failure == ("fstat:1",)
    elif stage == "final_fstat":
        assert failure_fstat == 2
        assert events_at_failure == (
            "fstat:1",
            "write",
            "flush",
            "fsync",
            "fchmod",
            "fstat:2",
        )
    after = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before
    for relative in (
        "src/operatebench/contributions/northstar/account_recovery",
        "tests/contributions/northstar/account_recovery",
        "examples/contributions/northstar/account_recovery",
        "docs/contributions/northstar/account_recovery.md",
    ):
        assert not (root / relative).exists()
    if before_fds is not None:
        assert len(list(fd_root.iterdir())) == before_fds


def test_persistent_initial_fstat_blindness_closes_fd_without_unproven_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path)
    import operatebench.sdk.company_scaffold as company

    fd_root = Path("/proc/self/fd")
    before_fds = len(list(fd_root.iterdir())) if fd_root.is_dir() else None
    real_fstat = company.os.fstat

    def persistently_failing_fstat(fd: int):
        try:
            target = (fd_root / str(fd)).resolve(strict=True)
        except OSError:
            return real_fstat(fd)
        if (
            root in target.parents
            and target.is_file()
            and target.name not in {"operation_pack_registry.yaml", "__init__.py"}
        ):
            raise OSError("persistent descriptor blindness")
        return real_fstat(fd)

    monkeypatch.setattr(company.os, "fstat", persistently_failing_fstat)
    with pytest.raises(OSError, match="persistent descriptor blindness"):
        _create(root)
    # Ownership cannot be established, so the exclusive-create entry is retained.
    assert (root / "docs/contributions/northstar/invoice_review.md").is_file()
    if before_fds is not None:
        assert len(list(fd_root.iterdir())) == before_fds


def test_generated_tree_passes_static_contribution_checker(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _create(root)
    assert check_contributions(root) == (
        "northstar:northstar_invoice_review_synthetic_v1",
    )
    for path in (root / "src/operatebench/contributions/northstar").rglob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_cli_json_is_deterministic_relative_and_has_no_destination_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _repository(tmp_path)
    argv = [
        "init-company-operation",
        "--repository-root",
        str(root),
        "--owner-id",
        "northstar",
        "--owner-display-name",
        "Northstar Fictional Company",
        "--operation-name",
        "invoice_review",
        "--pack-id",
        "partner.northstar.invoice_review.synthetic",
        "--operation-id",
        "northstar_invoice_review_synthetic_v1",
        "--json",
    ]
    assert main(argv) == EXIT_OK
    output = capsys.readouterr().out
    assert str(root) not in output
    assert '"registered": false' in output
    assert "src/operatebench/contributions/northstar/OWNER.yaml" in output

    assert main([*argv[:-1], "--destination", "elsewhere"]) != EXIT_OK
    assert "Traceback" not in capsys.readouterr().err


def test_cli_domain_error_has_no_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _repository(tmp_path)
    code = main(
        [
            "init-company-operation",
            "--repository-root",
            str(root),
            "--owner-id",
            "northstar",
            "--owner-display-name",
            "Northstar",
            "--operation-name",
            "bad-name",
            "--pack-id",
            "partner.northstar.x",
            "--operation-id",
            "northstar_x_v1",
        ]
    )
    assert code == EXIT_ERROR
    captured = capsys.readouterr()
    assert "operation_name" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("component", "accepted"),
    [
        ("a", True),
        ("a" * 63, True),
        ("a_b", True),
        ("a__b", True),
        ("a_", True),
        ("_a", False),
        ("1a", False),
        ("A", False),
        ("é", False),
        ("a" * 64, False),
        ("class", False),
        ("con", False),
    ],
)
def test_physical_component_matches_trusted_checker_boundary(
    tmp_path: Path, component: str, accepted: bool
) -> None:
    root = _repository(tmp_path)

    def call():
        return scaffold_company_operation(
            repository_root=root,
            owner_id="northstar",
            owner_display_name="Northstar Fictional Company",
            pack_id="partner.northstar.invoice.synthetic",
            operation_id="northstar_invoice_synthetic_v1",
            operation_name=component,
        )

    if not accepted:
        with pytest.raises(OperationScaffoldError, match="operation_name"):
            call()
        assert not (root / "src/operatebench/contributions/northstar").exists()
        return
    call()
    assert check_contributions(root) == ("northstar:northstar_invoice_synthetic_v1",)


def test_existing_four_root_foreign_entry_is_pre_resource_and_pre_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path)
    _create(root)
    foreign = root / "tests/contributions/northstar/foreign.txt"
    foreign.write_bytes(b"foreign\x00sentinel")
    monkeypatch.setattr(
        "operatebench.sdk.company_scaffold.files",
        lambda *args: pytest.fail("layout failure must precede template access"),
    )
    with pytest.raises(OperationScaffoldError, match=r"test root|foreign|allowed"):
        _create(root, "second")
    assert foreign.read_bytes() == b"foreign\x00sentinel"
    assert not (root / "src/operatebench/contributions/northstar/second").exists()


@pytest.mark.parametrize(
    "relative",
    [
        "src/operatebench/contributions/northstar/__init__.py",
        "tests/contributions/northstar/conftest.py",
    ],
)
def test_existing_python_support_is_bounded_before_templates_and_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    root = _repository(tmp_path)
    _create(root)
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x" * (1024 * 1024 + 1))
    before = target.read_bytes()
    real_read_bytes = Path.read_bytes

    def bounded_only(path: Path) -> bytes:
        if path == target:
            pytest.fail("existing support file used unbounded Path.read_bytes")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", bounded_only)
    monkeypatch.setattr(
        company,
        "files",
        lambda *args: pytest.fail("support rejection must precede template access"),
    )
    monkeypatch.setattr(
        company,
        "_write_file",
        lambda *args, **kwargs: pytest.fail("support rejection must precede writes"),
    )
    with pytest.raises(OperationScaffoldError, match=r"large|bounded|size"):
        _create(root, "second")
    assert real_read_bytes(target) == before
    assert not (root / "src/operatebench/contributions/northstar/second").exists()


def test_bounded_stable_reader_rejects_non_regular_and_oversize_inputs(
    tmp_path: Path,
) -> None:
    regular = tmp_path / "regular"
    regular.write_bytes(b"12345")
    with pytest.raises(OperationScaffoldError, match=r"large|size"):
        company._bounded_stable_regular_file(regular, 4, "test input")

    symlink = tmp_path / "symlink"
    try:
        symlink.symlink_to(regular)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(OperationScaffoldError, match=r"non-symlink|open"):
        company._bounded_stable_regular_file(symlink, 8, "test input")

    fifo = tmp_path / "special"
    fifo.mkdir()
    with pytest.raises(OperationScaffoldError, match="regular"):
        company._bounded_stable_regular_file(fifo, 8, "test input")


@pytest.mark.parametrize("name", ["OWNER.yaml", "CONTRIBUTION.yaml"])
@pytest.mark.parametrize("reader", ["scaffold", "checker"])
def test_untrusted_manifest_fifo_is_rejected_without_hanging(
    tmp_path: Path, name: str, reader: str
) -> None:
    fifo = tmp_path / name
    os.mkfifo(fifo)
    if reader == "scaffold":
        source = (
            "from pathlib import Path\n"
            "from operatebench.sdk.company_scaffold import _bounded_stable_regular_file\n"
            f"_bounded_stable_regular_file(Path({str(fifo)!r}), 16384, 'manifest')\n"
        )
    else:
        source = (
            "from pathlib import Path\n"
            "from tools.check_contribution_isolation import _read_bounded\n"
            f"_read_bounded(Path({str(fifo)!r}), 16384)\n"
        )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        timeout=2,
        check=False,
    )
    assert completed.returncode != 0
    assert b"regular file" in completed.stderr


def test_bounded_stable_reader_rejects_same_size_torn_read_and_closes_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "mutable.py"
    target.write_bytes(b"first")
    fd_root = Path("/proc/self/fd")
    before_fds = len(list(fd_root.iterdir())) if fd_root.is_dir() else None
    real_lseek = os.lseek
    mutated = False

    def mutate_before_second_pass(fd: int, offset: int, whence: int) -> int:
        nonlocal mutated
        result = real_lseek(fd, offset, whence)
        if not mutated and offset == 0 and whence == os.SEEK_SET:
            mutated = True
            target.write_bytes(b"other")
        return result

    monkeypatch.setattr(company.os, "lseek", mutate_before_second_pass)
    with pytest.raises(OperationScaffoldError, match="changed during stable read"):
        company._bounded_stable_regular_file(target, 5, "test input")
    assert target.read_bytes() == b"other"
    if before_fds is not None:
        assert len(list(fd_root.iterdir())) == before_fds


def test_scaffold_bounded_reader_uses_three_passes_and_fixed_quiescence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "three-passes.py"
    target.write_bytes(b"first")
    real_fstat = os.fstat
    real_read = os.read
    real_lseek = os.lseek
    snapshots = 0
    reads = 0
    seeks = 0
    sleeps: list[float] = []

    def traced_fstat(fd: int) -> os.stat_result:
        nonlocal snapshots
        snapshots += 1
        return real_fstat(fd)

    def traced_read(fd: int, count: int) -> bytes:
        nonlocal reads
        reads += 1
        return real_read(fd, count)

    def traced_lseek(fd: int, offset: int, whence: int) -> int:
        nonlocal seeks
        seeks += 1
        return real_lseek(fd, offset, whence)

    monkeypatch.setattr(company.os, "fstat", traced_fstat)
    monkeypatch.setattr(company.os, "read", traced_read)
    monkeypatch.setattr(company.os, "lseek", traced_lseek)
    monkeypatch.setattr(company.time, "sleep", sleeps.append)

    assert company._bounded_stable_regular_file(target, 5, "test input") == b"first"
    assert snapshots == 6
    assert reads == 3
    assert seeks == 2
    assert sleeps == [company._READ_QUIESCENCE_SECONDS] * 2


def test_scaffold_bounded_reader_rejects_third_pass_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "third-pass.py"
    target.write_bytes(b"first")
    real_read = os.read
    real_lseek = os.lseek
    pass_number = 1

    def seek(fd: int, offset: int, whence: int) -> int:
        nonlocal pass_number
        pass_number += 1
        return real_lseek(fd, offset, whence)

    def replaced_read(fd: int, count: int) -> bytes:
        if pass_number == 3:
            return b"other"[:count]
        return real_read(fd, count)

    monkeypatch.setattr(company.os, "lseek", seek)
    monkeypatch.setattr(company.os, "read", replaced_read)
    monkeypatch.setattr(company.time, "sleep", lambda _: None)
    with pytest.raises(OperationScaffoldError, match="changed during stable read"):
        company._bounded_stable_regular_file(target, 5, "test input")


def test_generated_repository_test_reaches_todo_without_collection_error(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    shutil.copytree(
        ROOT / "src/operatebench",
        root / "src/operatebench",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("contributions", "__pycache__", "*.pyc"),
    )
    _create(root)
    test = root / "tests/contributions/northstar/invoice_review/test_pack.py"
    environment = dict(os.environ, PYTHONPATH=str(root / "src"))
    run = subprocess.run(
        [sys.executable, "-m", "pytest", str(test), "-q"],
        cwd=root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    assert run.returncode == 1
    assert "1 failed" in run.stdout
    assert "error" not in run.stdout.casefold()
    assert "TODO: replace this" in run.stdout


def test_partner_display_collision_is_pre_resource_and_preserves_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path)
    scaffold_company_operation(
        repository_root=root,
        owner_id="alpha_co",
        owner_display_name="Example Partner Ltd",
        pack_id="partner.alpha_co.one.synthetic",
        operation_id="alpha_co_one_synthetic_v1",
        operation_name="one",
    )
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(
        "operatebench.sdk.company_scaffold.files",
        lambda *args: pytest.fail("display collision must precede template access"),
    )
    with pytest.raises(OperationScaffoldError, match="display name"):
        scaffold_company_operation(
            repository_root=root,
            owner_id="beta_co",
            owner_display_name="EXAMPLE PARTNER LTD",
            pack_id="partner.beta_co.two.synthetic",
            operation_id="beta_co_two_synthetic_v1",
            operation_name="two",
        )
    assert {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    } == before


def test_canonical_owner_cache_is_shared_and_malformed_cache_is_prewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path)
    _create(root)
    init = root / "src/operatebench/contributions/northstar/__init__.py"
    py_compile.compile(str(init), doraise=True)
    _create(root, "second")
    cache = init.parent / "__pycache__"
    (cache / "foreign.txt").write_text("sentinel", encoding="utf-8")
    monkeypatch.setattr(
        "operatebench.sdk.company_scaffold.files",
        lambda *args: pytest.fail("malformed cache must precede templates"),
    )
    with pytest.raises(OperationScaffoldError, match="cache"):
        _create(root, "third")


@pytest.mark.parametrize("kind", ["src/operatebench", "tests"])
@pytest.mark.parametrize("shape", ["malformed", "noncanonical", "nested", "symlink"])
def test_shared_source_and_test_roots_reject_invalid_caches(
    tmp_path: Path, kind: str, shape: str
) -> None:
    root = _repository(tmp_path)
    cache = root / kind / "contributions/__pycache__"
    if shape == "symlink":
        target = tmp_path / f"cache-target-{kind.replace('/', '-')}"
        target.mkdir()
        cache.symlink_to(target, target_is_directory=True)
    else:
        cache.mkdir()
        if shape == "malformed":
            (cache / "foreign.txt").write_text("foreign", encoding="utf-8")
        elif shape == "noncanonical":
            (cache / "module.pyc").write_bytes(b"cache")
        else:
            nested = cache / "nested"
            nested.mkdir()
            (nested / "module.cpython-311.pyc").write_bytes(b"cache")
    with pytest.raises(OperationScaffoldError, match=r"cache|shared"):
        _create(root)


@pytest.mark.parametrize(
    "relative",
    ["examples/contributions/__pycache__", "docs/contributions/__pycache__"],
)
def test_shared_examples_and_docs_reject_even_canonical_caches(
    tmp_path: Path, relative: str
) -> None:
    root = _repository(tmp_path)
    cache = root / relative
    cache.mkdir()
    (cache / "module.cpython-311.pyc").write_bytes(b"cache")
    with pytest.raises(OperationScaffoldError, match=r"cache|shared"):
        _create(root)


def test_balanced_placeholder_display_round_trips_without_reentrant_render(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    display = "Acme {{CONTRIBUTION_KIND}} Ltd"
    scaffold_company_operation(
        repository_root=root,
        owner_id="northstar",
        owner_display_name=display,
        pack_id="partner.northstar.one.synthetic",
        operation_id="northstar_one_synthetic_v1",
        operation_name="one",
    )
    assert (
        display
        in (root / "src/operatebench/contributions/northstar/OWNER.yaml").read_text()
    )
    assert (
        display
        in (root / "src/operatebench/contributions/northstar/one/pack.py").read_text()
    )
    assert check_contributions(root) == ("northstar:northstar_one_synthetic_v1",)


@pytest.mark.parametrize("display", ["Acme {{ Ltd", "Acme }} Ltd"])
def test_unmatched_template_delimiter_display_is_pre_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, display: str
) -> None:
    root = _repository(tmp_path)
    monkeypatch.setattr(
        "operatebench.sdk.company_scaffold.files",
        lambda *args: pytest.fail("display validation must precede templates"),
    )
    with pytest.raises(OperationScaffoldError, match="owner_display_name"):
        scaffold_company_operation(
            repository_root=root,
            owner_id="northstar",
            owner_display_name=display,
            pack_id="partner.northstar.one.synthetic",
            operation_id="northstar_one_synthetic_v1",
            operation_name="one",
        )


@pytest.mark.parametrize(
    "document",
    [
        "schema_version: &version 1\noperations: []\n",
        "schema_version: 1\noperations: &rows []\n",
        "schema_version: 1\noperations: !!seq []\n",
    ],
)
def test_registry_rejects_yaml_graph_and_explicit_tags_pre_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, document: str
) -> None:
    root = _repository(tmp_path)
    (root / "src/operatebench/resources/operation_pack_registry.yaml").write_text(
        document
    )
    monkeypatch.setattr(
        "operatebench.sdk.company_scaffold.files",
        lambda *args: pytest.fail("YAML literal failure must precede templates"),
    )
    with pytest.raises(OperationScaffoldError, match=r"anchor|alias|tag|literal"):
        _create(root)


@pytest.mark.parametrize("collision", ["owner", "pack"])
def test_reserved_registry_rejects_inconsistent_owner_and_duplicate_pack_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, collision: str
) -> None:
    root = _repository(tmp_path)
    registry_path = root / "src/operatebench/resources/operation_pack_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    row = dict(registry["operations"][0])
    row["aliases"] = list(row["aliases"])
    if collision == "owner":
        row.update(
            owner_display_name="Inconsistent Reserved Display",
            pack_id="reserved.inconsistent.synthetic",
            operation_type="reserved.inconsistent.synthetic",
            operation_id="reserved_inconsistent_v1",
            aliases=[],
            source_module="operatebench.contributions.prospire.inconsistent.pack",
            source_path="src/operatebench/contributions/prospire/inconsistent",
        )
    else:
        row["operation_id"] = "divergent_duplicate_pack_v1"
    registry["operations"].append(row)
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(
        "operatebench.sdk.company_scaffold.files",
        lambda *args: pytest.fail("registry ambiguity must precede templates"),
    )
    with pytest.raises(OperationScaffoldError, match=r"owner authority|pack row"):
        _create(root)


def test_rollback_parent_domain_failure_preserves_original_and_continues_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path)
    import operatebench.sdk.company_scaffold as company

    real_write = company._write_file
    real_stat = company.os.stat
    writes = 0
    failed = False
    rollback_blinded = False

    def fail_second(*args: object, **kwargs: object):
        nonlocal writes, failed
        writes += 1
        if writes == 2:
            failed = True
            raise OSError("original write failure")
        return real_write(*args, **kwargs)

    def blind_one_parent(path: object, *args: object, **kwargs: object):
        nonlocal rollback_blinded
        if failed and not rollback_blinded and kwargs.get("dir_fd") is not None:
            rollback_blinded = True
            raise OSError("rollback parent blind")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(company, "_write_file", fail_second)
    monkeypatch.setattr(company.os, "stat", blind_one_parent)
    with pytest.raises(OSError, match="original write failure"):
        _create(root)
    assert rollback_blinded
    assert not (root / "src/operatebench/contributions/northstar").exists()
    assert not (root / "tests/contributions/northstar").exists()
    assert not (root / "examples/contributions/northstar").exists()
