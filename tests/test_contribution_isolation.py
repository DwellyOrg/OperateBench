# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed static contribution ownership checks."""

from __future__ import annotations

import os
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.check_contribution_isolation as isolation
from tools.check_contribution_isolation import ContributionError, check_contributions


def test_manifest_literal_tree_checker_is_iterative_bounded_and_cycle_safe() -> None:
    deeply_nested: object = None
    for _ in range(2000):
        deeply_nested = [deeply_nested]
    assert isolation._is_literal_tree(deeply_nested)

    cycle: list[object] = []
    cycle.append(cycle)
    assert not isolation._is_literal_tree(cycle)
    assert not isolation._is_literal_tree([None] * isolation.LITERAL_NODE_LIMIT)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _git_repository(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "tests@example.invalid")
    _git(root, "config", "user.name", "Isolation Tests")


@pytest.mark.parametrize(
    "relative",
    [
        "src/operatebench/contributions/acme/__pycache__/module.cpython-311.pyc",
        "tests/contributions/acme/module.pyo",
        "tests/contributions/acme/__pycache__/README.txt",
    ],
)
def test_tracked_contribution_artifacts_are_rejected(
    tmp_path: Path, relative: str
) -> None:
    _git_repository(tmp_path)
    artifact = tmp_path / relative
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"tracked artifact")
    _git(tmp_path, "add", "-f", "--", relative)

    with pytest.raises(ContributionError, match="tracked contribution artifact"):
        isolation.check_tracked_contribution_artifacts(tmp_path)


def test_clean_tracked_contribution_source_is_accepted(tmp_path: Path) -> None:
    _git_repository(tmp_path)
    source = tmp_path / "src/operatebench/contributions/acme/module.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n")
    _git(tmp_path, "add", "--", source.relative_to(tmp_path).as_posix())

    isolation.check_tracked_contribution_artifacts(tmp_path)


def test_non_repository_fails_closed_with_domain_error(tmp_path: Path) -> None:
    with pytest.raises(ContributionError, match="Git tracked-path discovery failed"):
        isolation.check_tracked_contribution_artifacts(tmp_path)


def test_git_discovery_uses_argv_nul_output_timeout_and_no_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: tuple[list[str], dict[str, object]] | None = None
    real_popen = subprocess.Popen

    def record(command: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
        nonlocal observed
        observed = (command, kwargs)
        return real_popen(command, **kwargs)

    _git_repository(tmp_path)
    monkeypatch.setattr(subprocess, "Popen", record)
    isolation.check_tracked_contribution_artifacts(tmp_path)

    assert observed is not None
    command, kwargs = observed
    assert command[:4] == ["git", "ls-files", "-z", "--"]
    assert "shell" not in kwargs
    assert kwargs["stdout"] == subprocess.PIPE
    assert kwargs["stderr"] == subprocess.PIPE
    assert "capture_output" not in kwargs


def test_untracked_canonical_cache_remains_accepted_by_library(tmp_path: Path) -> None:
    source = _owner(tmp_path, "northstar")
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-311.pyc").write_bytes(b"opaque bytecode")

    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


def test_missing_git_is_a_domain_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "Popen", fail)
    with pytest.raises(ContributionError, match="Git executable is unavailable"):
        isolation.check_tracked_contribution_artifacts(tmp_path)


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ((128, b"", b"fatal \xff"), "Git tracked-path discovery failed: fatal �"),
        ((0, b"bad\xff\x00", b""), "non-UTF-8"),
        ((0, b"unterminated", b""), "malformed"),
    ],
)
def test_tracked_path_invalid_results_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    result: tuple[int, bytes, bytes],
    message: str,
) -> None:
    monkeypatch.setattr(isolation, "_run_git_bounded", lambda *args, **kwargs: result)
    with pytest.raises(ContributionError, match=message):
        isolation.check_tracked_contribution_artifacts(tmp_path)


def _python_bounded(tmp_path: Path, source: str) -> tuple[int, bytes, bytes]:
    return isolation._run_git_bounded([sys.executable, "-c", source], tmp_path)


def test_git_stdout_exact_limit_is_accepted_if_valid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prefix = b"tests/contributions/"
    shortest = prefix + b"a\0"
    repeated = (isolation.GIT_OUTPUT_LIMIT - len(shortest)) // len(shortest)
    remainder = isolation.GIT_OUTPUT_LIMIT - repeated * len(shortest)
    final = prefix + b"a" * (remainder - len(prefix) - 1) + b"\0"
    output = shortest * repeated + final
    assert len(output) == isolation.GIT_OUTPUT_LIMIT

    monkeypatch.setattr(
        isolation, "_run_git_bounded", lambda *args, **kwargs: (0, output, b"")
    )
    isolation.check_tracked_contribution_artifacts(tmp_path)


def test_git_stdout_overflow_stops_a_live_child_and_reaps_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    children: list[subprocess.Popen[bytes]] = []
    real_popen = subprocess.Popen

    def capture(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", capture)
    started = time.monotonic()
    with pytest.raises(ContributionError, match="output exceeded 1 MiB"):
        _python_bounded(
            tmp_path,
            "import os,time; os.write(1, b'x' * 1048577); time.sleep(30)",
        )
    assert time.monotonic() - started < 2
    assert children[0].poll() is not None
    assert children[0].stdout is not None and children[0].stdout.closed
    assert children[0].stderr is not None and children[0].stderr.closed


def test_git_stderr_overflow_is_bounded(tmp_path: Path) -> None:
    with pytest.raises(ContributionError, match="stderr exceeded 64 KiB"):
        _python_bounded(
            tmp_path,
            f"import os; os.write(2, b'e' * {isolation.GIT_STDERR_LIMIT + 1})",
        )


def test_git_hanging_with_open_pipes_times_out_and_is_reaped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    children: list[subprocess.Popen[bytes]] = []
    real_popen = subprocess.Popen
    monkeypatch.setattr(isolation, "GIT_TIMEOUT_SECONDS", 0.1)

    def capture(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", capture)
    with pytest.raises(ContributionError, match="timed out"):
        _python_bounded(tmp_path, "import time; time.sleep(30)")
    assert children[0].poll() is not None


def test_git_simultaneous_stdout_and_stderr_does_not_deadlock(tmp_path: Path) -> None:
    size = isolation.GIT_STDERR_LIMIT
    returncode, stdout, stderr = _python_bounded(
        tmp_path,
        "import os,threading; size="
        f"{size}; t=threading.Thread(target=lambda: os.write(2,b'e'*size)); "
        "t.start(); os.write(1,b'o'*size); t.join()",
    )
    assert returncode == 0
    assert len(stdout) == len(stderr) == size


def test_git_termination_escalates_to_kill_and_closes_pipes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    children: list[subprocess.Popen[bytes]] = []
    real_popen = subprocess.Popen

    def capture(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", capture)
    source = (
        "import os,signal,time; "
        "signal.signal(signal.SIGTERM, lambda *_: None); "
        "os.write(1, b'x' * 1048577); time.sleep(30)"
    )
    started = time.monotonic()
    with pytest.raises(ContributionError, match="output exceeded"):
        _python_bounded(tmp_path, source)
    assert time.monotonic() - started >= isolation.GIT_STOP_GRACE_SECONDS
    assert time.monotonic() - started < 2
    assert children[0].returncode == -signal.SIGKILL
    assert children[0].stdout is not None and children[0].stdout.closed
    assert children[0].stderr is not None and children[0].stderr.closed


def _owner(
    root: Path,
    owner: str,
    operation: str = "sample",
    operation_id: str | None = None,
    *,
    display_name: str | None = None,
    pack_id: str | None = None,
    operation_type: str | None = None,
    aliases: tuple[str, ...] = (),
    contribution_kind: str = "partner",
) -> Path:
    reserved = root / "src/operatebench/resources/operation_pack_registry.yaml"
    if not reserved.exists():
        reserved.parent.mkdir(parents=True, exist_ok=True)
        reserved.write_text(
            Path("src/operatebench/resources/operation_pack_registry.yaml").read_text()
        )
    source = root / f"src/operatebench/contributions/{owner}/{operation}"
    test = root / f"tests/contributions/{owner}/{operation}"
    fixture = root / f"examples/contributions/{owner}/{operation}"
    docs = root / f"docs/contributions/{owner}"
    for path in (source, test, fixture, docs):
        path.mkdir(parents=True, exist_ok=True)
    contribution_root = root / "src/operatebench/contributions"
    (contribution_root / "__init__.py").write_text('"""Contribution package."""\n')
    (root / "tests/contributions/__init__.py").write_text("")
    display = display_name or f"{owner.title()} Fictional Company"
    pack = pack_id or f"partner.{owner}.{operation}"
    operation_type = operation_type or f"{owner}.{operation}.synthetic"
    (source.parent / "__init__.py").write_text('"""Inert owner package."""\n')
    (test.parent / "__init__.py").write_text("")
    (test / "__init__.py").write_text("")
    (source.parent / "OWNER.yaml").write_text(
        "schema_version: 1\n"
        f"owner_id: {owner}\n"
        f"owner_display_name: {display}\n"
        f"contribution_kind: {contribution_kind}\n"
        "code_license: Apache-2.0\n"
        "content_license: CC-BY-4.0\n"
    )
    op_id = operation_id or owner + "_" + operation
    (source / "CONTRIBUTION.yaml").write_text(
        "schema_version: 1\n"
        f"owner_id: {owner}\npack_id: {pack}\noperation_id: {op_id}\n"
        f"operation_type: {operation_type}\n"
        f"aliases: {list(aliases)!r}\n"
        f"module: operatebench.contributions.{owner}.{operation}\n"
        f"source_path: src/operatebench/contributions/{owner}/{operation}\n"
        f"test_path: tests/contributions/{owner}/{operation}/test_pack.py\n"
        f"fixture_path: examples/contributions/{owner}/{operation}/operation.yaml\n"
        f"documentation_path: docs/contributions/{owner}/{operation}.md\n"
    )
    (source / "__init__.py").write_text('"""Operation package."""\n')
    (source / "pack.py").write_text(
        "from operatebench.sdk import OperationPackMetadata\n"
        "class SamplePack:\n"
        "    metadata = OperationPackMetadata(\n"
        f"        pack_id={pack!r}, operation_type={operation_type!r},\n"
        f"        operation_id={op_id!r},\n"
        "        pack_version='1.0.0', display_name='Synthetic sample',\n"
        f"        owner_id={owner!r}, owner_display_name={display!r},\n"
        f"        contribution_kind={contribution_kind!r}, status='incubator',\n"
        "        privacy_status='SYNTHETIC_ONLY',\n"
        "        default_spec="
        f"'examples/contributions/{owner}/{operation}/operation.yaml',\n"
        "        agent_ids=('reference',), evidence_eligible=False,\n"
        f"        aliases={aliases!r},\n"
        "    )\n"
        "PACK = SamplePack()\n"
    )
    (test / "test_pack.py").write_text("def test_inert(): pass\n")
    (fixture / "operation.yaml").write_text("synthetic: true\n")
    (docs / f"{operation}.md").write_text("Synthetic only.\n")
    return source


def _tree_snapshot(root: Path) -> dict[str, bytes | None]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None
        for path in sorted(root.rglob("*"))
    }


def test_positive_controls(tmp_path: Path) -> None:
    _owner(tmp_path, "northstar")
    _owner(tmp_path, "riverside")
    assert check_contributions(tmp_path) == (
        "northstar:northstar_sample",
        "riverside:riverside_sample",
    )
    assert check_contributions(Path.cwd()) == (
        "prospire:lettings_property_compliance_synthetic_v1",
    )


@pytest.mark.parametrize(
    "operation",
    [
        ".hidden",
        "odd.name",
        "odd name",
        "class",
        "con",
        "1operation",
        "Operation",
        "odd-name",
        "a" * 64,
        "prn",
        "com1",
    ],
)
def test_operation_directory_name_must_be_safe_python_component(
    tmp_path: Path, operation: str
) -> None:
    _owner(
        tmp_path,
        "northstar",
        operation=operation,
        operation_id="northstar_safe_directory_test",
        pack_id="partner.northstar.safe_directory_test",
        operation_type="northstar.safe_directory_test.synthetic",
    )

    with pytest.raises(
        ContributionError,
        match=(
            r"operation directory name is not a safe lowercase Python identifier"
            r"|hidden entry"
        ),
    ):
        check_contributions(tmp_path)


@pytest.mark.parametrize("operation", ["return_refund", "property_compliance", "a" * 63])
def test_operation_directory_name_accepts_safe_python_component(
    tmp_path: Path, operation: str
) -> None:
    _owner(tmp_path, "northstar", operation=operation)

    assert check_contributions(tmp_path) == (f"northstar:northstar_{operation}",)


def test_unique_unregistered_prospire_maintainer_pack_is_authorship_only(
    tmp_path: Path,
) -> None:
    from operatebench.sdk.builtins import BUILTIN_PACKS

    operation_id = "prospire_unregistered_authorship_example_v1"
    _owner(
        tmp_path,
        "prospire",
        operation="authorship_example",
        operation_id=operation_id,
        display_name="PROSPIRE TECHNOLOGIES LTD",
        pack_id="prospire.authorship_example.synthetic",
        operation_type="prospire.authorship_example.synthetic",
        contribution_kind="maintainer",
    )

    assert check_contributions(tmp_path) == (f"prospire:{operation_id}",)
    assert "prospire.authorship_example.synthetic" not in BUILTIN_PACKS.canonical_ids()
    assert all(
        metadata.operation_id != operation_id for metadata in BUILTIN_PACKS.metadata()
    )


def test_reserved_prospire_owner_cannot_be_redeclared_as_partner(
    tmp_path: Path,
) -> None:
    _owner(
        tmp_path,
        "prospire",
        operation="impostor",
        operation_id="prospire_impostor_synthetic_v1",
        display_name="Impostor Fictional Company",
        pack_id="partner.prospire.impostor",
        operation_type="prospire.impostor.synthetic",
        contribution_kind="partner",
    )

    with pytest.raises(ContributionError, match="reserved owner identity"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    ("display_name", "contribution_kind"),
    [
        ("PROSPIRE TECHNOLOGIES LTD", "partner"),
        ("Impostor Fictional Company", "maintainer"),
    ],
)
def test_reserved_owner_requires_exact_display_and_contribution_kind(
    tmp_path: Path, display_name: str, contribution_kind: str
) -> None:
    _owner(
        tmp_path,
        "prospire",
        operation="authority_mismatch",
        operation_id="prospire_authority_mismatch_synthetic_v1",
        display_name=display_name,
        pack_id="partner.prospire.authority_mismatch",
        operation_type="prospire.authority_mismatch.synthetic",
        contribution_kind=contribution_kind,
    )

    with pytest.raises(ContributionError, match="reserved owner identity"):
        check_contributions(tmp_path)


@pytest.mark.parametrize("name", ["helper.py", "conftest.py", "payload.bin", ".hidden"])
def test_contribution_root_rejects_undeclared_regular_entries(
    tmp_path: Path, name: str
) -> None:
    _owner(tmp_path, "northstar")
    (tmp_path / "src/operatebench/contributions" / name).write_bytes(b"payload\x00")
    with pytest.raises(ContributionError, match="contribution root"):
        check_contributions(tmp_path)


def test_contribution_root_requires_inert_init_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    _owner(tmp_path, "northstar")
    root = tmp_path / "src/operatebench/contributions"
    (root / "__init__.py").write_text("MUTATED = True\n")
    with pytest.raises(ContributionError, match="package docstring"):
        check_contributions(tmp_path)
    (root / "__init__.py").write_text('"""Contribution package."""\n')
    (root / "linked").symlink_to("northstar", target_is_directory=True)
    with pytest.raises(ContributionError, match="symlink"):
        check_contributions(tmp_path)


def test_contribution_root_rejects_empty_init_and_special_entry(tmp_path: Path) -> None:
    _owner(tmp_path, "northstar")
    root = tmp_path / "src/operatebench/contributions"
    (root / "__init__.py").write_text("")
    with pytest.raises(ContributionError, match="package docstring"):
        check_contributions(tmp_path)
    (root / "__init__.py").write_text('"""Contribution package."""\n')
    special = root / "special"
    try:
        os.mkfifo(special)
    except (AttributeError, OSError):
        pytest.skip("FIFO creation unavailable")
    with pytest.raises(ContributionError, match="contribution root"):
        check_contributions(tmp_path)


@pytest.mark.parametrize("name", ["conftest.py", "__init__.py", "payload.bin", ".hidden"])
def test_shared_test_root_rejects_root_level_files(tmp_path: Path, name: str) -> None:
    _owner(tmp_path, "northstar")
    (tmp_path / "tests/contributions" / name).write_bytes(b"payload\x00")
    with pytest.raises(ContributionError, match=r"shared test root|cannot parse Python"):
        check_contributions(tmp_path)


def test_shared_test_root_rejects_cross_owner_conftest_and_orphan_owner(
    tmp_path: Path,
) -> None:
    _owner(tmp_path, "northstar")
    root = tmp_path / "tests/contributions"
    (root / "conftest.py").write_text(
        "import operatebench.contributions.riverside.sample\n"
    )
    with pytest.raises(ContributionError, match="shared test root"):
        check_contributions(tmp_path)
    (root / "conftest.py").unlink()
    (root / "orphan").mkdir()
    with pytest.raises(ContributionError, match="owner directories"):
        check_contributions(tmp_path)


def test_shared_test_root_rejects_symlink(tmp_path: Path) -> None:
    _owner(tmp_path, "northstar")
    root = tmp_path / "tests/contributions"
    (root / "linked").symlink_to("northstar", target_is_directory=True)
    with pytest.raises(ContributionError, match=r"shared test root.*symlink"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    "display",
    ["unsafe\u202ename", "unsafe\u202dname", "unsafe\u2068name", "zero\u200bwidth"],
)
def test_owner_manifest_rejects_unicode_format_display(
    tmp_path: Path, display: str
) -> None:
    source = _owner(tmp_path, "northstar")
    owner = source.parent / "OWNER.yaml"
    owner.write_text(owner.read_text().replace("Northstar Fictional Company", display))
    with pytest.raises(ContributionError, match="owner_display_name"):
        check_contributions(tmp_path)


def test_owner_manifest_accepts_ordinary_international_display(tmp_path: Path) -> None:
    _owner(tmp_path, "northstar", display_name="Étoile 技術株式会社 № ٢")
    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


@pytest.mark.parametrize(
    "snippet",
    [
        "import operatebench\nvalue = operatebench.domains\n",
        "import operatebench as ob\nvalue = ob.sdk\n",
        "import operatebench.sdk\n",
        "import operatebench.contributions.northstar\n",
    ],
)
def test_bare_operatebench_imports_are_forbidden(tmp_path: Path, snippet: str) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "extra.py").write_text(snippet)
    with pytest.raises(ContributionError, match="bare operatebench import"):
        check_contributions(tmp_path)


def test_explicit_sdk_and_stdlib_imports_remain_allowed(tmp_path: Path) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "extra.py").write_text(
        "from operatebench.sdk import OperationPackMetadata\nimport json\n"
    )
    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


@pytest.mark.parametrize("location", ["source", "test"])
@pytest.mark.parametrize(
    "target",
    [
        "src.operatebench",
        "src.operatebench.domains.lettings.maintenance.pack",
        "src.operatebench.contributions.riverside.sample",
        "src.operatebench.contributions.northstar.other_operation",
        "src.operatebench._resource_access",
    ],
)
@pytest.mark.parametrize("syntax", ["import", "from"])
def test_repository_layout_import_spelling_is_rejected_without_writes(
    tmp_path: Path, location: str, target: str, syntax: str
) -> None:
    source = _owner(tmp_path, "northstar")
    if location == "source":
        nested = source / "nested"
        nested.mkdir()
        (nested / "__init__.py").write_text('"""Nested package."""\n')
        path = nested / "hostile.py"
    else:
        path = tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    snippet = (
        f"import {target} as x\n" if syntax == "import" else f"from {target} import X\n"
    )
    path.write_text(snippet)
    before = _tree_snapshot(tmp_path)

    with pytest.raises(ContributionError) as caught:
        check_contributions(tmp_path)

    assert caught.value.args == (
        "non-canonical import spelling 'src.operatebench' is forbidden",
    )
    assert _tree_snapshot(tmp_path) == before


@pytest.mark.parametrize("location", ["source", "test"])
def test_canonical_same_operation_import_remains_allowed(
    tmp_path: Path, location: str
) -> None:
    source = _owner(tmp_path, "northstar")
    path = (
        source / "extra.py"
        if location == "source"
        else tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    )
    path.write_text("from operatebench.contributions.northstar.sample import pack\n")
    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


def test_unreserved_owner_cannot_self_declare_as_maintainer(tmp_path: Path) -> None:
    source = _owner(tmp_path, "northstar")
    owner = source.parent / "OWNER.yaml"
    owner.write_text(
        owner.read_text().replace(
            "contribution_kind: partner", "contribution_kind: maintainer"
        )
    )
    pack = source / "pack.py"
    pack.write_text(
        pack.read_text().replace(
            "contribution_kind='partner'", "contribution_kind='maintainer'"
        )
    )
    with pytest.raises(ContributionError, match="maintainer"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    "snippet",
    [
        "from operatebench._resource_access import files, as_file\n"
        "resource = files('operatebench.domains.lettings.maintenance')\n",
        "from operatebench._resource_access import files, as_file\n"
        "resource = files('operatebench.contributions.riverside.sample')\n",
        "from operatebench._resource_access import files, as_file\n"
        "package = __package__\nresource = files(package)\n",
        pytest.param(
            "import operatebench._resource_access as resources\n"
            "resource = resources.files(__package__)\n",
            id="bare-resource-import",
        ),
        "from operatebench._resource_access import files, as_file\nloader = files\n",
        "from operatebench._resource_access import files, as_file\n"
        "def accept(loader): return loader\nresource = accept(files)\n",
        "from operatebench._resource_access import files, as_file\n"
        "resource = files(package=__package__)\n",
        "from operatebench._resource_access import files, as_file, Package\n",
    ],
)
def test_resource_access_is_limited_to_direct_own_package_call(
    tmp_path: Path, snippet: str
) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "extra.py").write_text(snippet)
    with pytest.raises(ContributionError, match=r"resource access|bare operatebench"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    "snippet",
    [
        "LOADER = files\nif True:\n"
        "    from operatebench._resource_access import files, as_file\n",
        "LOADER = files\ndef nested():\n"
        "    from operatebench._resource_access import files, as_file\n",
        "LOADER = files\ntry:\n"
        "    from operatebench._resource_access import files, as_file\n"
        "except ImportError:\n    pass\n",
    ],
)
def test_resource_access_import_must_be_top_level(tmp_path: Path, snippet: str) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "extra.py").write_text(snippet)
    with pytest.raises(ContributionError, match="resource access import"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    "snippet",
    [
        "from operatebench._resource_access import files as resource_files, as_file\n",
        "from operatebench._resource_access import files, as_file as resource_as_file\n",
        "from operatebench._resource_access import files, as_file\nfiles = object()\n",
        "from operatebench._resource_access import files, as_file\n"
        "def give_back():\n    return files\n",
    ],
)
def test_resource_access_rejects_aliases_and_rebinding(
    tmp_path: Path, snippet: str
) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "extra.py").write_text(snippet)
    with pytest.raises(ContributionError, match="resource access"):
        check_contributions(tmp_path)


def test_resource_access_cannot_be_reexported_by_same_operation_helper(
    tmp_path: Path,
) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "helper.py").write_text(
        "from operatebench._resource_access import files, as_file\n"
    )
    (source / "extra.py").write_text(
        "from .helper import files, as_file\n"
        "resource = files(__package__).joinpath('fixture.yaml')\n"
    )
    with pytest.raises(ContributionError, match="resource access"):
        check_contributions(tmp_path)


def test_direct_own_package_resource_access_is_allowed(tmp_path: Path) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "fixture.yaml").write_text("synthetic: true\n")
    (source / "extra.py").write_text(
        "from operatebench._resource_access import files, as_file\n"
        "resource = files(__package__).joinpath('fixture.yaml')\n"
        "def concrete():\n"
        "    return as_file(resource)\n"
    )
    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


def test_resource_alias_cannot_launder_cross_owner_parent_traversal(
    tmp_path: Path,
) -> None:
    source = _owner(tmp_path, "northstar")
    foreign = _owner(tmp_path, "riverside", operation="riverside_sample")
    (source / "fixture.yaml").write_text("synthetic: true\n")
    (foreign / "foreign_secret.yaml").write_text("owner: riverside\n")
    (source / "extra.py").write_text(
        "from operatebench._resource_access import files, as_file\n"
        "resource = files(__package__).joinpath('fixture.yaml')\n"
        "foreign = resource.parent.joinpath(\n"
        "    '..', '..', 'riverside', 'riverside_sample', "
        "'foreign_secret.yaml'\n"
        ")\n"
        "def concrete():\n"
        "    return as_file(foreign)\n"
    )

    with pytest.raises(ContributionError, match="resource"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    "use",
    [
        "foreign = resource.parent\n",
        "contents = resource.read_text()\n",
        "other = resource\n",
        "left = right = resource\n",
        "resource = object()\n",
        "del resource\n",
        "resource += 'other'\n",
        "if (other := resource):\n    pass\n",
        "def leak():\n    return resource\n",
        "def shadow(resource):\n    return as_file(resource)\n",
        "def accept(value):\n    return value\naccept(resource)\n",
        "items = [resource]\n",
    ],
)
def test_resource_alias_rejects_every_non_as_file_use(tmp_path: Path, use: str) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "fixture.yaml").write_text("synthetic: true\n")
    (source / "extra.py").write_text(
        "from operatebench._resource_access import files, as_file\n"
        "resource = files(__package__).joinpath('fixture.yaml')\n"
        f"{use}"
    )

    with pytest.raises(ContributionError, match="resource"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    "use",
    [
        "files(__package__).joinpath('fixture.yaml').parent",
        "files(__package__).joinpath('fixture.yaml').read_text()",
        "(files(__package__).joinpath('fixture.yaml'),)",
        "as_file(resource=files(__package__).joinpath('fixture.yaml'))",
        "as_file(files(__package__).joinpath('fixture.yaml'), None)",
    ],
)
def test_direct_resource_result_rejects_attribute_and_method_use(
    tmp_path: Path, use: str
) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "fixture.yaml").write_text("synthetic: true\n")
    (source / "extra.py").write_text(
        f"from operatebench._resource_access import files, as_file\nvalue = {use}\n"
    )

    with pytest.raises(ContributionError, match="resource"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    "snippet",
    [
        "with as_file(files(__package__).joinpath('fixture.yaml')) as concrete:\n"
        "    value = concrete.read_text()\n",
        "resource = files(__package__).joinpath('fixture.yaml')\n"
        "with as_file(resource) as concrete:\n"
        "    value = concrete.read_text()\n",
    ],
)
def test_resource_result_accepts_exact_as_file_patterns(
    tmp_path: Path, snippet: str
) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "fixture.yaml").write_text("synthetic: true\n")
    (source / "extra.py").write_text(
        "from operatebench._resource_access import files, as_file\n" + snippet
    )

    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


def test_resource_access_rejects_cross_owner_multi_arg_parent_traversal(
    tmp_path: Path,
) -> None:
    source = _owner(tmp_path, "northstar")
    foreign = _owner(tmp_path, "riverside", operation="riverside_sample")
    (foreign / "foreign_secret.yaml").write_text("owner: riverside\n")
    (source / "extra.py").write_text(
        "from operatebench._resource_access import files, as_file\n"
        "resource = files(__package__).joinpath(\n"
        "    '..', '..', 'riverside', 'riverside_sample', 'foreign_secret.yaml'\n"
        ")\n"
    )

    with pytest.raises(ContributionError, match="resource"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    "expression",
    [
        "files(__package__).joinpath('../foreign.yaml')",
        "files(__package__).joinpath('/absolute.yaml')",
        "files(__package__).joinpath('nested/fixture.yaml')",
        "files(__package__).joinpath('nested\\\\fixture.yaml')",
        "files(__package__).joinpath(resource_name)",
        "files(__package__).joinpath('fixture.yaml', 'other.yaml')",
        "files(__package__)",
        "files(__package__) / 'fixture.yaml'",
        "files(__package__).parent.joinpath('fixture.yaml')",
        "files(__package__).joinpath('fixture.yaml').joinpath('other.yaml')",
        "files(__package__).joinpath(name='fixture.yaml')",
        "files(__package__).joinpath('')",
        "files(__package__).joinpath('.')",
        "files(__package__).joinpath('..')",
        "files(__package__).joinpath('.hidden.yaml')",
        "files(__package__).joinpath('unsafe\\n.yaml')",
        "files(__package__).joinpath('unsafe\\u202e.yaml')",
    ],
)
def test_resource_access_rejects_non_direct_literal_file_uses(
    tmp_path: Path, expression: str
) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "fixture.yaml").write_text("synthetic: true\n")
    (source / "extra.py").write_text(
        "from operatebench._resource_access import files, as_file\n"
        "resource_name = 'fixture.yaml'\n"
        f"resource = {expression}\n"
    )

    with pytest.raises(ContributionError, match="resource"):
        check_contributions(tmp_path)


@pytest.mark.parametrize("kind", ["missing", "directory", "symlink", "suffix"])
def test_resource_access_rejects_invalid_direct_child(tmp_path: Path, kind: str) -> None:
    source = _owner(tmp_path, "northstar")
    name = "fixture.yaml"
    if kind == "directory":
        (source / name).mkdir()
    elif kind == "symlink":
        (source / "target.yaml").write_text("synthetic: true\n")
        (source / name).symlink_to("target.yaml")
    elif kind == "suffix":
        name = "fixture.bin"
        (source / name).write_bytes(b"synthetic")
    (source / "extra.py").write_text(
        "from operatebench._resource_access import files, as_file\n"
        f"resource = files(__package__).joinpath({name!r})\n"
    )

    with pytest.raises(ContributionError, match="resource"):
        check_contributions(tmp_path)


@pytest.mark.parametrize("name", ["fixture.yaml", "fixture.json", "fixture.txt"])
def test_resource_access_accepts_same_package_public_file_and_as_file(
    tmp_path: Path, name: str
) -> None:
    source = _owner(tmp_path, "northstar")
    (source / name).write_text("synthetic\n")
    (source / "extra.py").write_text(
        "from operatebench._resource_access import files, as_file\n"
        f"resource = files(__package__).joinpath({name!r})\n"
        "def concrete():\n"
        "    return as_file(resource)\n"
    )

    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


@pytest.mark.parametrize("location", ["source", "test"])
def test_namespace_escape_cannot_hijack_own_package_resource_anchor(
    tmp_path: Path, location: str
) -> None:
    source = _owner(tmp_path, "northstar")
    path = (
        source / "extra.py"
        if location == "source"
        else tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    )
    path.write_text(
        "from operatebench._resource_access import files, as_file\n"
        "globals()['__package__'] = "
        "'operatebench.contributions.prospire.property_compliance'\n"
        "ROOT = files(__package__)\n"
    )
    with pytest.raises(ContributionError, match="namespace escape primitive 'globals'"):
        check_contributions(tmp_path)


@pytest.mark.parametrize("location", ["source", "test"])
@pytest.mark.parametrize("primitive", ["globals", "locals", "vars"])
def test_namespace_escape_primitives_are_forbidden_in_nested_contexts(
    tmp_path: Path, location: str, primitive: str
) -> None:
    source = _owner(tmp_path, "northstar")
    path = (
        source / "extra.py"
        if location == "source"
        else tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    )
    path.write_text(
        f"def hostile():\n    namespace = {primitive}\n    return namespace\n"
    )
    with pytest.raises(
        ContributionError, match=rf"namespace escape primitive '{primitive}'"
    ):
        check_contributions(tmp_path)


def test_globals_update_cannot_hijack_package_in_nested_context(tmp_path: Path) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "extra.py").write_text(
        "def hostile():\n"
        "    globals().update(__package__='operatebench.contributions.prospire')\n"
    )
    with pytest.raises(ContributionError, match="namespace escape primitive 'globals'"):
        check_contributions(tmp_path)


def test_safe_owner_level_fixture_conftest_is_allowed(tmp_path: Path) -> None:
    _owner(tmp_path, "northstar")
    conftest = tmp_path / "tests/contributions/northstar/conftest.py"
    conftest.write_text(
        "import pytest\n@pytest.fixture\ndef synthetic_value():\n    return 1\n"
    )
    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


@pytest.mark.parametrize(
    "snippet",
    [
        "MUTATED = True\n",
        "import os\n",
        "def helper():\n    pass\n",
        "class Helper:\n    pass\n",
        "print('side effect')\n",
    ],
)
def test_owner_test_init_must_be_inert(tmp_path: Path, snippet: str) -> None:
    _owner(tmp_path, "northstar")
    init = tmp_path / "tests/contributions/northstar/__init__.py"
    init.write_text(snippet)
    with pytest.raises(ContributionError, match=rf"owner root {init} must be inert"):
        check_contributions(tmp_path)


@pytest.mark.parametrize("snippet", ["", '"""Owner test package."""\n'])
def test_owner_test_init_accepts_empty_or_docstring_only(
    tmp_path: Path, snippet: str
) -> None:
    _owner(tmp_path, "northstar")
    (tmp_path / "tests/contributions/northstar/__init__.py").write_text(snippet)
    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


@pytest.mark.parametrize(
    ("snippet", "message"),
    [
        ("import operatebench.contributions.riverside.sample\n", "bare operatebench"),
        ("import operatebench.domains.lettings.maintenance\n", "bare operatebench"),
        ("import operatebench.contributions.northstar.sample\n", "bare operatebench"),
        ("import importlib\n", "dynamic loading"),
        ("exec('pass')\n", "dynamic execution"),
    ],
)
def test_owner_level_conftest_is_domain_neutral(
    tmp_path: Path, snippet: str, message: str
) -> None:
    _owner(tmp_path, "northstar")
    (tmp_path / "tests/contributions/northstar/conftest.py").write_text(snippet)
    with pytest.raises(ContributionError, match=message):
        check_contributions(tmp_path)


@pytest.mark.parametrize("name", ["payload.pyc", "payload.so", ".cache"])
def test_owner_test_root_rejects_binary_hidden_entries(tmp_path: Path, name: str) -> None:
    _owner(tmp_path, "northstar")
    (tmp_path / f"tests/contributions/northstar/{name}").write_bytes(b"bad\x00")
    with pytest.raises(ContributionError, match="owner test root"):
        check_contributions(tmp_path)


def test_owner_test_root_rejects_symlink_and_undeclared_directory(
    tmp_path: Path,
) -> None:
    _owner(tmp_path, "northstar")
    owner_tests = tmp_path / "tests/contributions/northstar"
    (owner_tests / "undeclared").mkdir()
    with pytest.raises(ContributionError, match="undeclared"):
        check_contributions(tmp_path)
    (owner_tests / "undeclared").rmdir()
    (owner_tests / "linked").symlink_to("sample", target_is_directory=True)
    with pytest.raises(ContributionError, match="symlink"):
        check_contributions(tmp_path)


def test_reserved_maintenance_identity_cannot_be_claimed_by_disposable_owner(
    tmp_path: Path,
) -> None:
    _owner(
        tmp_path,
        "disposable",
        operation_id="lettings_maintenance_synthetic_v1",
        pack_id="lettings.maintenance.synthetic",
        operation_type="lettings.maintenance.synthetic",
    )
    with pytest.raises(ContributionError, match="reserved"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    "display_name",
    [
        "prospire technologies ltd",
        "PROSPIRE TECHNOLOGIES LTD",
        "Prospire Technologies Ltd",
    ],
)
def test_reserved_display_cannot_be_claimed_by_disposable_owner(
    tmp_path: Path, display_name: str
) -> None:
    _owner(tmp_path, "attacker", display_name=display_name)
    with pytest.raises(ContributionError, match="display name"):
        check_contributions(tmp_path)


def test_non_nfc_owner_display_manifest_is_rejected(tmp_path: Path) -> None:
    _owner(tmp_path, "northstar", display_name="E\u0301toile")
    with pytest.raises(ContributionError, match="owner_display_name"):
        check_contributions(tmp_path)


def test_reserved_display_canonical_equivalent_cannot_be_claimed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _owner(tmp_path, "attacker", display_name="E\u0301toile")
    manifest = tmp_path / "src/operatebench/resources/operation_pack_registry.yaml"
    manifest.write_text(
        manifest.read_text().replace("PROSPIRE TECHNOLOGIES LTD", "Étoile")
    )
    monkeypatch.setattr(isolation, "is_safe_owner_display_name", lambda _value: True)
    with pytest.raises(ContributionError, match="display name"):
        check_contributions(tmp_path)


def test_reserved_manifest_rejects_casefold_display_claimed_by_other_owner(
    tmp_path: Path,
) -> None:
    _owner(tmp_path, "northstar")
    manifest = tmp_path / "src/operatebench/resources/operation_pack_registry.yaml"
    text = manifest.read_text()
    text = text.replace("owner_id: prospire", "owner_id: northstar", 1)
    text = text.replace(
        "owner_display_name: PROSPIRE TECHNOLOGIES LTD",
        "owner_display_name: prospire technologies ltd",
        1,
    )
    manifest.write_text(text)
    with pytest.raises(ContributionError, match="display name"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    "value",
    ["Uppercase", "has space", "café", ".leading", "a" * 129, True, 7],
)
@pytest.mark.parametrize("field", ["pack_id", "operation_type", "operation_id", "alias"])
def test_contribution_manifest_identity_uses_runtime_grammar(
    tmp_path: Path, field: str, value: object
) -> None:
    source = _owner(tmp_path, "northstar")
    manifest = source / "CONTRIBUTION.yaml"
    text = manifest.read_text()
    if field == "alias":
        text = text.replace("aliases: []", f"aliases: [{value!r}]")
    else:
        original = {
            "pack_id": "partner.northstar.sample",
            "operation_type": "northstar.sample.synthetic",
            "operation_id": "northstar_sample",
        }[field]
        text = text.replace(f"{field}: {original}", f"{field}: {value!r}")
    manifest.write_text(text)
    with pytest.raises(ContributionError, match="operation identity"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    "value",
    ["Uppercase", "has space", "café", ".leading", "a" * 129, True, 7],
)
@pytest.mark.parametrize("field", ["pack_id", "operation_type", "operation_id", "alias"])
def test_reserved_manifest_identity_uses_runtime_grammar(
    tmp_path: Path, field: str, value: object
) -> None:
    _owner(tmp_path, "northstar")
    manifest = tmp_path / "src/operatebench/resources/operation_pack_registry.yaml"
    text = manifest.read_text()
    if field == "alias":
        text = text.replace("      - return-refund", f"      - {value!r}", 1)
    else:
        original = {
            "pack_id": "commerce.return_refund.synthetic.v1",
            "operation_type": "commerce.return_refund.synthetic.v1",
            "operation_id": "commerce_return_refund_synthetic_v1",
        }[field]
        text = text.replace(f"    {field}: {original}", f"    {field}: {value!r}", 1)
    manifest.write_text(text)
    with pytest.raises(ContributionError, match="operation identity"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    ("pack_id", "operation_type", "operation_id", "aliases"),
    [
        ("commerce.return_refund.synthetic.v1", "unique.type", "unique_operation", ()),
        (
            "partner.northstar.sample",
            "commerce.return_refund.synthetic.v1",
            "unique_operation",
            (),
        ),
        (
            "partner.northstar.sample",
            "unique.type",
            "commerce_return_refund_synthetic_v1",
            (),
        ),
        (
            "partner.northstar.sample",
            "unique.type",
            "unique_operation",
            ("return-refund",),
        ),
    ],
)
def test_each_reserved_commerce_identity_is_rejected(
    tmp_path: Path,
    pack_id: str,
    operation_type: str,
    operation_id: str,
    aliases: tuple[str, ...],
) -> None:
    _owner(
        tmp_path,
        "northstar",
        pack_id=pack_id,
        operation_type=operation_type,
        operation_id=operation_id,
        aliases=aliases,
    )
    with pytest.raises(ContributionError, match="reserved"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    ("reserved_identity", "incoming_field"),
    [
        (identity, field)
        for identity in (
            "commerce.return_refund.synthetic.v1",
            "commerce.return_refund.synthetic.v1",
            "return-refund",
            "commerce_return_refund_synthetic_v1",
        )
        for field in ("pack_id", "operation_type", "aliases", "operation_id")
    ],
)
def test_every_reserved_identity_kind_is_rejected_in_every_incoming_field(
    tmp_path: Path, reserved_identity: str, incoming_field: str
) -> None:
    value: object = (
        (reserved_identity,) if incoming_field == "aliases" else reserved_identity
    )
    _owner(tmp_path, "northstar", **{incoming_field: value})  # type: ignore[arg-type]
    with pytest.raises(ContributionError, match="reserved"):
        check_contributions(tmp_path)


def test_reserved_property_compliance_identity_rejects_wrong_owner_and_path(
    tmp_path: Path,
) -> None:
    _owner(
        tmp_path,
        "northstar",
        operation_id="lettings_property_compliance_synthetic_v1",
        pack_id="lettings.property_compliance.synthetic",
        operation_type="lettings.property_compliance.synthetic",
    )
    with pytest.raises(ContributionError, match="reserved"):
        check_contributions(tmp_path)


def test_two_unregistered_contributions_cannot_share_operation_id(tmp_path: Path) -> None:
    _owner(tmp_path, "northstar", operation_id="shared_operation_v1")
    _owner(tmp_path, "riverside", operation_id="shared_operation_v1")
    with pytest.raises(ContributionError, match="duplicate operation_id"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    ("left_field", "right_field"),
    [
        ("aliases", "operation_id"),
        ("pack_id", "operation_id"),
        ("operation_type", "operation_id"),
        ("operation_id", "pack_id"),
        ("operation_id", "operation_type"),
        ("operation_id", "aliases"),
    ],
)
def test_unregistered_rows_reject_cross_kind_identity_collisions(
    tmp_path: Path, left_field: str, right_field: str
) -> None:
    identity = (
        "partner.northstar.sample"
        if left_field == "pack_id"
        else "partner.riverside.sample"
        if right_field == "pack_id"
        else "shared"
    )
    values = []
    for field in (left_field, right_field):
        values.append({field: (identity,) if field == "aliases" else identity})
    _owner(tmp_path, "northstar", **values[0])
    _owner(tmp_path, "riverside", **values[1])
    with pytest.raises(ContributionError, match="operation identity"):
        check_contributions(tmp_path)


@pytest.mark.parametrize("own_field", ["pack_id", "operation_type", "aliases"])
def test_unregistered_row_rejects_operation_id_equal_to_own_identity(
    tmp_path: Path, own_field: str
) -> None:
    identity = "partner.northstar.sample" if own_field == "pack_id" else "shared"
    values: dict[str, object] = {
        "operation_id": identity,
        own_field: (identity,) if own_field == "aliases" else identity,
    }
    _owner(tmp_path, "northstar", **values)  # type: ignore[arg-type]
    with pytest.raises(ContributionError, match="operation identity"):
        check_contributions(tmp_path)


def test_unregistered_row_accepts_same_pack_id_and_operation_type(tmp_path: Path) -> None:
    _owner(
        tmp_path,
        "northstar",
        pack_id="partner.northstar.sample",
        operation_type="partner.northstar.sample",
    )
    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_key",
        "duplicate_row",
        "unknown_field",
        "non_string_key",
        "wrong_order",
    ],
)
def test_reserved_manifest_is_closed_and_strict(tmp_path: Path, mutation: str) -> None:
    _owner(tmp_path, "northstar")
    manifest = tmp_path / "src/operatebench/resources/operation_pack_registry.yaml"
    text = manifest.read_text()
    if mutation == "duplicate_key":
        text = text.replace(
            "schema_version: 1", "schema_version: 1\nschema_version: 1", 1
        )
    elif mutation == "duplicate_row":
        first, remainder = text.split("  - owner_id:", 1)
        row, rest = remainder.split("  - owner_id:", 1)
        text = (
            first + "  - owner_id:" + row + "  - owner_id:" + row + "  - owner_id:" + rest
        )
    elif mutation == "unknown_field":
        text = text.replace("operations:", "unknown: value\noperations:", 1)
    elif mutation == "non_string_key":
        text = text.replace("schema_version: 1", "1: invalid", 1)
    else:
        text = text.replace(
            "schema_version: 1\noperations:", "operations:\nschema_version: 1", 1
        )
    manifest.write_text(text)
    with pytest.raises(ContributionError):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    ("field", "forged"),
    [
        ("owner_id", "attacker"),
        ("owner_display_name", "Attacker Ltd"),
        ("contribution_kind", "maintainer"),
        ("pack_id", "partner.attacker.sample"),
        ("operation_type", "attacker.sample.synthetic"),
    ],
)
def test_metadata_fields_cannot_be_forged_by_docstrings(
    tmp_path: Path, field: str, forged: str
) -> None:
    source = _owner(tmp_path, "northstar")
    pack = source / "pack.py"
    text = pack.read_text()
    expected = {
        "owner_id": "northstar",
        "owner_display_name": "Northstar Fictional Company",
        "contribution_kind": "partner",
        "pack_id": "partner.northstar.sample",
        "operation_type": "northstar.sample.synthetic",
    }[field]
    pack.write_text(
        repr(expected)
        + "\n"
        + text.replace(f"{field}={expected!r}", f"{field}={forged!r}")
    )
    with pytest.raises(ContributionError, match=field):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    "replacement",
    [
        "metadata = OperationPackMetadata(**FIELDS)",
        "metadata = OperationPackMetadata('laundered')",
        "metadata = Alias(owner_id='northstar')",
        "metadata = wrapper(OperationPackMetadata(owner_id='northstar'))",
        "metadata = OperationPackMetadata(owner_id='north' + 'star')",
        "metadata = OperationPackMetadata(owner_id=OWNER_ID)",
        "metadata = OperationPackMetadata(owner_id='northstar')\n    metadata = object()",
        "metadata = OperationPackMetadata(owner_id='northstar')\n"
        "    other = OperationPackMetadata(owner_id='northstar')",
    ],
)
def test_metadata_rejects_laundering(tmp_path: Path, replacement: str) -> None:
    source = _owner(tmp_path, "northstar")
    pack = source / "pack.py"
    text = pack.read_text()
    start = text.index("    metadata =")
    pack.write_text(text[:start] + "    " + replacement + "\n")
    with pytest.raises(ContributionError, match="metadata"):
        check_contributions(tmp_path)


def test_metadata_constructor_cannot_be_shadowed(tmp_path: Path) -> None:
    source = _owner(tmp_path, "northstar")
    pack = source / "pack.py"
    pack.write_text(
        pack.read_text().replace(
            "class SamplePack:",
            "def OperationPackMetadata(**kwargs): return kwargs\nclass SamplePack:",
        )
    )
    with pytest.raises(ContributionError, match="direct SDK import"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    [
        ("pack_version='1.0.0'", "pack_version=VERSION", "pack_version"),
        ("pack_version='1.0.0'", "pack_version='1.' + '0.0'", "pack_version"),
        ("pack_version='1.0.0'", "pack_version='1.0'", "pack_version"),
        (
            "display_name='Synthetic sample'",
            "display_name=make_name()",
            "display_name",
        ),
        (
            "display_name='Synthetic sample'",
            "display_name='Synthetic ' + 'sample'",
            "display_name",
        ),
        (
            "display_name='Synthetic sample'",
            "display_name='unsafe\\u200bname'",
            "display_name",
        ),
        ("agent_ids=('reference',)", "agent_ids=AGENTS", "agent_ids"),
        (
            "agent_ids=('reference',)",
            "agent_ids=tuple(agent for agent in AGENTS)",
            "agent_ids",
        ),
        (
            "agent_ids=('reference',)",
            "agent_ids=('ref' + 'erence',)",
            "agent_ids",
        ),
        (
            "agent_ids=('reference',)",
            "agent_ids=(*AGENTS,)",
            "agent_ids",
        ),
        (
            "agent_ids=('reference',)",
            "agent_ids=('zeta', 'alpha')",
            "agent_ids",
        ),
    ],
)
def test_metadata_semantic_fields_are_direct_valid_literals(
    tmp_path: Path, original: str, replacement: str, message: str
) -> None:
    source = _owner(tmp_path, "northstar")
    pack = source / "pack.py"
    pack.write_text(pack.read_text().replace(original, replacement))
    with pytest.raises(ContributionError, match=message):
        check_contributions(tmp_path)


def test_pack_class_decorator_substitution_is_rejected_before_import(
    tmp_path: Path,
) -> None:
    source = _owner(tmp_path, "northstar")
    pack = source / "pack.py"
    text = pack.read_text()
    pack.write_text(
        "from dataclasses import replace\n"
        "def forge(cls):\n"
        "    class Generated(cls):\n"
        "        metadata = replace(cls.metadata, contribution_kind='maintainer', "
        "owner_display_name='Forged Ltd')\n"
        "    return Generated\n"
        + text.replace("class SamplePack:", "@forge\nclass SamplePack:")
    )
    with pytest.raises(ContributionError, match="decorator"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    "class_header",
    [
        "class SamplePack(object):",
        "class SamplePack(Base):",
        "class SamplePack(metaclass=Meta):",
    ],
)
def test_pack_class_rejects_bases_and_class_keywords(
    tmp_path: Path, class_header: str
) -> None:
    source = _owner(tmp_path, "northstar")
    pack = source / "pack.py"
    pack.write_text(pack.read_text().replace("class SamplePack:", class_header))
    with pytest.raises(ContributionError, match=r"base|keyword|metaclass"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    "method",
    ["__new__", "__init__", "__init_subclass__", "__setattr__", "__getattribute__"],
)
def test_pack_class_rejects_construction_and_metadata_interception(
    tmp_path: Path, method: str
) -> None:
    source = _owner(tmp_path, "northstar")
    pack = source / "pack.py"
    pack.write_text(
        pack.read_text().replace(
            "    metadata =", f"    def {method}(self, *args): pass\n    metadata ="
        )
    )
    with pytest.raises(ContributionError, match="interception"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    "snippet",
    [
        "SamplePack.metadata = SamplePack.metadata\n",
        "SamplePack = type('SamplePack', (), {})\n",
        "setattr(SamplePack, 'metadata', SamplePack.metadata)\n",
        "if True:\n    SamplePack = object\n",
        "for _ in ():\n    pass\n",
        "try:\n    pass\nexcept Exception:\n    pass\n",
        "launder(SamplePack)\n",
    ],
)
def test_pack_module_rejects_post_definition_execution_surface(
    tmp_path: Path, snippet: str
) -> None:
    source = _owner(tmp_path, "northstar")
    pack = source / "pack.py"
    pack.write_text(
        pack.read_text().replace("PACK = SamplePack()", snippet + "PACK = SamplePack()")
    )
    with pytest.raises(ContributionError, match=r"module|mutation|binding"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    ("snippet", "message"),
    [
        (
            "import operatebench.contributions.riverside.sample as x\n",
            "bare operatebench",
        ),
        ("from ...riverside import sample\n", "cross-owner"),
        ("from ..other import secret\n", "cross-operation"),
        ("import operatebench.domains.lettings.maintenance\n", "bare operatebench"),
        ("from importlib import import_module as load\nload('x')\n", "dynamic loading"),
        ("from importlib.machinery import SourceFileLoader\n", "dynamic loading"),
        ("from builtins import __import__ as load\nload('x')\n", "dynamic execution"),
        ("compile('1','x','exec')\n", "dynamic execution"),
    ],
)
def test_import_policy_bypasses(tmp_path: Path, snippet: str, message: str) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "extra.py").write_text(snippet)
    with pytest.raises(ContributionError, match=message):
        check_contributions(tmp_path)


@pytest.mark.parametrize("location", ["source", "test"])
@pytest.mark.parametrize(
    "snippet",
    [
        "__builtins__['eval']('1')\n",
        "getattr(__builtins__, 'eval')('1')\n",
        "vars(__builtins__)['eval']('1')\n",
        "__builtins__ = {}\n",
        "__builtins__['eval'] = None\n",
        "del __builtins__['eval']\n",
    ],
)
def test_exact_builtins_namespace_is_forbidden(
    tmp_path: Path, location: str, snippet: str
) -> None:
    source = _owner(tmp_path, "northstar")
    path = (
        source / "extra.py"
        if location == "source"
        else tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    )
    path.write_text(snippet)
    with pytest.raises(ContributionError, match="__builtins__"):
        check_contributions(tmp_path)


@pytest.mark.parametrize("location", ["source", "test"])
@pytest.mark.parametrize(
    "snippet",
    [
        "from dataclasses import replace\n"
        "from .pack import PACK\n"
        "PACK.__dict__['metadata'] = "
        "replace(PACK.metadata, display_name='FORGED')\n",
        "def hostile(PACK):\n    del PACK.__dict__['metadata']\n",
        "def hostile(PACK):\n    PACK.__dict__.update(metadata=None)\n",
        "def hostile(PACK):\n    vars(PACK)['metadata'] = None\n",
    ],
)
def test_object_namespace_mutation_is_forbidden(
    tmp_path: Path, location: str, snippet: str
) -> None:
    source = _owner(tmp_path, "northstar")
    path = (
        source / "__init__.py"
        if location == "source"
        else tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    )
    path.write_text(snippet)
    with pytest.raises(
        ContributionError, match=r"object namespace mutation|namespace escape"
    ):
        check_contributions(tmp_path)


def test_computed_dunder_dict_reflection_cannot_mutate_metadata(tmp_path: Path) -> None:
    source = _owner(tmp_path, "northstar")
    path = source / "pack.py"
    path.write_text(
        path.read_text().replace(
            "class SamplePack:\n",
            "class SamplePack:\n"
            "    def poison(self, value):\n"
            "        getattr(self, '__' + 'dict__')['metadata'] = value\n",
        )
    )
    with pytest.raises(ContributionError, match="reflection primitive 'getattr'"):
        check_contributions(tmp_path)


@pytest.mark.parametrize("location", ["source", "test"])
@pytest.mark.parametrize(
    "snippet",
    [
        "def hostile(value):\n    return getattr(value, '__' + 'dict__')\n",
        "def hostile(value):\n    setattr(value, 'meta' + 'data', None)\n",
        "def hostile(value):\n    delattr(value, 'meta' + 'data')\n",
        "g = getattr\n",
        "def hostile(value):\n    return object.__getattribute__(value, '__dict__')\n",
        "def hostile(value):\n    return value.__getattribute__('__dict__')\n",
        "def hostile(value):\n    object.__setattr__(value, 'metadata', None)\n",
        "def hostile(value):\n    value.__setattr__('metadata', None)\n",
        "def hostile(value):\n    object.__delattr__(value, 'metadata')\n",
        "def hostile(value):\n    value.__delattr__('metadata')\n",
    ],
)
def test_dynamic_reflection_is_forbidden(
    tmp_path: Path, location: str, snippet: str
) -> None:
    source = _owner(tmp_path, "northstar")
    path = (
        source / "extra.py"
        if location == "source"
        else tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    )
    path.write_text(snippet)
    with pytest.raises(ContributionError, match="reflection primitive"):
        check_contributions(tmp_path)


@pytest.mark.parametrize("location", ["source", "test"])
def test_pack_metadata_assignment_is_forbidden_across_contribution_files(
    tmp_path: Path, location: str
) -> None:
    source = _owner(tmp_path, "northstar")
    path = (
        source / "operation.py"
        if location == "source"
        else tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    )
    path.write_text(
        "from operatebench.sdk import OperationPackMetadata\n"
        "from .pack import PACK\n"
        "PACK.metadata = OperationPackMetadata(\n"
        "    pack_id='partner.northstar.sample',\n"
        "    operation_type='northstar.sample.synthetic',\n"
        "    operation_id='northstar_sample', pack_version='9.9.9',\n"
        "    display_name='FORGED', owner_id='northstar',\n"
        "    owner_display_name='Northstar Fictional Company',\n"
        "    contribution_kind='partner', status='incubator',\n"
        "    privacy_status='SYNTHETIC_ONLY',\n"
        "    default_spec='examples/contributions/northstar/sample/operation.yaml',\n"
        "    agent_ids=('reference', 'forged'), evidence_eligible=False, aliases=(),\n"
        ")\n"
    )
    with pytest.raises(ContributionError, match="reserved metadata mutation"):
        check_contributions(tmp_path)


@pytest.mark.parametrize("location", ["source", "test"])
@pytest.mark.parametrize(
    "statement",
    [
        "del target.metadata",
        "setattr(target, 'metadata', None)",
        "delattr(target, 'metadata')",
        "target.__setattr__('metadata', None)",
        "object.__setattr__(target, 'metadata', None)",
        "target.__delattr__('metadata')",
        "object.__delattr__(target, 'metadata')",
    ],
)
def test_metadata_mutation_variants_are_forbidden_across_contribution_files(
    tmp_path: Path, location: str, statement: str
) -> None:
    source = _owner(tmp_path, "northstar")
    path = (
        source / "operation.py"
        if location == "source"
        else tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    )
    path.write_text(f"def admitted_context(target):\n    {statement}\n")
    with pytest.raises(
        ContributionError, match=r"reserved metadata mutation|reflection primitive"
    ):
        check_contributions(tmp_path)


@pytest.mark.parametrize("location", ["source", "test"])
@pytest.mark.parametrize("primitive", ["setattr", "delattr"])
def test_mutation_primitive_alias_sources_are_forbidden(
    tmp_path: Path, location: str, primitive: str
) -> None:
    source = _owner(tmp_path, "northstar")
    path = (
        source / "operation.py"
        if location == "source"
        else tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    )
    path.write_text(f"def admitted_context():\n    mutator = {primitive}\n")
    with pytest.raises(ContributionError, match="reflection primitive"):
        check_contributions(tmp_path)


@pytest.mark.parametrize("location", ["source", "test"])
def test_metadata_reads_and_ordinary_attribute_mutation_remain_allowed(
    tmp_path: Path, location: str
) -> None:
    source = _owner(tmp_path, "northstar")
    path = (
        source / "operation.py"
        if location == "source"
        else tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    )
    path.write_text(
        "from .pack import PACK\n"
        "def admitted_context(state):\n"
        "    metadata = PACK.metadata\n"
        "    state.value = metadata.display_name\n"
        "    return metadata\n"
    )
    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


def test_ordinary_mapping_mutation_remains_allowed(tmp_path: Path) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "extra.py").write_text(
        "def mutate(mapping):\n"
        "    mapping['key'] = 'value'\n"
        "    del mapping['obsolete']\n"
        "    mapping.update(fresh=True)\n"
        "    return mapping\n"
    )
    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


def test_read_only_dunder_dict_items_remains_allowed(tmp_path: Path) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "state.py").write_text(
        "class State:\n"
        "    def public(self):\n"
        "        return {name: value for name, value in self.__dict__.items()}\n"
    )
    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


@pytest.mark.parametrize(
    "snippet",
    [
        "SamplePack.__module__ = 'operatebench.contributions.northstar.sample.pack'\n",
        "SamplePack.__class__ = type\n",
        "__name__ = 'laundered'\n",
        "__package__: str = 'laundered'\n",
        "PACK = SamplePack()\nPACK = SamplePack()\n",
        "PACK = SamplePack()\nPACK += SamplePack()\n",
        "PACK = SamplePack()\ndel PACK\n",
    ],
)
def test_nominal_binding_laundering_is_rejected(tmp_path: Path, snippet: str) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "pack.py").write_text(
        (source / "pack.py").read_text().replace("PACK = SamplePack()\n", snippet)
    )
    with pytest.raises(ContributionError, match=r"binding|mutation|PACK"):
        check_contributions(tmp_path)


@pytest.mark.parametrize("location", ["source", "test"])
@pytest.mark.parametrize(
    "snippet",
    [
        "import sys as s\ns.modules['laundered'] = object()\n",
        "from sys import modules as m\nm['laundered'] = object()\n",
        "import sys\nm = sys.modules\nm['laundered'] = object()\n",
        "import sys\ns = sys\ns.modules['laundered'] = object()\n",
        "from sys import modules\nm = modules\nm['laundered'] = object()\n",
        "import sys.audit\n",
        "from sys import version_info\n",
        "from os import sys as s\n",
        "import os\ns = os.sys\n",
    ],
)
def test_sys_imports_are_forbidden_before_module_state_can_escape(
    tmp_path: Path, location: str, snippet: str
) -> None:
    source = _owner(tmp_path, "northstar")
    path = (
        source / "pack.py"
        if location == "source"
        else tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    )
    if location == "source":
        path.write_text(
            path.read_text().replace(
                "PACK = SamplePack()", snippet + "PACK = SamplePack()"
            )
        )
    else:
        path.write_text(snippet)
    with pytest.raises(
        ContributionError,
        match=r"interpreter-global module 'sys' is forbidden",
    ):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    "declaration",
    [
        "PACK = OtherPack()",
        "PACK = SamplePack(1)",
        "PACK = SamplePack(value=1)",
        "PACK: object = SamplePack()",
    ],
)
def test_pack_must_instantiate_exact_metadata_owner(
    tmp_path: Path, declaration: str
) -> None:
    source = _owner(tmp_path, "northstar")
    pack = source / "pack.py"
    pack.write_text(pack.read_text().replace("PACK = SamplePack()", declaration))
    with pytest.raises(ContributionError, match="PACK"):
        check_contributions(tmp_path)


@pytest.mark.parametrize("location", ["source", "test"])
@pytest.mark.parametrize(
    ("snippet", "message"),
    [
        (
            "from operatebench.contributions.northstar import other_operation\n",
            "cross-operation",
        ),
        ("from .. import other_operation\n", "cross-operation"),
        ("from ..other.deep import secret\n", "cross-operation"),
        ("from ... import riverside\n", "cross-owner"),
        ("from operatebench import domains\n", "domain pack"),
        ("from operatebench.domains import lettings\n", "domain pack"),
        ("from operatebench import contributions\n", "cross-owner"),
        ("from operatebench.contributions import riverside\n", "cross-owner"),
        ("from importlib import abc\n", "dynamic loading"),
        ("from runpy import _run_module_code\n", "dynamic loading"),
        ("from pkgutil import iter_modules\n", "dynamic loading"),
        ("from builtins import open\n", "dynamic execution"),
        ("from .agents import *\n", "star import"),
        ("from ...... import os\n", "beyond top-level package"),
    ],
)
def test_import_from_targets_are_checked_exhaustively(
    tmp_path: Path, location: str, snippet: str, message: str
) -> None:
    source = _owner(tmp_path, "northstar")
    path = (
        source / "extra.py"
        if location == "source"
        else tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    )
    path.write_text(snippet)
    with pytest.raises(ContributionError, match=message):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    "snippet",
    [
        "from . import agents\n",
        "from .agents import reference_events\n",
        "from pathlib import Path\n",
        "from operatebench.sdk import OperationPackMetadata\n",
    ],
)
def test_import_from_same_operation_stdlib_and_sdk_controls(
    tmp_path: Path, snippet: str
) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "extra.py").write_text(snippet)
    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


@pytest.mark.parametrize("helper", ["util", "loader", "machinery"])
@pytest.mark.parametrize(
    "consumer",
    ["source", "owner_test_source_helper", "owner_test_helper"],
)
def test_same_operation_helper_module_names_are_allowed(
    tmp_path: Path, helper: str, consumer: str
) -> None:
    source = _owner(tmp_path, "northstar")
    test_root = tmp_path / "tests/contributions/northstar/sample"
    helper_root = test_root if consumer == "owner_test_helper" else source
    (helper_root / f"{helper}.py").write_text("VALUE = 1\n")
    if consumer == "source":
        (source / "extra.py").write_text(f"from .{helper} import VALUE\n")
    elif consumer == "owner_test_source_helper":
        (test_root / "test_pack.py").write_text(
            f"from operatebench.contributions.northstar.sample.{helper} import VALUE\n"
        )
    else:
        (test_root / "test_pack.py").write_text(f"from .{helper} import VALUE\n")

    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


@pytest.mark.parametrize("helper", ["utility", "myloader", "remachinery"])
def test_module_names_merely_containing_dynamic_loading_words_are_allowed(
    tmp_path: Path, helper: str
) -> None:
    source = _owner(tmp_path, "northstar")
    (source / f"{helper}.py").write_text("VALUE = 1\n")
    (source / "extra.py").write_text(f"from .{helper} import VALUE\n")

    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


@pytest.mark.parametrize(
    ("module", "member"),
    [
        ("importlib", "abc"),
        ("importlib.util", "find_spec"),
        ("importlib.machinery", "SourceFileLoader"),
        ("importlib.abc", "Loader"),
        ("runpy", "run_path"),
        ("pkgutil", "iter_modules"),
    ],
)
@pytest.mark.parametrize("location", ["source", "test"])
@pytest.mark.parametrize("import_style", ["module_alias", "from_alias"])
def test_dynamic_loading_module_prefixes_remain_forbidden(
    tmp_path: Path, module: str, member: str, location: str, import_style: str
) -> None:
    source = _owner(tmp_path, "northstar")
    path = (
        source / "extra.py"
        if location == "source"
        else tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    )
    snippet = (
        f"import {module} as harmless_name\n"
        if import_style == "module_alias"
        else f"from {module} import {member} as harmless_name\n"
    )
    path.write_text(snippet)

    with pytest.raises(ContributionError, match="dynamic loading"):
        check_contributions(tmp_path)


@pytest.mark.parametrize("helper", ["util", "loader", "machinery"])
@pytest.mark.parametrize(
    "snippet",
    [
        "from operatebench.contributions.riverside.sample.{helper} import VALUE\n",
        "from operatebench.contributions.northstar.other.{helper} import VALUE\n",
    ],
)
def test_helper_module_names_do_not_bypass_ownership_boundaries(
    tmp_path: Path, helper: str, snippet: str
) -> None:
    source = _owner(tmp_path, "northstar")
    _owner(tmp_path, "riverside")
    _owner(tmp_path, "northstar", operation="other")
    (source / "extra.py").write_text(snippet.format(helper=helper))

    with pytest.raises(ContributionError, match=r"cross-(owner|operation)"):
        check_contributions(tmp_path)


@pytest.mark.parametrize("location", ["source", "test"])
@pytest.mark.parametrize(
    "module",
    [
        "operatebench.sdk.builtins",
        "operatebench.sdk.registry",
        "operatebench.core.engine",
        "operatebench.unknown",
    ],
)
def test_non_allowlisted_operatebench_modules_are_rejected(
    tmp_path: Path, location: str, module: str
) -> None:
    source = _owner(tmp_path, "northstar")
    path = (
        source / "extra.py"
        if location == "source"
        else tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    )
    path.write_text(f"from {module} import MUTABLE\nMUTABLE.clear()\n")

    with pytest.raises(ContributionError, match="not supported in contribution"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    ("module", "imported_name"),
    [
        ("operatebench._resource_access", "files, as_file"),
        ("operatebench._write_once", "_write_once_bytes, _WriteOnceFailure"),
        ("operatebench.core.errors", "SpecSchemaError"),
        ("operatebench.core.evaluation", "OperationEvaluation"),
        ("operatebench.core.outcomes", "AgentOutcome"),
        ("operatebench.core.protocol", "AgentObservation"),
        ("operatebench.core.read_contract", "ReadRequirementContract"),
        ("operatebench.core.retrieval", "RetrieveBatch"),
        ("operatebench.sdk", "OperationPackMetadata"),
        ("operatebench.sdk.errors", "OperationPackMismatchError"),
    ],
)
def test_supported_source_operatebench_module_list_is_finite(
    tmp_path: Path, module: str, imported_name: str
) -> None:
    source = _owner(tmp_path, "northstar")
    snippet = f"from {module} import {imported_name}\n"
    if module == "operatebench._resource_access":
        snippet = (
            "from operatebench._resource_access import files, as_file\n"
            "with as_file(files(__package__).joinpath('fixture.yaml')):\n"
            "    pass\n"
        )
        (source / "fixture.yaml").write_text("value: synthetic\n")
    elif module == "operatebench._write_once":
        snippet = (
            "from operatebench._write_once import _write_once_bytes, _WriteOnceFailure\n"
        )
    (source / "extra.py").write_text(snippet)

    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


@pytest.mark.parametrize(
    ("module", "imported_name"),
    [
        ("operatebench.cli", "main"),
        ("operatebench.core.errors", "SpecSchemaError"),
        ("operatebench.sdk", "CheckRequest"),
    ],
)
def test_supported_test_operatebench_module_list_is_finite(
    tmp_path: Path, module: str, imported_name: str
) -> None:
    _owner(tmp_path, "northstar")
    test = tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    test.write_text(f"from {module} import {imported_name}\n")

    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


APPROVED_OPERATEBENCH_SYMBOLS = {
    "operatebench.cli": ("main",),
    "operatebench.core.errors": (
        "AgentRegistryError",
        "ArtifactError",
        "OracleManifestError",
        "SpecSchemaError",
    ),
    "operatebench.core.evaluation": ("OperationEvaluation",),
    "operatebench.core.outcomes": ("AgentOutcome",),
    "operatebench.core.protocol": (
        "AgentObservation",
        "EnvironmentContext",
        "EpisodePlan",
        "Verdict",
    ),
    "operatebench.core.read_contract": ("ReadRequirementContract",),
    "operatebench.core.retrieval": ("RetrievalRequest", "RetrieveBatch", "ToolResult"),
    "operatebench.sdk": (
        "CheckRequest",
        "CommandResult",
        "OperationPackError",
        "OperationPackMetadata",
        "ReplayRequest",
        "RunRequest",
        "ValidateRequest",
    ),
    "operatebench.sdk.errors": ("OperationPackMismatchError",),
}


@pytest.mark.parametrize(("module", "symbols"), APPROVED_OPERATEBENCH_SYMBOLS.items())
def test_every_template_and_prospire_public_import_symbol_is_allowed(
    tmp_path: Path, module: str, symbols: tuple[str, ...]
) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "extra.py").write_text(f"from {module} import {', '.join(symbols)}\n")
    if module == "operatebench.cli":
        test = tmp_path / "tests/contributions/northstar/sample/test_pack.py"
        test.write_text(f"from {module} import {', '.join(symbols)}\n")
        (source / "extra.py").unlink()

    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


@pytest.mark.parametrize(
    ("module", "symbols"),
    [
        ("operatebench.cli", ("main",)),
        (
            "operatebench.core.errors",
            (
                "AgentRegistryError",
                "ArtifactError",
                "OracleManifestError",
                "SpecSchemaError",
            ),
        ),
        ("operatebench.sdk", ("CheckRequest", "ReplayRequest", "RunRequest")),
    ],
)
def test_every_real_prospire_test_public_import_symbol_is_allowed(
    tmp_path: Path, module: str, symbols: tuple[str, ...]
) -> None:
    _owner(tmp_path, "northstar")
    test = tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    test.write_text(f"from {module} import {', '.join(symbols)}\n")

    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


@pytest.mark.parametrize("location", ["source", "test"])
@pytest.mark.parametrize("module", APPROVED_OPERATEBENCH_SYMBOLS)
def test_unknown_symbol_or_submodule_fails_closed_on_every_public_surface(
    tmp_path: Path, location: str, module: str
) -> None:
    source = _owner(tmp_path, "northstar")
    path = (
        source / "extra.py"
        if location == "source"
        else tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    )
    path.write_text(f"from {module} import unknown_submodule as value\n")

    with pytest.raises(ContributionError):
        check_contributions(tmp_path)


@pytest.mark.parametrize("location", ["source", "test"])
@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize(
    "leaf", ["builtins", "registry", "api", "scaffold", "_validation"]
)
def test_sdk_from_list_submodules_are_rejected(
    tmp_path: Path, location: str, alias: bool, leaf: str
) -> None:
    source = _owner(tmp_path, "northstar")
    path = (
        source / "extra.py"
        if location == "source"
        else tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    )
    imported = f"{leaf} as target" if alias else leaf
    path.write_text(
        f"from operatebench.sdk import {imported}\n"
        + ("target.BUILTIN_PACKS = ()\n" if alias else "")
    )

    with pytest.raises(ContributionError, match="symbol"):
        check_contributions(tmp_path)


@pytest.mark.parametrize("location", ["source", "test"])
def test_one_forbidden_sdk_name_rejects_multiple_import(
    tmp_path: Path, location: str
) -> None:
    source = _owner(tmp_path, "northstar")
    path = (
        source / "extra.py"
        if location == "source"
        else tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    )
    path.write_text("from operatebench.sdk import CheckRequest, builtins as target\n")

    with pytest.raises(ContributionError, match="symbol"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    "snippet",
    [
        "from operatebench._write_once import _write_once_bytes\n",
        "from operatebench._write_once import _WriteOnceFailure\n",
        "from operatebench._write_once import _write_once_bytes, Other\n",
        "from operatebench._write_once import "
        "_write_once_bytes as write, _WriteOnceFailure\n",
    ],
)
def test_write_once_import_requires_exact_helper_names(
    tmp_path: Path, snippet: str
) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "extra.py").write_text(snippet)

    with pytest.raises(ContributionError, match="write-once helper import must be exact"):
        check_contributions(tmp_path)


def test_declared_test_may_import_only_its_operation_package(tmp_path: Path) -> None:
    _owner(tmp_path, "northstar")
    test = tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    test.write_text("from operatebench.contributions.northstar.sample import pack\n")
    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)
    test.write_text("from operatebench.contributions.northstar import other\n")
    with pytest.raises(ContributionError, match="cross-operation"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    "snippet",
    [
        "from tests.contributions.northstar.sample.test_pack import helper\n",
        "from tests.contributions.northstar.other.test_pack import helper\n",
        "from tests.contributions.riverside.sample.test_pack import helper\n",
        "from tests.helper import helper\n",
    ],
)
def test_contribution_source_cannot_import_tests_namespace(
    tmp_path: Path, snippet: str
) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "extra.py").write_text(snippet)

    with pytest.raises(ContributionError, match="tests namespace"):
        check_contributions(tmp_path)


def test_contribution_test_may_import_own_test_helpers(tmp_path: Path) -> None:
    _owner(tmp_path, "northstar")
    test_root = tmp_path / "tests/contributions/northstar/sample"
    (test_root / "helper.py").write_text("VALUE = 1\n")
    (test_root / "test_pack.py").write_text(
        "from tests.contributions.northstar.sample.helper import VALUE\n"
        "from .helper import VALUE as RELATIVE_VALUE\n"
    )

    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


@pytest.mark.parametrize(
    ("snippet", "message"),
    [
        (
            "from tests.contributions.northstar.other import helper\n",
            "cross-operation",
        ),
        (
            "from tests.contributions.riverside.sample import helper\n",
            "cross-owner",
        ),
        ("from tests.helper import helper\n", "test namespace"),
    ],
)
def test_contribution_test_cannot_import_outside_own_test_namespace(
    tmp_path: Path, snippet: str, message: str
) -> None:
    _owner(tmp_path, "northstar")
    test = tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    test.write_text(snippet)

    with pytest.raises(ContributionError, match=message):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    ("filename", "snippet", "message"),
    [
        (
            "helper.py",
            "import operatebench.contributions.riverside.sample\n",
            "bare operatebench",
        ),
        (
            "test_second.py",
            "import operatebench.domains.lettings.maintenance\n",
            "bare operatebench",
        ),
        ("helper.py", "import importlib\n", "dynamic loading"),
    ],
)
def test_undeclared_test_sibling_is_scanned(
    tmp_path: Path, filename: str, snippet: str, message: str
) -> None:
    _owner(tmp_path, "northstar")
    sibling = tmp_path / f"tests/contributions/northstar/sample/{filename}"
    sibling.write_text(snippet)
    with pytest.raises(ContributionError, match=message):
        check_contributions(tmp_path)


def test_nested_packages_use_their_actual_current_module(tmp_path: Path) -> None:
    source = _owner(tmp_path, "northstar")
    nested = source / "sub"
    nested.mkdir()
    (nested / "__init__.py").write_text("from . import helper\n")
    (nested / "helper.py").write_text("from .. import pack\n")
    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


@pytest.mark.parametrize(
    ("snippet", "message"),
    [
        ("from ...other import secret\n", "cross-operation"),
        ("from ....riverside import sample\n", "cross-owner"),
    ],
)
def test_nested_packages_reject_imports_outside_the_operation(
    tmp_path: Path, snippet: str, message: str
) -> None:
    source = _owner(tmp_path, "northstar")
    nested = source / "sub"
    nested.mkdir()
    (nested / "__init__.py").write_text(snippet)
    with pytest.raises(ContributionError, match=message):
        check_contributions(tmp_path)


@pytest.mark.parametrize("area", ["source", "test"])
@pytest.mark.parametrize(
    "name", ["payload.pyc", "payload.pyo", "payload.so", "payload.pyd", ".cache"]
)
def test_unscanned_or_hidden_operation_entries_are_rejected(
    tmp_path: Path, area: str, name: str
) -> None:
    source = _owner(tmp_path, "northstar")
    root = (
        source if area == "source" else tmp_path / "tests/contributions/northstar/sample"
    )
    (root / name).write_bytes(b"binary\x00payload")
    with pytest.raises(ContributionError, match=r"forbidden|suffix|hidden"):
        check_contributions(tmp_path)


def _cache_root(tmp_path: Path, source: Path, area: str) -> Path:
    return {
        "contribution": tmp_path / "src/operatebench/contributions",
        "owner": source.parent,
        "source": source,
        "shared_test": tmp_path / "tests/contributions",
        "owner_test": tmp_path / "tests/contributions/northstar",
        "operation_test": tmp_path / "tests/contributions/northstar/sample",
    }[area]


_CACHE_AREAS = (
    "contribution",
    "owner",
    "source",
    "shared_test",
    "owner_test",
    "operation_test",
)


@pytest.mark.parametrize("area", _CACHE_AREAS)
def test_canonical_bytecode_cache_files_are_accepted(tmp_path: Path, area: str) -> None:
    source = _owner(tmp_path, "northstar")
    probe = tmp_path / "ordinary_cache_probe.py"
    probe.write_text("VALUE = 1\n")
    child_env = os.environ.copy()
    child_env.pop("PYTHONDONTWRITEBYTECODE", None)
    child_env.pop("PYTHONPYCACHEPREFIX", None)
    subprocess.run(
        [sys.executable, "-c", "import ordinary_cache_probe"],
        cwd=tmp_path,
        check=True,
        env=child_env,
    )
    generated = next((tmp_path / "__pycache__").iterdir())
    cache = _cache_root(tmp_path, source, area) / "__pycache__"
    cache.mkdir()
    (cache / generated.name).write_bytes(generated.read_bytes())
    (cache / "module.cpython-311.pyc").write_bytes(b"opaque bytecode")
    (cache / "module.cpython-314.opt-2.pyc").write_bytes(b"opaque bytecode")
    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


@pytest.mark.parametrize("area", _CACHE_AREAS)
@pytest.mark.parametrize(
    "mutant",
    [
        "evil.py",
        "native.so",
        "native.pyd",
        "__init__.py",
        ".hidden",
        "arbitrary.pyc",
        "subdir",
        "nested_cache",
        "symlink",
        "special",
    ],
)
def test_noncanonical_bytecode_cache_entries_are_rejected(
    tmp_path: Path, area: str, mutant: str
) -> None:
    source = _owner(tmp_path, "northstar")
    cache = _cache_root(tmp_path, source, area) / "__pycache__"
    cache.mkdir()
    entry = cache / mutant
    if mutant in {"subdir", "nested_cache"}:
        entry.mkdir()
        if mutant == "nested_cache":
            (entry / "__pycache__").mkdir()
    elif mutant == "symlink":
        entry.symlink_to("missing-target")
    elif mutant == "special":
        os.mkfifo(entry)
    else:
        entry.write_bytes(b"authored payload")
    with pytest.raises(ContributionError, match="non-canonical"):
        check_contributions(tmp_path)


@pytest.mark.parametrize("area", _CACHE_AREAS)
def test_symlinked_bytecode_cache_is_rejected(tmp_path: Path, area: str) -> None:
    source = _owner(tmp_path, "northstar")
    root = _cache_root(tmp_path, source, area)
    target = tmp_path / f"{area}-generated-cache"
    target.mkdir()
    (root / "__pycache__").symlink_to(target, target_is_directory=True)
    with pytest.raises(ContributionError, match="symlink"):
        check_contributions(tmp_path)


def test_owner_root_bytecode_file_is_rejected(tmp_path: Path) -> None:
    source = _owner(tmp_path, "northstar")
    (source.parent / "__init__.cpython-313.pyc").write_bytes(b"generated\x00bytecode")
    with pytest.raises(ContributionError, match="allowed regular file"):
        check_contributions(tmp_path)


@pytest.mark.parametrize("area", ["source", "test"])
def test_public_utf8_resources_are_accepted(tmp_path: Path, area: str) -> None:
    source = _owner(tmp_path, "northstar")
    root = (
        source if area == "source" else tmp_path / "tests/contributions/northstar/sample"
    )
    for suffix in ("yaml", "yml", "json", "md", "txt"):
        (root / f"resource.{suffix}").write_text("synthetic resource\n")
    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


@pytest.mark.parametrize("area", ["source", "test"])
def test_binary_public_resource_is_rejected(tmp_path: Path, area: str) -> None:
    source = _owner(tmp_path, "northstar")
    root = (
        source if area == "source" else tmp_path / "tests/contributions/northstar/sample"
    )
    (root / "resource.txt").write_bytes(b"\xff\x00")
    with pytest.raises(ContributionError, match="UTF-8"):
        check_contributions(tmp_path)


@pytest.mark.parametrize("area", ["source", "test"])
def test_symlink_anywhere_in_scanned_trees_is_rejected(tmp_path: Path, area: str) -> None:
    source = _owner(tmp_path, "northstar")
    root = (
        source if area == "source" else tmp_path / "tests/contributions/northstar/sample"
    )
    target = root / "target.txt"
    target.write_text("synthetic\n")
    (root / "linked.txt").symlink_to(target.name)
    with pytest.raises(ContributionError, match="symlink"):
        check_contributions(tmp_path)


def test_declared_test_path_must_be_python(tmp_path: Path) -> None:
    source = _owner(tmp_path, "northstar")
    test_root = tmp_path / "tests/contributions/northstar/sample"
    (test_root / "declared.txt").write_text("synthetic\n")
    manifest = source / "CONTRIBUTION.yaml"
    manifest.write_text(manifest.read_text().replace("test_pack.py", "declared.txt"))
    with pytest.raises(ContributionError, match=r"regular \.py"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    "snippet",
    [
        "import builtins as b\nb.eval('1')\n",
        "x = eval\nx('1')\n",
        "from . import importlib\n",
        "from .. import runpy\n",
        "from .importlib import import_module\n",
    ],
)
def test_static_dynamic_code_aliases_are_rejected(tmp_path: Path, snippet: str) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "extra.py").write_text(snippet)
    with pytest.raises(ContributionError, match=r"dynamic (loading|execution)"):
        check_contributions(tmp_path)


def test_harmless_ordinary_calls_are_allowed(tmp_path: Path) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "extra.py").write_text(
        "def identity(value): return value\nx = identity\nx(1)\n"
    )
    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


def test_global_collisions(tmp_path: Path) -> None:
    _owner(tmp_path, "northstar", operation_type="shared.operation.type")
    _owner(tmp_path, "riverside", operation_type="shared.operation.type")
    with pytest.raises(ContributionError, match="duplicate operation_type"):
        check_contributions(tmp_path)
    other = tmp_path / "display"
    _owner(other, "northstar", display_name="Shared Corp")
    _owner(other, "riverside", display_name="SHARED CORP")
    with pytest.raises(ContributionError, match="display name"):
        check_contributions(other)


def test_declared_file_cannot_be_directory(tmp_path: Path) -> None:
    source = _owner(tmp_path, "northstar")
    manifest = source / "CONTRIBUTION.yaml"
    manifest.write_text(
        manifest.read_text().replace(
            "docs/contributions/northstar/sample.md", "docs/contributions/northstar"
        )
    )
    with pytest.raises(ContributionError, match="regular file"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    ("area", "field"),
    [
        ("tests", "test_path"),
        ("examples", "fixture_path"),
        ("docs", "documentation_path"),
    ],
)
@pytest.mark.parametrize("destination", ["other-owner", "outside"])
def test_declared_paths_reject_symlinked_owner_ancestors(
    tmp_path: Path, area: str, field: str, destination: str
) -> None:
    _owner(tmp_path, "northstar")
    owner_dir = tmp_path / area / "contributions/northstar"
    real = owner_dir.with_name(destination)
    owner_dir.rename(real)
    if destination == "outside":
        outside = tmp_path / f"outside-{area}"
        real.rename(outside)
        real = outside
    owner_dir.symlink_to(real, target_is_directory=True)
    expected = (
        "shared test root.*symlink"
        if field == "test_path"
        else rf"{field}.*symlink|shared (?:example|documentation) root entry is forbidden"
    )
    with pytest.raises(ContributionError, match=expected):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    "target", ["OWNER.yaml", "sample/CONTRIBUTION.yaml", "sample/pack.py"]
)
@pytest.mark.parametrize("kind", ["symlink", "oversize", "sparse"])
def test_bounded_reads(tmp_path: Path, target: str, kind: str) -> None:
    source = _owner(tmp_path, "northstar")
    path = source.parent / target
    if kind == "symlink":
        copy = path.with_suffix(path.suffix + ".real")
        path.rename(copy)
        path.symlink_to(copy.name)
    else:
        with path.open("wb") as stream:
            if kind == "sparse":
                stream.seek(2_000_000)
            stream.write(b"x" * (2_000_000 if kind == "oversize" else 1))
    with pytest.raises(ContributionError, match=r"symlink|regular|limit|large"):
        check_contributions(tmp_path)


def test_short_read_is_named(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _owner(tmp_path, "northstar")
    real_read = os.read
    calls = 0

    def short(fd: int, count: int) -> bytes:
        nonlocal calls
        calls += 1
        return b"" if calls == 1 else real_read(fd, count)

    monkeypatch.setattr(os, "read", short)
    with pytest.raises(ContributionError, match="short read"):
        check_contributions(tmp_path)


def test_bounded_read_rejects_byte_mismatch_with_stable_stats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "stable-size"
    path.write_bytes(b"AAAA")
    real_read = os.read
    second_pass = False
    returned_replacement = False

    def seek(fd: int, offset: int, whence: int) -> int:
        nonlocal second_pass
        assert offset == 0
        assert whence == os.SEEK_SET
        second_pass = True
        return 0

    def replaced_read(fd: int, count: int) -> bytes:
        nonlocal returned_replacement
        if second_pass:
            if returned_replacement:
                return b""
            returned_replacement = True
            return b"BBBB"[:count]
        return real_read(fd, count)

    monkeypatch.setattr(os, "lseek", seek)
    monkeypatch.setattr(os, "read", replaced_read)
    with pytest.raises(ContributionError, match="changed during bounded read"):
        isolation._read_bounded(path, 4)


def test_bounded_read_requires_three_validated_passes_and_fixed_quiescence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "three-passes"
    path.write_bytes(b"AAAA")
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

    monkeypatch.setattr(isolation.os, "fstat", traced_fstat)
    monkeypatch.setattr(isolation.os, "read", traced_read)
    monkeypatch.setattr(isolation.os, "lseek", traced_lseek)
    monkeypatch.setattr(isolation.time, "sleep", sleeps.append)

    assert isolation._read_bounded(path, 4) == b"AAAA"
    assert snapshots == 6
    assert reads == 6
    assert seeks == 2
    assert sleeps == [isolation.READ_QUIESCENCE_SECONDS] * 2


def test_bounded_read_rejects_third_pass_mutation_with_stable_stats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "third-pass-mutation"
    path.write_bytes(b"AAAA")
    real_read = os.read
    real_lseek = os.lseek
    pass_number = 1
    returned_replacement = False

    def seek(fd: int, offset: int, whence: int) -> int:
        nonlocal pass_number
        pass_number += 1
        return real_lseek(fd, offset, whence)

    def replaced_read(fd: int, count: int) -> bytes:
        nonlocal returned_replacement
        if pass_number == 3:
            if returned_replacement:
                return b""
            returned_replacement = True
            return b"BBBB"[:count]
        return real_read(fd, count)

    monkeypatch.setattr(isolation.os, "lseek", seek)
    monkeypatch.setattr(isolation.os, "read", replaced_read)
    monkeypatch.setattr(isolation.time, "sleep", lambda _: None)
    with pytest.raises(ContributionError, match="changed during bounded read"):
        isolation._read_bounded(path, 4)


@pytest.mark.parametrize("field", ["st_mode", "st_nlink", "st_mtime_ns", "st_ctime_ns"])
def test_bounded_read_rejects_metadata_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str
) -> None:
    path = tmp_path / "metadata-change"
    path.write_bytes(b"AAAA")
    stable = os.stat(path)
    changed = SimpleNamespace(
        st_mode=stable.st_mode,
        st_dev=stable.st_dev,
        st_ino=stable.st_ino,
        st_nlink=stable.st_nlink,
        st_size=stable.st_size,
        st_mtime_ns=stable.st_mtime_ns,
        st_ctime_ns=stable.st_ctime_ns,
    )
    setattr(changed, field, getattr(changed, field) + 1)
    calls = 0

    def changing_fstat(fd: int) -> object:
        nonlocal calls
        calls += 1
        return stable if calls == 1 else changed

    monkeypatch.setattr(os, "fstat", changing_fstat)
    with pytest.raises(ContributionError, match="changed during bounded read"):
        isolation._read_bounded(path, 4)


@pytest.mark.parametrize("field", ["st_mtime_ns", "st_ctime_ns"])
def test_bounded_read_requires_nanosecond_statistics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str
) -> None:
    path = tmp_path / "missing-nanoseconds"
    path.write_bytes(b"AAAA")
    observed = os.stat(path)
    incomplete = SimpleNamespace(
        **{
            name: getattr(observed, name)
            for name in (
                "st_mode",
                "st_dev",
                "st_ino",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
            if name != field
        }
    )
    monkeypatch.setattr(os, "fstat", lambda fd: incomplete)
    with pytest.raises(ContributionError, match=field):
        isolation._read_bounded(path, 4)


@pytest.mark.parametrize("operation", ["read", "lseek", "close"])
def test_bounded_read_translates_descriptor_oserrors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, operation: str
) -> None:
    path = tmp_path / "descriptor-error"
    path.write_bytes(b"AAAA")
    original = getattr(os, operation)

    def refuse(*args: object) -> object:
        if operation == "close":
            original(*args)
        raise OSError("descriptor sentinel")

    monkeypatch.setattr(os, operation, refuse)
    message = operation.replace("lseek", "seek")
    with pytest.raises(ContributionError, match=rf"cannot {message}"):
        isolation._read_bounded(path, 4)


def test_concurrent_equal_size_overwrite_is_canonical_or_fails_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "concurrent-overwrite"
    size = 256 * 1024
    values = (b"A" * size, b"B" * size)
    path.write_bytes(values[0])
    stop = threading.Event()

    def overwrite() -> None:
        index = 0
        while not stop.is_set():
            with path.open("r+b", buffering=0) as stream:
                stream.write(values[index])
            index ^= 1

    writer = threading.Thread(target=overwrite)
    writer.start()
    classified = 0
    try:
        for _ in range(80):
            try:
                result = isolation._read_bounded(path, size)
            except ContributionError:
                classified += 1
                continue
            assert result in values
            classified += 1
    finally:
        stop.set()
        writer.join(timeout=5)
    assert not writer.is_alive()
    assert classified == 80
    result = isolation._read_bounded(path, size)
    assert result in values


@pytest.mark.parametrize(
    ("module", "capability", "message"),
    [
        (os, "O_NOFOLLOW", "O_NOFOLLOW"),
        (os, "O_CLOEXEC", "O_CLOEXEC"),
        (os, "fstat", "os.fstat"),
        (os, "lseek", "os.lseek"),
        (os, "SEEK_SET", "SEEK_SET"),
        (stat, "S_ISREG", "stat.S_ISREG"),
    ],
)
def test_descriptor_capabilities_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module: object,
    capability: str,
    message: str,
) -> None:
    _owner(tmp_path, "northstar")
    monkeypatch.delattr(module, capability)
    with pytest.raises(ContributionError, match=message):
        check_contributions(tmp_path)


def test_enumeration_oserror_is_translated_at_owner_seam(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _owner(tmp_path, "northstar")
    owner_root = source.parent
    real_iterdir = Path.iterdir

    def refuse(path: Path):  # type: ignore[no-untyped-def]
        if path == owner_root:
            raise OSError("enumeration sentinel")
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", refuse)
    with pytest.raises(ContributionError, match="cannot enumerate owner root"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    ("field", "old", "new"),
    [
        ("code_license", "Apache-2.0", "MIT"),
        ("content_license", "CC-BY-4.0", "Apache-2.0"),
        ("code_license", "Apache-2.0", "2"),
    ],
)
def test_exact_licenses(tmp_path: Path, field: str, old: str, new: str) -> None:
    source = _owner(tmp_path, "northstar")
    owner = source.parent / "OWNER.yaml"
    owner.write_text(owner.read_text().replace(f"{field}: {old}", f"{field}: {new}"))
    with pytest.raises(ContributionError, match=field):
        check_contributions(tmp_path)


def test_owner_root_is_inert(tmp_path: Path) -> None:
    source = _owner(tmp_path, "northstar")
    (source.parent / "hidden.py").write_text("import os\n")
    with pytest.raises(ContributionError, match="owner root"):
        check_contributions(tmp_path)
    (source.parent / "hidden.py").unlink()
    (source.parent / "__init__.py").write_text("import os\n")
    with pytest.raises(ContributionError, match="inert"):
        check_contributions(tmp_path)


def test_nested_symlink_and_source_directory_substitution_are_rejected(
    tmp_path: Path,
) -> None:
    source = _owner(tmp_path, "northstar")
    target = source / "real.py"
    target.write_text("VALUE = 1\n")
    (source / "nested.py").symlink_to(target.name)
    with pytest.raises(ContributionError, match="nested symlink"):
        check_contributions(tmp_path)
    (source / "nested.py").unlink()
    source.rename(source.with_name("source-real"))
    source.write_text("not a directory\n")
    with pytest.raises(ContributionError, match="real directory"):
        check_contributions(tmp_path)


def test_equal_declared_paths_across_operations_are_rejected(tmp_path: Path) -> None:
    _owner(tmp_path, "northstar", "first")
    second = _owner(tmp_path, "northstar", "second")
    shared = "docs/contributions/northstar/first.md"
    manifest = second / "CONTRIBUTION.yaml"
    manifest.write_text(
        manifest.read_text().replace("docs/contributions/northstar/second.md", shared)
    )
    with pytest.raises(ContributionError, match="overlaps another claim"):
        check_contributions(tmp_path)


def test_generated_pack_template_is_a_metadata_positive_control(tmp_path: Path) -> None:
    source = _owner(tmp_path, "northstar")
    template = (
        Path("src/operatebench/resources/operation_pack_template/pack.py.tmpl")
        .read_text(encoding="utf-8")
        .replace("{{PACK_ID}}", "partner.northstar.sample")
        .replace("{{OPERATION_TYPE}}", "northstar.sample.synthetic")
        .replace("{{OPERATION_ID}}", "northstar_sample")
        .replace("{{MODULE_NAME}}", "sample")
        .replace("{{OWNER_ID}}", "northstar")
        .replace("{{OWNER_DISPLAY_NAME_LITERAL}}", repr("Northstar Fictional Company"))
        .replace("{{CONTRIBUTION_KIND}}", "partner")
    )
    (source / "pack.py").write_text(template)
    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


@pytest.mark.parametrize("sdk", ["SDK_VERSION", "1", "True", "2 + 0"])
def test_sdk_api_version_must_be_v2_direct_or_default(tmp_path: Path, sdk: str) -> None:
    source = _owner(tmp_path, "northstar")
    pack = source / "pack.py"
    pack.write_text(
        pack.read_text().replace(
            "evidence_eligible=False,", f"evidence_eligible=False, sdk_api_version={sdk},"
        )
    )
    with pytest.raises(ContributionError, match="sdk_api_version"):
        check_contributions(tmp_path)


def test_parser_failure_is_named(tmp_path: Path) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "pack.py").write_bytes(b"\xff")
    with pytest.raises(ContributionError, match="cannot parse"):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("default_spec", "'examples/contributions/northstar/other/operation.yaml'"),
        ("status", "'development'"),
        ("privacy_status", "'PRIVATE'"),
        ("evidence_eligible", "True"),
    ],
)
def test_metadata_binds_fixture_and_admission_fields(
    tmp_path: Path, field: str, replacement: str
) -> None:
    source = _owner(tmp_path, "northstar")
    pack = source / "pack.py"
    text = pack.read_text()
    current = {
        "default_spec": "'examples/contributions/northstar/sample/operation.yaml'",
        "status": "'incubator'",
        "privacy_status": "'SYNTHETIC_ONLY'",
        "evidence_eligible": "False",
    }[field]
    pack.write_text(text.replace(f"{field}={current}", f"{field}={replacement}"))
    with pytest.raises(ContributionError, match=field):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    ("preamble", "value"),
    [
        (
            "FIXTURE = 'examples/contributions/northstar/sample/operation.yaml'\n",
            "FIXTURE",
        ),
        ("", "'examples/contributions/northstar/sample/operation.yaml' + ''"),
        (
            "'examples/contributions/northstar/sample/operation.yaml'\n",
            "'examples/contributions/northstar/other/operation.yaml'",
        ),
    ],
)
def test_fixture_binding_rejects_dead_docstring_and_computed_laundering(
    tmp_path: Path, preamble: str, value: str
) -> None:
    source = _owner(tmp_path, "northstar")
    pack = source / "pack.py"
    current = "'examples/contributions/northstar/sample/operation.yaml'"
    pack.write_text(
        preamble
        + pack.read_text().replace(f"default_spec={current}", f"default_spec={value}")
    )
    with pytest.raises(ContributionError, match="default_spec"):
        check_contributions(tmp_path)


@pytest.mark.parametrize("sentinel", [RuntimeError("runtime"), TypeError("type")])
def test_main_propagates_trusted_programmer_exception_identity(
    monkeypatch: pytest.MonkeyPatch, sentinel: Exception
) -> None:
    def explode(root: Path) -> tuple[str, ...]:
        del root
        raise sentinel

    monkeypatch.setattr(isolation, "check_contributions", explode)
    with pytest.raises(type(sentinel)) as raised:
        isolation.main()
    assert raised.value is sentinel


def test_main_runs_tracked_artifact_gate_before_library_check(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def reject(root: Path) -> None:
        del root
        raise ContributionError("tracked contribution artifact is forbidden")

    def must_not_run(root: Path) -> tuple[str, ...]:
        del root
        raise AssertionError("library check ran after tracked-artifact rejection")

    monkeypatch.setattr(isolation, "check_tracked_contribution_artifacts", reject)
    monkeypatch.setattr(isolation, "check_contributions", must_not_run)
    assert isolation.main() == 1
    assert "tracked contribution artifact" in capsys.readouterr().err


def test_main_translates_malformed_input_only_through_contribution_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    source = _owner(tmp_path, "northstar")
    (source / "CONTRIBUTION.yaml").write_bytes(b"\xff")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    monkeypatch.setattr(
        isolation, "check_tracked_contribution_artifacts", lambda root: None
    )
    assert isolation.main() == 1
    assert "cannot parse" in capsys.readouterr().err


def test_documented_policy_gate_is_not_presented_as_a_python_sandbox() -> None:
    text = " ".join(
        Path("docs/OPERATION_PACK_SDK.md").read_text(encoding="utf-8").split()
    )
    assert "Source, import and declared-path isolation" in text
    assert "not a sandbox against arbitrary reviewed Python" in text
    assert "public-fork CI has no secrets" in text
    assert "ephemeral hosted runners" in text
    assert "do not make malicious Python safe" in text
