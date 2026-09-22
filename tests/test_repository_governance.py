"""Repository governance and publication-claim boundaries."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = "@khanukov"
REQUIRED_OWNER_PATTERNS = {
    "/src/operatebench/",
    "/tools/",
    "/.github/",
    "/docs/METHODOLOGY.md",
    "/docs/REPOSITORY_GOVERNANCE.md",
    "/PUBLICATION_MANIFEST.json",
    "/pyproject.toml",
    "/uv.lock",
    "/REUSE.toml",
}
MANDATORY_OWNED_FILES = {
    ".github/CODEOWNERS",
    ".github/pull_request_template.md",
    ".github/workflows/ci.yml",
    ".github/workflows/dco.yml",
    "docs/METHODOLOGY.md",
    "docs/REPOSITORY_GOVERNANCE.md",
    "PUBLICATION_MANIFEST.json",
    "pyproject.toml",
    "uv.lock",
    "REUSE.toml",
    "src/operatebench/runner.py",
    "src/operatebench/execution_ledger.py",
    "src/operatebench/providers/executor.py",
    "src/operatebench/agents/openai_responses.py",
    "tools/check_public_release.py",
    "tools/check_dco.py",
}


def _flowed(path: str) -> str:
    return " ".join((ROOT / path).read_text(encoding="utf-8").split())


def _codeowner_rules() -> list[tuple[str, tuple[str, ...]]]:
    rules: list[tuple[str, tuple[str, ...]]] = []
    for raw in (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pattern, *owners = line.split()
        assert owners, f"CODEOWNERS rule has no owner: {raw!r}"
        assert not pattern.startswith("!"), "CODEOWNERS does not support negation"
        assert "[" not in pattern and "]" not in pattern
        rules.append((pattern, tuple(owners)))
    assert rules, "CODEOWNERS must contain at least one ownership rule"
    return rules


def _matches(pattern: str, repository_path: str) -> bool:
    """Match the exact-file/directory GitHub syntax this policy permits."""
    assert pattern.startswith("/"), (
        f"sensitive ownership must be root-anchored: {pattern}"
    )
    assert not any(character in pattern for character in "*?\\"), (
        f"policy CODEOWNERS rules must be exact files or directories: {pattern}"
    )
    anchored = pattern.removeprefix("/")
    if anchored.endswith("/"):
        return repository_path.startswith(anchored)
    return repository_path == anchored


def _owners_for(repository_path: str) -> tuple[str, ...]:
    owners: tuple[str, ...] = ()
    for pattern, candidate_owners in _codeowner_rules():
        if _matches(pattern, repository_path):
            owners = candidate_owners
    return owners


def test_codeowners_covers_current_and_future_sensitive_paths() -> None:
    current_source_and_tools = {
        path.relative_to(ROOT).as_posix()
        for root in (ROOT / "src" / "operatebench", ROOT / "tools")
        for path in root.rglob("*")
        if path.is_file()
    }
    current_github_files = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / ".github").rglob("*")
        if path.is_file()
    }
    sensitive = current_source_and_tools | current_github_files | MANDATORY_OWNED_FILES

    assert sensitive >= MANDATORY_OWNED_FILES
    assert current_source_and_tools
    assert current_github_files
    assert all(OWNER in _owners_for(path) for path in sensitive)


def test_codeowners_uses_broad_root_anchored_rules() -> None:
    patterns = {pattern for pattern, owners in _codeowner_rules() if OWNER in owners}
    assert patterns >= REQUIRED_OWNER_PATTERNS


def test_readme_states_the_bounded_non_claim_and_execution_boundary() -> None:
    readme = _flowed("README.md")
    assert (
        "Not built: any release-grade matched-control model evaluation or published "
        "benchmark model result." in readme
    )
    assert "documented public/offline benchmark commands make no provider calls" in readme
    assert "Private operator-only diagnostic runs" in readme
    assert "excluded from published benchmark evidence" in readme
    assert "Provider cost is not part of the benchmark result vector" in readme


ABSOLUTE_PROVIDER_CLAIMS = (
    "no run makes a provider call",
    "no run in this build makes a provider call",
    "nothing in this build calls a provider",
    "the preview runs no provider calls",
)


def test_public_docs_do_not_overstate_the_provider_call_boundary() -> None:
    for name in (
        "README.md",
        "docs/METHODOLOGY.md",
        "docs/ARCHITECTURE_RFC.md",
        "SECURITY.md",
    ):
        lowered = _flowed(name).lower()
        for stale in ABSOLUTE_PROVIDER_CLAIMS:
            assert stale not in lowered, f"{name} retains absolute claim: {stale}"

    methodology = _flowed("docs/METHODOLOGY.md")
    assert "documented public/offline benchmark commands" in methodology
    assert "make no provider calls" in methodology
    assert "operator-only diagnostics" in methodology
    assert "excluded from published benchmark evidence" in methodology
    assert "Provider cost is not part of the benchmark result vector" in methodology
    assert "cannot establish current capability" in methodology

    security = _flowed("SECURITY.md")
    assert (
        "documented public/offline benchmark commands make no provider calls" in security
    )


def test_manifest_excludes_private_diagnostics_and_capability_authority() -> None:
    manifest = json.loads((ROOT / "PUBLICATION_MANIFEST.json").read_text())
    declarations = manifest["declarations"]
    assert declarations["no_published_benchmark_model_results"]["value"] is True
    assert declarations["no_published_live_provider_evidence"]["value"] is True
    assert "private_lifecycle_diagnostics" not in declarations
    assert "approvals" not in manifest
    assert manifest["published_artifacts"] == []
    for key in (
        "assigned_episodes",
        "integrity_valid_scored_episodes",
        "preregistered_provider_budget_exclusions",
        "successful_completions",
        "interface_commit",
        "repository_baseline_commit",
    ):
        assert key not in json.dumps(manifest)


def test_governance_names_stable_aggregate_checks_without_claiming_ruleset_state() -> (
    None
):
    governance = _flowed("docs/REPOSITORY_GOVERNANCE.md")
    governance_lower = governance.lower()
    required = {"test (py3.11)", "test (py3.14)", "build distribution", "dco"}
    assert all(f"`{context}`" in governance for context in required)
    assert "Ruff" in governance
    assert "mypy" in governance
    assert "REUSE" in governance
    assert "causal" in governance
    assert "public scanner" in governance
    assert "Twine" in governance
    for phrase in (
        "transitively",
        "not separately require-able",
        "forge-side configuration",
        "does not prove that a ruleset is enabled",
        "one independent approval",
        "stale approvals",
        "resolved",
        "force-push",
        "named emergency role",
        "signed annotated tags",
        "exact-candidate verification",
        "separate authorization",
    ):
        assert phrase in governance_lower


def test_ruleset_activation_is_blocked_until_both_distinct_roles_are_verified() -> None:
    governance = _flowed("docs/REPOSITORY_GOVERNANCE.md")
    assert (
        "The approval-, CODEOWNERS-, and bypass-dependent ruleset MUST NOT be "
        "activated while only `@khanukov` exists and no distinct emergency role "
        "exists." in governance
    )
    assert (
        "Activation prerequisites are both: (a) at least one distinct reviewer or "
        "team with repository access and a verified approval path; and (b) a named, "
        "distinct emergency team or custom role with a verified bypass path."
        in governance
    )
    assert (
        "Until both prerequisites exist, forge enforcement is blocked and the "
        "repository remains unprotected; partial activation is not safe." in governance
    )
