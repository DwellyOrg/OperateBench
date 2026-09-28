"""Offline contracts for repository-only supply-chain controls."""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _require_repository_checkout() -> None:
    # Hatch adds PKG-INFO to sdists; absence of a control file alone must never
    # exempt a checkout (including linked worktrees, whose .git is a file).
    if (ROOT / "PKG-INFO").is_file() and not (ROOT / ".git").exists():
        pytest.skip("repository-only control is not shipped in the source distribution")


def test_dependency_review_is_pinned_read_only_and_pr_only() -> None:
    _require_repository_checkout()
    path = ROOT / ".github/workflows/dependency-review.yml"
    assert path.is_file(), "Pull requests need dependency review"
    text = path.read_text()
    # BaseLoader preserves the YAML 1.2 Actions key `on` (not a YAML 1.1 bool).
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    assert set(workflow) == {"name", "on", "permissions", "jobs"}
    assert workflow["on"] == {"pull_request": ""}
    assert workflow["permissions"] == {"contents": "read", "pull-requests": "read"}
    assert set(workflow["jobs"]) == {"dependency-review"}
    job = workflow["jobs"]["dependency-review"]
    assert set(job) == {"name", "runs-on", "timeout-minutes", "steps"}
    assert job["name"] == "dependency-review"
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == "5"
    # No checkout or execution of PR contents, extra steps, or permission overrides.
    assert len(job["steps"]) == 1
    step = job["steps"][0]
    assert set(step) == {"name", "uses", "with"}
    assert re.fullmatch(r"actions/dependency-review-action@[0-9a-f]{40}", step["uses"])
    assert step["with"] == {
        "fail-on-severity": "high",
        "fail-on-scopes": "runtime, development, unknown",
        "vulnerability-check": "true",
        "license-check": "false",
        "warn-only": "false",
        "comment-summary-in-pr": "never",
        "show-openssf-scorecard": "false",
    }
    assert "secrets." not in text


def test_private_reporting_route_retains_email_fallback() -> None:
    policy = (ROOT / "SECURITY.md").read_text()
    assert "https://github.com/DwellyOrg/OperateBench/security/advisories/new" in policy
    assert "dmitry@dwelly.group" in policy
    assert "unavailable, use the email address above" in policy
    assert "**Do not open a public issue**" in policy


def test_dependabot_uses_weekly_bounded_version_updates() -> None:
    _require_repository_checkout()
    path = ROOT / ".github/dependabot.yml"
    assert path.is_file(), "Dependabot version updates must be configured"
    config = yaml.safe_load(path.read_text())
    assert set(config) == {"version", "updates"}
    assert config["version"] == 2
    assert len(config["updates"]) == 2
    assert {u["package-ecosystem"] for u in config["updates"]} == {"uv", "github-actions"}
    for update in config["updates"]:
        assert set(update) == {
            "package-ecosystem",
            "directory",
            "schedule",
            "open-pull-requests-limit",
            "groups",
        }
        assert update["directory"] == "/"
        assert update["schedule"] == {"interval": "weekly"}
        assert update["open-pull-requests-limit"] == 3
        assert update["groups"] == {
            "minor-and-patch": {
                "patterns": ["*"],
                "update-types": ["minor", "patch"],
                "applies-to": "version-updates",
            }
        }
