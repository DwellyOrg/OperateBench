"""Repository controls must not disappear behind a missing-file skip."""

from pathlib import Path

import pytest

from tests import test_repository_security as security


@pytest.mark.parametrize(
    "check",
    [
        security.test_dependency_review_is_pinned_read_only_and_pr_only,
        security.test_dependabot_uses_weekly_bounded_version_updates,
    ],
)
@pytest.mark.parametrize("git_marker", [None, "directory", "worktree"])
def test_missing_repository_control_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, check, git_marker: str | None
) -> None:
    if git_marker == "directory":
        (tmp_path / ".git").mkdir()
    elif git_marker == "worktree":
        (tmp_path / ".git").write_text("gitdir: elsewhere\n")
    if git_marker is not None:
        # Even stray distribution metadata cannot exempt a real checkout.
        (tmp_path / "PKG-INFO").write_text("Metadata-Version: 2.4\nName: operatebench\n")
    monkeypatch.setattr(security, "ROOT", tmp_path)
    with pytest.raises(AssertionError):
        check()


@pytest.mark.parametrize(
    "check",
    [
        security.test_dependency_review_is_pinned_read_only_and_pr_only,
        security.test_dependabot_uses_weekly_bounded_version_updates,
    ],
)
def test_distribution_metadata_exempts_only_repository_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, check
) -> None:
    (tmp_path / "PKG-INFO").write_text("Metadata-Version: 2.4\nName: operatebench\n")
    monkeypatch.setattr(security, "ROOT", tmp_path)
    with pytest.raises(pytest.skip.Exception, match="source distribution"):
        check()
    # SECURITY.md ships in the sdist and must still be checked there.
    with pytest.raises(FileNotFoundError):
        security.test_private_reporting_route_retains_email_fallback()
