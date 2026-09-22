"""The partner intake has one complete, safe, and honestly scoped entry path."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_BLOB_URL = "https://github.com/DwellyOrg/OperateBench/blob/main/"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_operation_proposal_form_is_valid_and_complete() -> None:
    form = yaml.safe_load(_read(".github/ISSUE_TEMPLATE/operation-proposal.yml"))
    assert form["name"] == "Operation proposal"
    assert isinstance(form["body"], list)
    items = [item for item in form["body"] if "id" in item]
    ids = [item["id"] for item in items]
    assert len(ids) == len(set(ids))
    assert {
        "title",
        "domain",
        "operation_fit",
        "actors_authority",
        "lifecycle_paths",
        "time_events",
        "humans_authority",
        "claims_truth",
        "recovery_finality",
        "failure_modes",
        "matched_controls",
        "capability_tags",
        "safety",
        "stage",
    } == set(ids)
    for item in items:
        if item["type"] in {"input", "textarea", "dropdown"}:
            assert item.get("validations", {}).get("required") is True
    safety = next(item for item in items if item["id"] == "safety")
    assert all(
        option.get("required") is True for option in safety["attributes"]["options"]
    )


def test_public_form_warns_before_collecting_free_text_and_links_the_kit() -> None:
    form = yaml.safe_load(_read(".github/ISSUE_TEMPLATE/operation-proposal.yml"))
    warning = form["body"][0]["attributes"]["value"]
    for required in (
        "public issue",
        "fully synthetic",
        "do not paste",
        "replacing names",
        "stop and do not submit",
        "security.md",
        "partner contribution kit",
    ):
        assert required in warning.lower()
    links = set(re.findall(r"\]\((https://github\.com/[^)]+)\)", warning))
    expected_paths = {
        "SECURITY.md",
        "docs/CONTRIBUTING_OPERATIONS.md",
        "docs/PARTNER_CONTRIBUTION_KIT.md",
        "examples/partner_operation_proposal.md",
    }
    assert links == {REPOSITORY_BLOB_URL + path for path in expected_paths}
    assert all((ROOT / path).is_file() for path in expected_paths)
    assert "../../" not in warning


def test_partner_guide_routes_to_every_required_surface() -> None:
    guide = _read("docs/PARTNER_CONTRIBUTION_KIT.md")
    for link in (
        "templates/OPERATION_PROPOSAL_WORKSHEET.md",
        "../examples/partner_operation_proposal.md",
        "../SECURITY.md",
        "CONTRIBUTING_OPERATIONS.md",
        "../CONTRIBUTING.md",
    ):
        assert f"]({link})" in guide
    for boundary in (
        "proposal self-service; executable packs remain guided",
        "SYNTHETIC_ONLY",
        "not the contributing organisation's official policy",
        "not official benchmark evidence",
    ):
        assert boundary in guide


def test_worked_example_is_explicitly_synthetic_and_semantically_complete() -> None:
    example = _read("examples/partner_operation_proposal.md")
    example_flat = " ".join(line.lstrip("> ").strip() for line in example.splitlines())
    assert "not an executable pack" in example_flat
    assert "SYNTHETIC_ONLY" in example_flat
    assert "not the policy or workflow" in example_flat
    for construct in (
        "authoritative",
        "Correct wait",
        "Required checkpoint",
        "Authority boundary",
        "Reopen",
        "Guarded finality",
        "Targeted causal failures",
        "Stateful time-removed",
    ):
        assert construct.lower() in example.lower()

    failure_section = example.split("## Targeted causal failures", 1)[1].split(
        "## Matched controls and tags", 1
    )[0]
    table_rows = [line for line in failure_section.splitlines() if line.startswith("|")]
    headings = [cell.strip() for cell in table_rows[0].strip("|").split("|")]
    assert headings == [
        "Failure",
        "Attacked guarantee",
        "Expected effect/finding",
        "Must remain correct",
    ]
    data_rows = table_rows[2:]
    assert len(data_rows) == 8
    for row in data_rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert len(cells) == 4
        assert all(cells)


def test_public_contribution_docs_state_the_same_stage() -> None:
    for relative in (
        "README.md",
        "CONTRIBUTING.md",
        "docs/CONTRIBUTING_OPERATIONS.md",
    ):
        text = " ".join(_read(relative).split()).lower()
        assert "proposal self-service" in text
        assert "executable" in text and "guided" in text


def test_development_registration_does_not_claim_independent_review() -> None:
    guide = " ".join(_read("docs/PARTNER_CONTRIBUTION_KIT.md").split()).lower()
    operations = " ".join(_read("docs/CONTRIBUTING_OPERATIONS.md").split()).lower()

    for text in (guide, operations):
        assert re.search(
            r"development registration.{0,240}neither requires nor records"
            r".{0,160}independent",
            text,
        )
        assert "unavailable in sdk v2" in text


def test_every_public_closed_exit_code_list_includes_usage_exit_64() -> None:
    lists = []
    for document in (ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))):
        text = document.read_text(encoding="utf-8")
        lists.extend(
            (document, match.group(0))
            for match in re.finditer(
                r"Exit codes(?: are)?:.*?(?:\n\n|$)", text, re.DOTALL
            )
        )

    assert lists
    for document, exit_codes in lists:
        assert "`64`" in exit_codes, document.relative_to(ROOT)
