"""Guard public sources against development-history and production-linkage residue."""

from __future__ import annotations

import ast
import io
import re
import subprocess
import tokenize
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNED_ROOTS = ("src/", "docs/", "tests/", "examples/")
SEMANTIC_PHASE_NAMES = {"tests/test_p1_probe_cube.py"}


def _public_prose(path: Path) -> list[tuple[int, str]]:
    """Markdown and Python prose surfaces, excluding executable literals."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".md":
        return [(1, text)]
    if path.suffix in {".yaml", ".yml"}:
        return [(1, text)]
    if path.suffix != ".py":
        return []

    prose = [
        (token.start[0], token.string)
        for token in tokenize.generate_tokens(io.StringIO(text).readline)
        if token.type == tokenize.COMMENT
    ]
    tree = ast.parse(text, filename=str(path))
    for node in ast.walk(tree):
        documented = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        if not isinstance(node, documented):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            prose.append((first.lineno, first.value.value))
    return prose


def _tracked_public_paths() -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files", "--", *SCANNED_ROOTS],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [REPO_ROOT / name for name in output.splitlines() if name]


def test_public_sources_avoid_development_history_and_production_linkage() -> None:
    # Semantic test names describe the guarantee under test. Phase-like prefixes,
    # numbered milestone names and review-cycle filenames expose authoring history
    # rather than public behavior.
    phase_filename = re.compile(
        r"test_(?:[a-z]\d+[a-z0-9_]*|.*(?:milestone|review_cycle|fix_cycle))",
        re.I,
    )
    forbidden_content = (
        re.compile(r"\breal AI-driven operations at\b", re.I),
        re.compile(r"\b(?:private|internal) (?:milestone|review cycle)\b", re.I),
    )

    violations: list[str] = []
    for path in _tracked_public_paths():
        relative = path.relative_to(REPO_ROOT).as_posix()
        if phase_filename.search(relative) and relative not in SEMANTIC_PHASE_NAMES:
            violations.append(f"{relative}: development phase in filename")
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden_content:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                violations.append(f"{relative}:{line}: {match.group(0)}")

    assert not violations, "forbidden public provenance markers:\n" + "\n".join(
        violations
    )


def test_public_prose_states_current_guarantees_without_private_process_claims() -> None:
    unsupported_validation = re.compile(
        r"\bpre[- ]?registr\w*\b|(?<!not )(?<!not\n)\bindependent\w*\s+"
        r"(?:human[- ]?)?validat\w*\b",
        re.I,
    )
    defect_chronology = re.compile(
        r"\b(?:reproduc\w*\s+(?:an?\s+)?defects?|blocker\s+(?-i:[A-Z0-9]+)|"
        r"(?:previous|prior|earlier)\s+(?:contract|version|release|build)|"
        r"(?:first|earlier)\s+implementation|earlier\s+audit(?:'s)?\s+authority|"
        r"used\s+to\s+(?:answer|mean|raise|live|reach|pass|produce|leave|fail|"
        r"allow|accept)|"
        r"before\s+(?:the\s+)?fix|during\s+(?:project\s+)?hardening|"
        r"after\s+(?:the\s+)?initial\s+implementation|regression\s+case|"
        r"(?:was|were)\s+accepted[.\s]+(?:reading|read)\s+now|"
        r"(?:disagreements?|differences?)\s+(?:were|was)\s+resolved)\b",
        re.I,
    )
    quantified_failure_history = re.compile(
        r"\b(?:reproduc\w*|previous\w*|prior|earlier|formerly|used\s+to)\b"
        r"[\s\S]{0,240}?(?:\bUSD\s+\d|\$\d|\b\d+\s+"
        r"(?:requests?|runs?|episodes?|rows?))\b",
        re.I,
    )
    sequenced_named_construct = re.compile(
        r"(?:\b[A-Z][a-z]+\s+Track\b[\s\S]{0,100}\b"
        r"(?:planned|deferred|later|after|follows|will\s+(?:evaluate|build))\b|"
        r"\b(?:planned|deferred|later|after|follows)\b[\s\S]{0,100}"
        r"\b[A-Z][a-z]+\s+Track\b|\bstatus\b[^\n]{0,80}\bproduct\s+direction\b)",
        re.I,
    )
    staged_run_sequence = re.compile(
        r"\b\d+\s*(?:→|->)\s*\d+\s*(?:→|->)\s*\d+\b|"
        r"\brepresentative\s+staging\b",
        re.I,
    )
    rules = (
        ("unsupported validation claim", unsupported_validation),
        ("internal defect chronology", defect_chronology),
        ("quantified prior-failure narrative", quantified_failure_history),
        ("sequenced named product construct", sequenced_named_construct),
        ("review-stage execution narrative", staged_run_sequence),
    )

    tracked = subprocess.run(
        ["git", "ls-files", "--", "*.py", "*.md", "*.yaml", "*.yml"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    violations: list[str] = []
    for relative in tracked:
        path = REPO_ROOT / relative
        for start_line, prose in _public_prose(path):
            for label, pattern in rules:
                for match in pattern.finditer(prose):
                    line = start_line + prose.count("\n", 0, match.start())
                    violations.append(f"{relative}:{line}: {label}")

    assert not violations, "unsupported public-content claims:\n" + "\n".join(violations)


def test_current_fixture_claims_have_executable_fixture_witnesses() -> None:
    rfc = (REPO_ROOT / "docs" / "ARCHITECTURE_RFC.md").read_text(encoding="utf-8")
    fixture = (
        REPO_ROOT / "examples" / "operatebench" / "maintenance_v0_1.yaml"
    ).read_text(encoding="utf-8")
    current = rfc.split("## 12. Current synthetic Maintenance fixture", 1)[1].split(
        "## 13.", 1
    )[0]
    assert "must include" not in current
    witnesses = {
        "simulated time": "starts_at:",
        "external events": "events:",
        "WAIT and scheduled wake-up": "delay_minutes:",
        "customer, supplier, approver and system actors": "actors:",
        "post-visit further-work quote": "outcome: FURTHER_WORK_REQUIRED",
        "approval checkpoint": "required_checkpoint_types: [repair_quote_approval]",
        "invoice validation and payment": (
            "authority: [validate_invoice, confirm_settlement]"
        ),
        "provisional close, reopen and finality": "provisional_close_minutes:",
        "three scenario variants": "  V3:",
    }
    for claim, witness in witnesses.items():
        assert claim in current
        assert witness in fixture
