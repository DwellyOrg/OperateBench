"""Public operator documentation describes contracts, not private work history."""

import ast
import io
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_operator_prose_describes_current_contracts():
    patterns = (
        r"\bparent[- ](?:owned|gates|review|integration)\b",
        r"\b(?:subsequent explicit owner decision|explicit owner choice)\b",
        r"\b(?:bounded lifetime probes|response repair|reviewed reducer bug)\b",
        r"\b\d+-case\b",
    )
    surfaces = []
    for path in sorted((ROOT / "docs").glob("THREE_FLOW_*.md")):
        surfaces.append((path.name, path.read_text()))
    for path in sorted((ROOT / "tools").glob("three_flow_*.py")):
        source = path.read_text()
        prose = [
            token.string
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.COMMENT
        ]
        for node in ast.walk(ast.parse(source)):
            if isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                prose.append(ast.get_docstring(node) or "")
        surfaces.append((path.name, "\n".join(prose)))
    problems = [
        (name, pattern)
        for name, text in surfaces
        for pattern in patterns
        if re.search(pattern, " ".join(text.split()), re.IGNORECASE)
    ]
    assert not problems


def test_native_assets_have_no_implicit_host_location():
    source = (ROOT / "tests" / "test_three_flow_native_modes.py").read_text()
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "THREE_FLOW_RIVER_ASSETS"
    ]
    assert len(calls) == 1
    assert len(calls[0].args) == 1
    assert not calls[0].keywords
