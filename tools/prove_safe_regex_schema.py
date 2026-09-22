"""Differential proof harness for the published Card v1 safe-regex grammar."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import string
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import cast

from operatebench import cards

REPO_ROOT = Path(__file__).resolve().parents[1]
BOUNDED_ALPHABET = r"a\[]-*?{},^$"
BOUNDED_MAX_LENGTH = 4
RANDOM_SEED = 0xC4A2D
RANDOM_CASES = 5_000


def _pattern_schema() -> dict[str, object]:
    document = json.loads(
        (REPO_ROOT / "docs/schemas/card-defs-v1.schema.json").read_text(encoding="utf-8")
    )
    return cast(dict[str, object], document["$defs"]["safeRegexPattern"])


def _runtime_accepts(pattern: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            cards._validate_safe_pattern(pattern, "$.pattern")
    except cards.CardValidationError:
        return False
    return True


def _schema_accepts(pattern: str, schema: dict[str, object]) -> bool:
    try:
        cards._validate_schema(pattern, schema, "$.pattern")
    except cards.CardValidationError:
        return False
    return True


def _bounded_cases() -> Iterable[str]:
    for length in range(BOUNDED_MAX_LENGTH + 1):
        yield from map("".join, itertools.product(BOUNDED_ALPHABET, repeat=length))


def _structured_cases() -> set[str]:
    literals = ["a", "-", " ", "~", r"\.", r"\|", r"\d", r"\W"]
    classes = [
        "[a]",
        "[^a]",
        "[-]",
        "[--]",
        "[a-]",
        "[-a]",
        "[a-z]",
        "[ -~]",
        r"[A-Z\d]",
        r"[(){}+*?.|\\]",
    ]
    fixed = [
        "",
        "{0}",
        "{1}",
        "{4095}",
        "{4096}",
        "{0,0}",
        "{4096,4096}",
        "{4096,}",
    ]
    variable = [
        "*",
        "+",
        "?",
        "{0,1}",
        "{1,8}",
        "{0,4096}",
        "{4095,4096}",
        "{0,}",
        "{4095,}",
    ]
    malformed = [
        "a|b",
        "(a)",
        ".",
        "a*a*",
        "a**",
        "a{2}b{3}",
        "a*b+",
        "[a-z",
        "[]",
        "[^]",
        "[z-a]",
        "[a--z]",
        "a\\",
        r"\q",
        r"\1",
        "a{4097}",
        "a{2,1}",
        "a{1,4097}",
        "a{,2}",
        "^a^",
        "$a",
    ]
    conservative_runtime_extensions = [
        "a{01}",
        "a{01,02}",
        "a{01,}",
        "a{2,2}",
        "a{10,20}",
        r"[\.-/]",
        "[b-y]",
        "[a-b-c]",
    ]
    cases = set(malformed + conservative_runtime_extensions)
    cases.update({"", "^", "$", "^$", "~" * 256, "~" * 257, "é", "a\n"})
    atoms = literals + classes
    for atom in atoms:
        cases.update(atom + quantifier for quantifier in fixed + variable)
        cases.update({atom, "^" + atom, atom + "$", "^" + atom + "$"})
    for left in atoms:
        for right in atoms:
            cases.update({left + right, left + "*" + right, left + "{2}" + right})
    return cases


def _random_cases() -> list[str]:
    rng = random.Random(RANDOM_SEED)
    alphabet = string.printable[:-6]
    return [
        "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 40)))
        for _ in range(RANDOM_CASES)
    ]


def prove() -> dict[str, object]:
    schema = _pattern_schema()
    corpora = {
        "bounded_exhaustive": list(_bounded_cases()),
        "structured": sorted(_structured_cases()),
        "seeded_adversarial": _random_cases(),
    }
    false_accepts: list[tuple[str, str]] = []
    conservative_rejections: list[tuple[str, str]] = []
    for corpus_name, patterns in corpora.items():
        for pattern in patterns:
            runtime = _runtime_accepts(pattern)
            published_schema = _schema_accepts(pattern, schema)
            if published_schema and not runtime:
                false_accepts.append((corpus_name, pattern))
            elif runtime and not published_schema:
                conservative_rejections.append((corpus_name, pattern))
    return {
        "counts": {name: len(patterns) for name, patterns in corpora.items()},
        "total": sum(len(patterns) for patterns in corpora.values()),
        "schema_accept_runtime_reject": false_accepts,
        "runtime_accept_schema_reject_count": len(conservative_rejections),
        "runtime_accept_schema_reject_examples": conservative_rejections[:20],
        "random_seed": RANDOM_SEED,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = prove()
    print(json.dumps(result, indent=2) if args.json else result)
    return 1 if result["schema_accept_runtime_reject"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
