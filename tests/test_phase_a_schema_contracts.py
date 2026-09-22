"""Cross-document contracts for the normative Phase A schemas."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = REPO_ROOT / "docs" / "schemas"


def _load_json(name: str) -> dict:
    return json.loads((SCHEMA_ROOT / name).read_text(encoding="utf-8"))


def test_embedded_schema_keywords_agree_with_the_strict_subset_registry() -> None:
    registry = _load_json("card-registry-v1.json")["strict_json_schema_subset"]
    branches = _load_json("card-defs-v1.schema.json")["$defs"]["jsonSchemaSubset"][
        "oneOf"
    ]

    # Only each branch's declared wire properties are embedded-schema keywords.
    # Recursive implementation details inside their validators (for example
    # $ref) are meta-schema machinery, not keys accepted on the embedded wire.
    embedded_keywords = {
        keyword for branch in branches for keyword in branch["properties"]
    }
    missing_from_allowed = embedded_keywords - set(registry["allowed_keywords"])
    present_in_forbidden = embedded_keywords & set(registry["forbidden_keywords"])

    assert (missing_from_allowed, present_in_forbidden) == (set(), set()), (
        "embedded jsonSchemaSubset keywords must all be registry-allowed and none "
        "registry-forbidden"
    )
