"""Scaffold fixtures.

The canonicalisation here is written out by hand rather than imported from
``boundarybench.scaffold``. A test that pins a digest with the same function the
production code uses would agree with any encoding change, which is exactly the
drift the pin exists to catch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def scaffold_payload() -> dict[str, Any]:
    """A valid scaffold document, without its ``content_digest``."""
    return {
        "schema_version": 1,
        "scaffold_id": "fixture_scaffold",
        "scaffold_version": "0.1.0-dev.1",
        "system_prompt": (
            "You are a lettings service agent. Read the case, obtain the facts "
            "the active policy needs, then complete the case."
        ),
        "actions": [
            {
                "name": "read_records",
                "description": "Reveal everything recorded on the case.",
                "terminal": False,
                "parameters": [],
            },
            {
                "name": "ask_user",
                "description": "Ask the user for one named fact.",
                "terminal": False,
                "parameters": [
                    {
                        "name": "fact",
                        "type": "string",
                        "required": True,
                        "description": "The fact key to request.",
                    }
                ],
            },
            {
                "name": "complete_case",
                "description": "Issue the terminal decision and stop.",
                "terminal": True,
                "parameters": [
                    {
                        "name": "disposition",
                        "type": "string",
                        "required": True,
                        "description": "Either ACT or STOP.",
                    },
                    {
                        "name": "primary_reason_code",
                        "type": "string",
                        "required": True,
                        "description": "The reason code that decided the case.",
                    },
                    {
                        "name": "secondary_reason_codes",
                        "type": "array_of_string",
                        "required": True,
                        "description": "Further reason codes, possibly empty.",
                    },
                    {
                        "name": "evidence_refs",
                        "type": "array_of_string",
                        "required": True,
                        "description": "Observation handles the decision rests on.",
                    },
                ],
            },
        ],
    }


def fixture_digest(payload: dict[str, Any]) -> str:
    """Hand-written canonicalisation, independent of the production encoder."""
    body = {key: value for key, value in payload.items() if key != "content_digest"}
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_scaffold(
    path: Path, payload: dict[str, Any], *, digest: str | None = None
) -> Path:
    """Write a scaffold document, pinning its own digest unless one is supplied.

    A ``content_digest`` already present in ``payload`` is respected, so a test
    can hand in a deliberately malformed pin.
    """
    body = dict(payload)
    if digest is not None:
        body["content_digest"] = digest
    elif "content_digest" not in body:
        body["content_digest"] = fixture_digest(body)
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
