"""Older-shaped artefacts derived from a run this build can still produce.

The suite freezes one real v1 artefact — ``tests/fixtures/artifact_v1_reference_V1.json``
— and never rewrites it. Its bytes are pinned, because it is evidence about the
contract as it actually stood and a fixture regenerated to make a test pass is
not evidence of anything.

What that fixture can no longer be is *executable* evidence that the v1 path
still works end to end. It names the operation spec digest of a fixture this
build no longer ships, so replaying it is a named incompatibility rather than a
comparison — which is the honest outcome, and is asserted as one.

So the two questions are asked separately, and this helper answers the second.
It takes an artefact the current build produced and projects it back onto an
older contract: keep exactly the fields that contract carried, write its
trajectory rows the way that contract wrote them, and say that version.

Two things about the projection are worth stating plainly, because both are
places a test could quietly stop being evidence.

**It only ever drops.** The projection is
:func:`operatebench.artifact.project_trajectory` — the same function replay uses
to reconstruct an older document — and it removes row fields a later contract
added rather than inventing anything. A projection cannot manufacture evidence
the run did not produce.

**The trajectory digest is recomputed, and only that one.** Contract 3 changed
the durable body of the ``effect_accepted`` row, so a v1- or v2-shaped
projection of a v3 run genuinely carries different trajectory content, and a
digest is over content. Every other digest a projected artefact carries is taken
over content the older contract already had, and none of them is touched.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from operatebench.artifact import (
    ARTIFACT_VERSION_V1,
    ARTIFACT_VERSION_V2,
    content_digest,
    evaluation_dimensions_for,
    project_top_level,
    project_trajectory,
)


def as_evaluation_shape(evaluation: Mapping[str, Any], version: int) -> dict[str, Any]:
    """The result vector as contract ``version`` graded one.

    Only ever drops, exactly as the trajectory projection does: the dimensions
    that contract did not grade go, and the three fields the writer derives from
    the vector are re-derived from what is left rather than carried over. Carried
    over, they would describe the run the current contract graded, which is the
    one thing a projection must never claim.
    """
    names = evaluation_dimensions_for(version)
    kept = [entry for entry in evaluation["dimensions"] if entry["name"] in names]
    projected = dict(evaluation)
    projected["dimensions"] = kept
    projected["failed_dimensions"] = [entry["name"] for entry in kept if not entry["ok"]]
    projected["finding_codes"] = [
        finding["code"] for entry in kept for finding in entry["findings"]
    ]
    projected["reliable"] = bool(evaluation["legitimate_completion"]) and all(
        entry["ok"] for entry in kept
    )
    return projected


def as_version_shape(payload: Mapping[str, Any], version: int) -> dict[str, Any]:
    """The projection of a current artefact onto contract ``version``."""
    body = json.loads(json.dumps(payload))
    # The same top-level projection the reader ships, rather than a second one
    # here: a helper that dropped fields by its own rule would be the place the
    # suite and the build silently disagree about what an older contract carried.
    projected = project_top_level(body, version)
    projected["artifact_version"] = version
    projected["trajectory"] = project_trajectory(projected["trajectory"], version)
    projected["trajectory_digest_sha256"] = content_digest(projected["trajectory"])
    projected["evaluation"] = as_evaluation_shape(projected["evaluation"], version)
    return projected


def as_v1_shape(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The v1 projection of a current artefact, detached from its source."""
    return as_version_shape(payload, ARTIFACT_VERSION_V1)


def as_v2_shape(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The v2 projection of a current artefact, detached from its source."""
    return as_version_shape(payload, ARTIFACT_VERSION_V2)


__all__ = [
    "as_evaluation_shape",
    "as_v1_shape",
    "as_v2_shape",
    "as_version_shape",
]
