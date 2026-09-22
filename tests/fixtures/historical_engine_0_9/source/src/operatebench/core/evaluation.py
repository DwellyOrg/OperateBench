"""The typed, multidimensional result. No weighted composite, no LLM judge.

``reliable`` is a strict conjunction over every dimension plus legitimate
completion, exactly as the RFC states it. It is deliberately impossible to trade
a critical invariant violation against a good SLA record: there is no weight to
tune, so no failure can be averaged away. Every dimension travels with the
result, so a single boolean can never stand in for the vector.

Findings carry a stable ``code`` as well as prose. The prose is for a human
reading a report; the code is what the negative-agent contract asserts against,
so "this agent failed for the intended reason" is a check rather than a hope.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Finding:
    """One reason a dimension did not hold, with a stable machine-readable code."""

    code: str
    detail: str
    at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "detail": self.detail, "at": self.at}


@dataclass(frozen=True)
class Dimension:
    """One evaluated property of an episode.

    ``counts`` carries the measured quantities the RFC asks to be reported
    alongside the boolean — human touches, duplicate actions, SLA breaches —
    so a passing dimension still says how much it cost to pass.
    """

    name: str
    ok: bool
    findings: Sequence[Finding] = ()
    counts: Mapping[str, int] = field(default_factory=dict)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "findings": [finding.as_dict() for finding in self.findings],
            "counts": dict(self.counts),
            "note": self.note,
        }


@dataclass(frozen=True)
class OperationEvaluation:
    """The whole result vector for one episode."""

    terminal_outcome: str
    legitimate_completion: bool
    dimensions: Sequence[Dimension]

    @property
    def reliable(self) -> bool:
        """Strict conjunction: every dimension holds and the outcome is legitimate."""
        return self.legitimate_completion and all(
            dimension.ok for dimension in self.dimensions
        )

    @property
    def failed_dimensions(self) -> tuple[str, ...]:
        return tuple(dimension.name for dimension in self.dimensions if not dimension.ok)

    @property
    def passed_dimensions(self) -> tuple[str, ...]:
        return tuple(dimension.name for dimension in self.dimensions if dimension.ok)

    @property
    def finding_codes(self) -> tuple[str, ...]:
        return tuple(
            finding.code
            for dimension in self.dimensions
            for finding in dimension.findings
        )

    def dimension(self, name: str) -> Dimension | None:
        for dimension in self.dimensions:
            if dimension.name == name:
                return dimension
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "terminal_outcome": self.terminal_outcome,
            "legitimate_completion": self.legitimate_completion,
            "reliable": self.reliable,
            "failed_dimensions": list(self.failed_dimensions),
            "finding_codes": list(self.finding_codes),
            "dimensions": [dimension.as_dict() for dimension in self.dimensions],
        }


__all__ = ["Dimension", "Finding", "OperationEvaluation"]
