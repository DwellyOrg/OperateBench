"""The separate negative-control oracle: versioned expectations for each control.

The acceptance gate needs expectations that the thing being graded cannot
author. If the dimensions a negative agent is allowed to break are declared in
the same module as the agent, then "the evaluator separates these behaviours" is
only ever a statement about internal consistency: change the agent and the
expectation moves with it, and the gate still passes.

So the expectations live somewhere else.
``oracles/maintenance_negative_controls_v0_5.yaml``
is a versioned expectation manifest about the *case* — for each control, the
intervention, the operation guarantee it attacks, the causal failure closure it
should produce, the finding codes that make that failure the intended one, and
the dimensions that must be left standing. It ships inside the package, so the
gate can read it from an installed wheel with no checkout present.

Two properties are load-bearing and are asserted by the suite rather than
assumed here:

* **Nothing generates it.** This module imports neither the agent registry nor
  the evaluator, and no code writes the manifest. The registry reads its
  expectations from here; the arrow never points the other way.
* **It is judged in full.** Every control's closure and its must-pass set
  together cover the whole result vector, disjointly, so no dimension is left
  unmentioned for a failure to hide in.

What this file supports is construct evidence for one synthetic operation: the
shipped evaluator separates the shipped controls under one explicit, versioned
expectation set. It is not model-evaluation evidence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from boundarybench.jsonsafe import JsonSafetyError, canonical_json_bytes
from operatebench.core.errors import OracleManifestError

#: The packaged manifest. Named once, because the release checks and the tests
#: both need to point at the same file the loader reads.
ORACLE_RESOURCE_PACKAGE = "operatebench.domains.lettings.maintenance.oracles"
ORACLE_RESOURCE_NAME = "maintenance_negative_controls_v0_5.yaml"

_TOP_LEVEL_FIELDS: tuple[str, ...] = (
    "oracle_id",
    "oracle_version",
    "operation_id",
    "semantic_scenario_id",
    "authored_against_operation_version",
    "provenance",
    "result_vector_dimensions",
    "controls",
)

_PROVENANCE_FIELDS: tuple[str, ...] = (
    "authored_by",
    "independence",
    "evidence_class",
    "review_status",
)

_CONTROL_FIELDS: tuple[str, ...] = (
    "agent_id",
    "scenario_id",
    "intervention",
    "guarantee_attacked",
    "expected_failed_dimensions",
    "required_finding_codes",
    "must_pass_dimensions",
    "rationale",
)


def _fields(payload: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    present = set(payload)
    unknown = sorted(present - set(allowed))
    if unknown:
        raise OracleManifestError(
            f"{where}: unknown field(s) {unknown}; this build knows {sorted(allowed)}"
        )
    missing = sorted(set(allowed) - present)
    if missing:
        raise OracleManifestError(f"{where}: missing required field(s) {missing}")


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OracleManifestError(
            f"{where}: expected a mapping, got {type(value).__name__}"
        )
    return value


def _text(payload: Mapping[str, Any], key: str, where: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise OracleManifestError(
            f"{where}: {key} must be a non-empty string, got {type(value).__name__}"
        )
    return value.strip()


def _names(payload: Mapping[str, Any], key: str, where: str) -> tuple[str, ...]:
    value = payload[key]
    if not isinstance(value, list) or not value:
        raise OracleManifestError(
            f"{where}: {key} must be a non-empty list of names, got "
            f"{type(value).__name__}"
        )
    names: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise OracleManifestError(f"{where}: {key} carries a non-string entry")
        if item in names:
            raise OracleManifestError(f"{where}: {key} names {item!r} twice")
        names.append(item)
    return tuple(names)


@dataclass(frozen=True)
class NegativeControl:
    """One oracle-declared negative control, as the oracle declares it.

    ``expected_failed_dimensions`` is the **causal failure closure**: every
    dimension the intervention is expected to take down, including the ones it
    takes down as a consequence of the one it was written for. It is compared as
    an exact set, so a failure outside it is an unrelated failure nobody
    registered. ``must_pass_dimensions`` is the other half of the same claim and
    the reason a generic crash cannot pass as a targeted negative.
    """

    agent_id: str
    scenario_id: str
    intervention: str
    guarantee_attacked: str
    expected_failed_dimensions: tuple[str, ...]
    required_finding_codes: tuple[str, ...]
    must_pass_dimensions: tuple[str, ...]
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "scenario_id": self.scenario_id,
            "intervention": self.intervention,
            "guarantee_attacked": self.guarantee_attacked,
            "expected_failed_dimensions": list(self.expected_failed_dimensions),
            "required_finding_codes": list(self.required_finding_codes),
            "must_pass_dimensions": list(self.must_pass_dimensions),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class NegativeControlOracle:
    """The whole manifest, with the identity of the document it was read from."""

    oracle_id: str
    oracle_version: str
    operation_id: str
    semantic_scenario_id: str
    authored_against_operation_version: str
    provenance: Mapping[str, str]
    result_vector_dimensions: tuple[str, ...]
    controls: tuple[NegativeControl, ...]
    oracle_digest_sha256: str
    source: str

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return tuple(control.agent_id for control in self.controls)

    def has_control(self, agent_id: str) -> bool:
        return any(control.agent_id == agent_id for control in self.controls)

    def control(self, agent_id: str) -> NegativeControl:
        for control in self.controls:
            if control.agent_id == agent_id:
                return control
        raise OracleManifestError(
            f"{self.source}: no oracle-declared control for agent {agent_id!r}; "
            f"this oracle declares {sorted(self.agent_ids)}"
        )

    def replacing(self, control: NegativeControl) -> NegativeControlOracle:
        """The same oracle with one control swapped in, for testing the gate's teeth.

        Appends the control if this oracle has none for that agent. Deliberately
        returns a new oracle rather than mutating this one: the loaded manifest
        is shared, and a test that could edit it in place would be able to
        weaken the gate for everything that ran after it.
        """
        if not self.has_control(control.agent_id):
            return self._with_controls((*self.controls, control))
        return self._with_controls(
            tuple(
                control if existing.agent_id == control.agent_id else existing
                for existing in self.controls
            )
        )

    def without(self, agent_id: str) -> NegativeControlOracle:
        """The same oracle with one control removed."""
        return self._with_controls(
            tuple(existing for existing in self.controls if existing.agent_id != agent_id)
        )

    def _with_controls(
        self, controls: tuple[NegativeControl, ...]
    ) -> NegativeControlOracle:
        # Deliberately not re-validated. This is the seam the suite uses to hand
        # the gate an expectation that is wrong on purpose, including wrong in
        # ways the loader would refuse from a file — a gate that only ever sees
        # well-formed input has not been shown to check anything.
        payload = {**self.as_dict(), "controls": [c.as_dict() for c in controls]}
        return NegativeControlOracle(
            oracle_id=self.oracle_id,
            oracle_version=self.oracle_version,
            operation_id=self.operation_id,
            semantic_scenario_id=self.semantic_scenario_id,
            authored_against_operation_version=(self.authored_against_operation_version),
            provenance=dict(self.provenance),
            result_vector_dimensions=self.result_vector_dimensions,
            controls=controls,
            oracle_digest_sha256=_digest(payload, self.source),
            source=self.source,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "oracle_id": self.oracle_id,
            "oracle_version": self.oracle_version,
            "operation_id": self.operation_id,
            "semantic_scenario_id": self.semantic_scenario_id,
            "authored_against_operation_version": (
                self.authored_against_operation_version
            ),
            "provenance": dict(self.provenance),
            "result_vector_dimensions": list(self.result_vector_dimensions),
            "controls": [control.as_dict() for control in self.controls],
        }

    def identity(self) -> dict[str, Any]:
        """What a report records so a reader knows which oracle graded a run."""
        return {
            "oracle_id": self.oracle_id,
            "oracle_version": self.oracle_version,
            "oracle_digest_sha256": self.oracle_digest_sha256,
        }


def _parse_control(raw: Any, vocabulary: tuple[str, ...], where: str) -> NegativeControl:
    payload = _mapping(raw, where)
    _fields(payload, _CONTROL_FIELDS, where)
    agent_id = _text(payload, "agent_id", where)
    named = f"{where} ({agent_id})"

    failed = _names(payload, "expected_failed_dimensions", named)
    passing = _names(payload, "must_pass_dimensions", named)

    unknown = sorted((set(failed) | set(passing)) - set(vocabulary))
    if unknown:
        raise OracleManifestError(
            f"{named}: names {unknown}, which are not in this oracle's declared "
            f"result_vector_dimensions {list(vocabulary)}"
        )
    both = sorted(set(failed) & set(passing))
    if both:
        raise OracleManifestError(
            f"{named}: {both} are declared as both failing and passing; a control "
            "cannot declare a dimension in two states at once"
        )
    unjudged = sorted(set(vocabulary) - set(failed) - set(passing))
    if unjudged:
        raise OracleManifestError(
            f"{named}: {unjudged} are left unjudged; every control states an "
            "expectation for every dimension, so a failure has nowhere to hide"
        )

    return NegativeControl(
        agent_id=agent_id,
        scenario_id=_text(payload, "scenario_id", named),
        intervention=_text(payload, "intervention", named),
        guarantee_attacked=_text(payload, "guarantee_attacked", named),
        expected_failed_dimensions=failed,
        required_finding_codes=_names(payload, "required_finding_codes", named),
        must_pass_dimensions=passing,
        rationale=_text(payload, "rationale", named),
    )


def _digest(payload: Mapping[str, Any], source: str) -> str:
    """SHA-256 over the manifest's parsed semantic content, not its file bytes.

    A comment, a reflow or a change of quoting style does not move it; an edit
    to an expectation does.
    """
    try:
        return hashlib.sha256(
            canonical_json_bytes(dict(payload), f"{source}: negative-control oracle")
        ).hexdigest()
    except JsonSafetyError as exc:
        raise OracleManifestError(
            f"{source}: this oracle cannot be given a content identity: {exc}"
        ) from exc


def _build(payload: Mapping[str, Any], source: str) -> NegativeControlOracle:
    _fields(payload, _TOP_LEVEL_FIELDS, source)
    provenance_raw = _mapping(payload["provenance"], f"{source}: provenance")
    _fields(provenance_raw, _PROVENANCE_FIELDS, f"{source}: provenance")
    provenance = {
        key: _text(provenance_raw, key, f"{source}: provenance")
        for key in _PROVENANCE_FIELDS
    }
    vocabulary = _names(payload, "result_vector_dimensions", source)
    raw_controls = payload["controls"]
    if not isinstance(raw_controls, list) or not raw_controls:
        raise OracleManifestError(
            f"{source}: controls must be a non-empty list of oracle-declared "
            "negative controls"
        )
    controls: list[NegativeControl] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_controls):
        control = _parse_control(raw, vocabulary, f"{source}: controls[{index}]")
        if control.agent_id in seen:
            raise OracleManifestError(
                f"{source}: agent {control.agent_id!r} is oracle-declared twice; "
                "one control per agent, or the gate has two answers"
            )
        seen.add(control.agent_id)
        controls.append(control)

    return NegativeControlOracle(
        oracle_id=_text(payload, "oracle_id", source),
        oracle_version=_text(payload, "oracle_version", source),
        operation_id=_text(payload, "operation_id", source),
        semantic_scenario_id=_text(payload, "semantic_scenario_id", source),
        authored_against_operation_version=_text(
            payload, "authored_against_operation_version", source
        ),
        provenance=provenance,
        result_vector_dimensions=tuple(vocabulary),
        controls=tuple(controls),
        oracle_digest_sha256=_digest(payload, source),
        source=source,
    )


def oracle_manifest_path() -> Path:
    """Where the packaged manifest lives, in a source tree or in a wheel."""
    return Path(str(resources.files(ORACLE_RESOURCE_PACKAGE) / ORACLE_RESOURCE_NAME))


def load_negative_control_oracle(path: str | Path) -> NegativeControlOracle:
    """Read and validate a negative-control manifest from disk."""
    source = str(path)
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise OracleManifestError(
            f"cannot read negative-control oracle {source}: {exc}"
        ) from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise OracleManifestError(f"{source}: not readable as YAML: {exc}") from exc
    return _build(_mapping(raw, source), source)


_CACHE: dict[str, NegativeControlOracle] = {}


def negative_control_oracle() -> NegativeControlOracle:
    """The shipped Maintenance oracle, loaded once per process.

    Cached because the gate reads it for every check and a manifest that could
    change between two checks in one run would make the report incoherent.
    """
    cached = _CACHE.get(ORACLE_RESOURCE_NAME)
    if cached is None:
        cached = load_negative_control_oracle(oracle_manifest_path())
        _CACHE[ORACLE_RESOURCE_NAME] = cached
    return cached


def oracle_declared_expectations(agent_id: str) -> NegativeControl | None:
    """The shipped control for an agent, or ``None`` if it is not oracle-declared.

    This is the only way the agent registry learns what a negative is expected
    to break. It cannot write, and there is nothing here to override.
    """
    oracle = negative_control_oracle()
    if not oracle.has_control(agent_id):
        return None
    return oracle.control(agent_id)


__all__ = [
    "ORACLE_RESOURCE_NAME",
    "ORACLE_RESOURCE_PACKAGE",
    "NegativeControl",
    "NegativeControlOracle",
    "load_negative_control_oracle",
    "negative_control_oracle",
    "oracle_declared_expectations",
    "oracle_manifest_path",
]
