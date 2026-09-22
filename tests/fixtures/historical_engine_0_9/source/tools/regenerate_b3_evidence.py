"""Regenerate the B3.1 frozen Artifact 8 / execution-ledger 2 evidence bundle.

The bundle under ``tests/fixtures/b3`` is a static historical oracle: the suite
reads and audits it, and :mod:`tests.test_artifact8_evidence_freeze` says in its
first paragraph that regenerating it is an explicit maintenance operation and
never test setup. Until this module existed there was no command that performed
that operation, so "regenerate the bundle" named a step nobody could take, and a
moved operation identity could only be carried into the oracle by editing frozen
bytes — the in-place edit of preserved evidence that ``docs/VERSIONING.md``
forbids. This is that missing command.

**It reaches a provider never, and a credential never.** Every answer comes from
the shipped reference agent's own decisions, replayed through the same in-process
``ScriptedProvider`` over ``httpx.MockTransport`` that
:mod:`tools.preflight_lifecycle_provider_evidence` uses, and that module's
environment refusal runs first here too. The model cell is literally that
preflight's ``reference_cell``, at the cost cap the committed ledger states, so
what comes out is the same document rebuilt rather than a second,
differently-parameterised one wearing its name.

**Four things are held still that a normal run must never let a caller name.**
``run_episode`` mints the operation instance identity itself and
``ProviderEvidenceRecorder.begin`` mints the execution run identity, both so two
runs can never share a label a caller chose; the ledger instant and the recorded
latencies are wall-clock; and the journal stamps whatever ledger contract the
build currently writes. Those refusals are right for running the benchmark and
wrong for a frozen oracle, which has to come out byte-identical every time it is
rebuilt or the pins it feeds are worth nothing. So this module substitutes the
two minters, the clocks, and the ledger contract for the duration of one
generation and restores them afterwards — including if the generation raises.
Nothing else in the build may do this, and it is only safe here because this
bundle is evidence *about a fixture*: it is never scored, pooled, compared or
published as a measurement of anything.

The ledger contract is pinned to **2** on purpose. The suite asserts the frozen
ledger is a version-2 document — that is what the bundle *is*, and its filename
says so — while the build has since moved its writer to 3. The build still owns
the complete version-2 vocabulary for reading, and a fault-free reference run
uses nothing version 3 added, so writing it under 2 produces a valid version-2
ledger rather than a version-3 one relabelled. Regenerating as 3 would silently
turn a historical oracle into a current one, which is the one thing a
regeneration must never do.

Two modes, and the second is the one CI wants::

    uv run python -m tools.regenerate_b3_evidence --output-dir DIR
    uv run python -m tools.regenerate_b3_evidence --check

``--check`` regenerates into a temporary directory and compares the result with
the committed bundle byte for byte, writing nothing. A drift between the shipped
oracle and what this build actually produces is then a finding, rather than a
surprise the next maintainer meets while trying to update a pin.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from operatebench import execution_ledger as ledger_module
from operatebench import runner as runner_module
from operatebench.agents import evidence as evidence_module
from operatebench.agents.evidence import ProviderEvidenceRecorder, WireCaptureTransport
from operatebench.agents.openai_responses import OpenAIResponsesTransport
from operatebench.agents.pricing import RATE_SOURCE_OPERATOR, LifecyclePricingPolicy
from operatebench.artifact import read_artifact, write_artifact
from operatebench.domains.lettings.maintenance.spec import OperationSpec, load_spec
from operatebench.execution_ledger import (
    EXECUTION_LEDGER_VERSION_V2,
    read_execution_ledger,
)
from operatebench.runner import run_episode
from tools.preflight_lifecycle_provider_evidence import (
    DEFAULT_MAX_PROVIDER_CALLS,
    FIXTURE,
    PreflightFailure,
    reference_cell,
    refuse_live_environment,
)

ROOT = Path(__file__).resolve().parents[1]
UV_LOCK = ROOT / "uv.lock"
FIXTURES_DIR = ROOT / "tests" / "fixtures" / "b3"
IDENTITY_RESOURCES = ROOT / "src" / "operatebench" / "resources" / "identity"
GOLDEN_RESOURCE = IDENTITY_RESOURCES / "identity-manifest-digest-v1.golden.json"
COMPONENT_REGISTRY = IDENTITY_RESOURCES / "identity-component-registry-v1.json"
COMPATIBILITY_REGISTRY = IDENTITY_RESOURCES / "identity-compatibility-v1.registry.json"

MANIFEST_NAME = "artifact8-evidence-freeze-v1.json"
DETERMINISTIC_NAME = "artifact8-deterministic-reference-v1.json"
MODEL_NAME = "artifact8-fake-openai-reference-v1.json"
LEDGER_NAME = "artifact8-fake-openai-reference-v1.execution-ledger-v2.ndjson"

SCHEMA = "operatebench.b3.artifact8_evidence_freeze.v1"
SCHEMA_VERSION = 1

#: The identities this bundle is pinned to. Deliberately not random and
#: deliberately not plausible: a run id of counted hex is one nobody can mistake
#: for a measured execution, and an instant a decade out is one no reader will
#: take for the date this evidence was gathered.
OPERATION_INSTANCE_ID = "opinst_0123456789abcdef0123456789abcdef"
EXECUTION_RUN_ID = "exec_0123456789abcdef0123456789abcdef"
LEDGER_INSTANT = "2035-01-02T03:04:05Z"

#: The step a pinned provider call advances the monotonic clock by. Real
#: latency is a measurement, and a measurement in a frozen oracle is a byte
#: that changes every rebuild for reasons that say nothing about the fixture.
#: A fixed tick makes the recorded latencies obviously synthetic and the
#: bundle reproducible, which is the only property this document needs.
CLOCK_TICK_SECONDS = 0.25

#: The envelope the committed ledger states in its own header row. Restated
#: here rather than defaulted from the preflight, because a regeneration under a
#: different cap or rate table would be a different document that happened to
#: reuse the filename.
COST_CAP_USD = Decimal("5.00")
INPUT_USD_PER_MTOK = Decimal("1.25")
OUTPUT_USD_PER_MTOK = Decimal("10.00")
PRICING_POLICY_ID = "offline_operator_pinned_v1"
MAX_PROVIDER_CALLS = DEFAULT_MAX_PROVIDER_CALLS

#: The ledger contract this bundle is frozen under. Not the writer's current
#: version: the suite asserts a version-2 document and the filename says so.
LEDGER_CONTRACT = EXECUTION_LEDGER_VERSION_V2

SCENARIO_ID = "V1"
REFERENCE_AGENT_ID = "reference"


class RegenerationFailure(Exception):
    """The bundle could not be produced, or did not come out as it must."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(payload: Any) -> bytes:
    """The one encoding the freeze manifest and both artefacts are written in."""
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


class _PinnedGeneration:
    """Hold every non-deterministic input still for one generation.

    Six seams: the two identity minters; the three constructors that would
    otherwise reach for a wall clock — the recorder's and the wire capture's
    instant, and the transport's monotonic clock that becomes the recorded
    latencies; and the ledger contract the journal stamps on every row.

    A context manager rather than a module-level patch, so the refusals the
    minters implement are absent for exactly as long as the bundle is being
    written and present everywhere else — including when the generation raises.

    The two minters are replaced in the namespace of the module that *calls*
    each one, not the one that defines it: both are imported by name at import
    time, so substituting on the defining module would patch something nothing
    reads. The ledger contract is the opposite case — the journal reads it from
    its own module's globals at construction — so that one is held on the
    defining module.
    """

    #: (module or class, attribute) pairs whose values are held still. Named as
    #: data so restoring is the same loop as substituting and cannot drift from
    #: it; ``setattr`` because these are deliberate rebindings of names the type
    #: checker is right to consider fixed everywhere else.
    _SEAMS = (
        (runner_module, "new_operation_instance_id"),
        (evidence_module, "new_execution_run_id"),
        (ProviderEvidenceRecorder, "__init__"),
        (WireCaptureTransport, "__init__"),
        (OpenAIResponsesTransport, "__init__"),
        (ledger_module, "EXECUTION_LEDGER_VERSION"),
    )

    def __enter__(self) -> _PinnedGeneration:
        self._saved = [getattr(owner, name) for owner, name in self._SEAMS]
        recorder_init, wire_init, transport_init = self._saved[2:5]
        ticks = itertools.count()

        def pinned_clock() -> float:
            return next(ticks) * CLOCK_TICK_SECONDS

        def pinned_instant() -> str:
            return LEDGER_INSTANT

        def pinned_recorder_init(inner: Any, **kwargs: Any) -> None:
            kwargs["now"] = pinned_instant
            recorder_init(inner, **kwargs)

        def pinned_wire_init(inner: Any, **kwargs: Any) -> None:
            kwargs["now"] = pinned_instant
            wire_init(inner, **kwargs)

        def pinned_transport_init(inner: Any, **kwargs: Any) -> None:
            kwargs["clock"] = pinned_clock
            transport_init(inner, **kwargs)

        replacements = (
            lambda: OPERATION_INSTANCE_ID,
            lambda: EXECUTION_RUN_ID,
            pinned_recorder_init,
            pinned_wire_init,
            pinned_transport_init,
            LEDGER_CONTRACT,
        )
        for (owner, name), value in zip(self._SEAMS, replacements, strict=True):
            setattr(owner, name, value)
        return self

    def __exit__(self, *exc: object) -> None:
        for (owner, name), value in zip(self._SEAMS, self._saved, strict=True):
            setattr(owner, name, value)


def _deterministic_cell(spec: OperationSpec, directory: Path) -> Path:
    """The reference agent's own run: no provider, no ledger, no binding."""
    run = run_episode(spec, SCENARIO_ID, REFERENCE_AGENT_ID)
    recorded = run.operation_instance_id
    if recorded != OPERATION_INSTANCE_ID:
        raise RegenerationFailure(
            "the deterministic run did not take the pinned operation instance "
            f"identity; it recorded {recorded!r}"
        )
    path = directory / DETERMINISTIC_NAME
    write_artifact(run, path)
    return path


def _identity_block() -> dict[str, Any]:
    """Carried from the packaged Identity v1 resources, which this never touches.

    The component and composite digests belong to that golden resource, not to
    the operation, so a moved operation identity leaves every one of them where
    it is. The three resource hashes are recomputed from the files on disk, so
    the manifest stays honest if one of those resources ever does move.
    """
    golden = json.loads(GOLDEN_RESOURCE.read_text(encoding="utf-8"))
    worked = golden["manifest"]
    manifest = worked["manifest"]
    return {
        "compatibility_registry_resource_sha256": _sha(
            COMPATIBILITY_REGISTRY.read_bytes()
        ),
        "component_registry_resource_sha256": _sha(COMPONENT_REGISTRY.read_bytes()),
        "components": {
            name: row["content_digest_sha256"]
            for name, row in sorted(manifest["components"].items())
        },
        "composites": {
            name: row["bundle_digest_sha256"]
            for name, row in sorted(manifest["composites"].items())
        },
        "golden_resource_sha256": _sha(GOLDEN_RESOURCE.read_bytes()),
        "manifest_digest_sha256": worked["digest_sha256"],
    }


def _build(directory: Path) -> dict[str, Any]:
    """Produce the four files into an empty ``directory``; return the manifest.

    Every writer underneath creates its file exclusively and refuses one that
    already exists — a run artefact is written once so a replay can never be
    compared against a rewritten record. That is the right contract for
    evidence and the wrong one for a regeneration command, whose whole purpose
    is to replace what is there. So this builds into a directory that holds
    nothing, and :func:`generate` is what moves the result into place.
    """
    refuse_live_environment()
    spec = load_spec(FIXTURE)
    spec.verify_identity()

    with _PinnedGeneration():
        deterministic_path = _deterministic_cell(spec, directory)
        policy = LifecyclePricingPolicy(
            policy_id=PRICING_POLICY_ID,
            input_usd_per_mtok=INPUT_USD_PER_MTOK,
            output_usd_per_mtok=OUTPUT_USD_PER_MTOK,
            rate_source=RATE_SOURCE_OPERATOR,
        )
        cell = reference_cell(
            spec,
            directory=directory,
            policy=policy,
            cap=COST_CAP_USD,
            max_calls=MAX_PROVIDER_CALLS,
        )

    model_path = (directory / cell["artifact"]).rename(directory / MODEL_NAME)
    ledger_path = (directory / cell["ledger"]).rename(directory / LEDGER_NAME)

    # Read all three back through the readers the suite itself uses, so a bundle
    # that cannot be audited is never written out as though it could be.
    read_artifact(deterministic_path)
    read_artifact(model_path)
    audit = read_execution_ledger(ledger_path, require_complete=True)

    deterministic_raw = deterministic_path.read_bytes()
    model_raw = model_path.read_bytes()
    ledger_raw = ledger_path.read_bytes()

    manifest = {
        "fixtures": {
            "deterministic": {
                "artifact": DETERMINISTIC_NAME,
                "artifact_bytes": len(deterministic_raw),
                "artifact_sha256": _sha(deterministic_raw),
                "ledger": None,
            },
            "fake_model": {
                "artifact": MODEL_NAME,
                "artifact_bytes": len(model_raw),
                "artifact_sha256": _sha(model_raw),
                "ledger": LEDGER_NAME,
                "ledger_bytes": len(ledger_raw),
                "ledger_chain_sha256": audit.ledger_digest_sha256,
                "ledger_file_sha256": _sha(ledger_raw),
            },
        },
        "generation": {
            "execution_run_id": EXECUTION_RUN_ID,
            "ledger_instant": LEDGER_INSTANT,
            "operation_fixture": "examples/operatebench/maintenance_v0_1.yaml",
            "operation_fixture_sha256": _sha(Path(FIXTURE).read_bytes()),
            "operation_instance_id": OPERATION_INSTANCE_ID,
            "uv_lock_sha256": _sha(UV_LOCK.read_bytes()),
        },
        "identity": _identity_block(),
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
    }
    (directory / MANIFEST_NAME).write_bytes(_canonical(manifest))
    return manifest


BUNDLE_NAMES = (MANIFEST_NAME, DETERMINISTIC_NAME, MODEL_NAME, LEDGER_NAME)


def generate(directory: Path) -> dict[str, Any]:
    """Write the four bundle files into ``directory``, replacing any already there.

    The bundle is built whole in a fresh staging directory beside nothing, read
    back through the suite's own readers, and only then moved over the
    destination names one file at a time with ``os.replace``. A generation that
    fails leaves ``directory`` exactly as it was: nothing is touched until the
    complete bundle exists and has been audited. This is what lets the
    documented command be run against ``tests/fixtures/b3`` itself, or re-run
    after an interrupted attempt, rather than aborting on the first name that
    already exists.
    """
    with tempfile.TemporaryDirectory(dir=directory) as raw:
        scratch = Path(raw)
        scratch.chmod(0o700)
        manifest = _build(scratch)
        for name in BUNDLE_NAMES:
            os.replace(scratch / name, directory / name)
    return manifest


def check(directory: Path = FIXTURES_DIR) -> list[str]:
    """Regenerate into a temporary directory and diff against ``directory``."""
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as raw:
        scratch = Path(raw)
        scratch.chmod(0o700)
        _build(scratch)
        for name in BUNDLE_NAMES:
            fresh = (scratch / name).read_bytes()
            committed_path = directory / name
            if not committed_path.exists():
                problems.append(f"{name}: not committed")
                continue
            committed = committed_path.read_bytes()
            if fresh != committed:
                problems.append(
                    f"{name}: this build produces {len(fresh)} bytes "
                    f"sha256 {_sha(fresh)}; the committed oracle is "
                    f"{len(committed)} bytes sha256 {_sha(committed)}"
                )
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the frozen B3.1 Artifact 8 evidence bundle. Makes no live "
            "provider call and reads no credential."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "an existing directory the four bundle files are written into; files "
            "of those names already there are replaced, and nothing is touched "
            "unless the whole bundle was built and audited first"
        ),
    )
    group.add_argument(
        "--check",
        action="store_true",
        help=(
            "regenerate into a temporary directory and report any difference from "
            "the committed bundle; writes nothing"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.check:
            problems = check()
            if problems:
                for problem in problems:
                    print(problem, file=sys.stderr)
                return 2
            print("the committed B3.1 bundle is exactly what this build produces")
            return 0
        directory: Path = args.output_dir
        if not directory.is_dir():
            print(f"{directory}: not an existing directory", file=sys.stderr)
            return 64
        manifest = generate(directory)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except (RegenerationFailure, PreflightFailure) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - entrypoint
    raise SystemExit(main())
