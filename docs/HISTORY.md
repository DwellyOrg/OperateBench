# OperateBench — Naming, Versioning and Historical Continuity

**Status:** authoritative naming policy
**Date:** 10 August 2026
**Public project:** OperateBench
**Historical project and implementation:** BoundaryBench

## Canonical relationship

OperateBench is not a rename of BoundaryBench.

BoundaryBench is the existing benchmark and methodology for consequential
operational decisions: given state, policy, authority, evidence and pressure,
should an agent act, stop, ask, wait or escalate? Its code, Cubes, manifests,
ledgers, audits and runs remain BoundaryBench artifacts.

OperateBench is the broader project introduced after the authors chose to study
longitudinal operation ownership as a synthetic benchmark construct. Its primary
construct follows an authored case across simulated time, actors, persistent
state, external events, waits, approvals, loops and recovery. This motivation is
not evidence that AI ran Dwelly operations, and the current Maintenance fixture
does not encode or describe a Dwelly workflow or policy.

The repository currently publishes one fully invented Maintenance fixture. Its
actors, records, policies, thresholds, events and expected outcomes were authored
for evaluation. It contains no production records or internal procedures and does
not purport to encode production activity. Future domain packs and their order are
intentionally unspecified pending public contribution and construct validation.

BoundaryBench becomes the foundation of the **Boundary Track** inside
OperateBench. This is product continuity and reuse, not retroactive renaming.

## Public terminology

Use:

- **OperateBench** — top-level longitudinal operations benchmark;
- **OperateBench Core** — domain-neutral operation/event/time/state/actor/ledger
  protocol;
- **Lifecycle Track** — whole-operation longitudinal evaluation and the first
  differentiating track;
- **Boundary Track** — BoundaryBench methodology for consequential operational
  boundaries;
- **BoundaryBench** — the real historical project, implementation, package and
  evidence corpus inherited by the Boundary Track.

Do not use:

- “BoundaryBench was renamed to OperateBench”;
- “BoundaryBench was always called OperateBench”;
- “OperateBench v0.x” as a retroactive label for BoundaryBench releases or runs;
- “BoundaryBench was abandoned or replaced.”

Preferred wording:

> BoundaryBench established this project's deterministic decision-boundary
> methodology and the runtime infrastructure underneath it. OperateBench
> preserves it as the Boundary Track while adding a longitudinal
> operation-ownership construct on top.

"This project's", not "the first": deterministic boundary evaluation has prior
art, and claiming primacy for it would fail the same review that
[RELATED_WORK.md](RELATED_WORK.md) exists to pass.

## Historical artifact rule

Do not rewrite or relabel historical evidence. The following remain unchanged:

- the `boundarybench` Python package and module names;
- Git commits and historical document titles;
- benchmark, suite, scaffold, configuration and execution ids;
- manifests, ledgers, run directories, audit filenames and hashes;
- provider request profiles, version numbers and report quotations.

When cited from OperateBench, use:

> “BoundaryBench evidence, reused by the OperateBench Boundary Track.”

Never claim that a BoundaryBench run executed the newer OperateBench Core
protocol. Compatibility must replay each artifact under its original contract.

## Package layout

The `boundarybench` package stays where it is and keeps its name. OperateBench
Core and the track/domain boundaries are introduced *around* it rather than by
moving it, so no historical import path, module name or artifact identity
changes underneath an existing run.

A future move of the package under `tracks/boundary` would require compatibility
tests proving that every historical artifact still replays under its original
contract. It is not needed for the vertical slice, and it is not done here.

## History policy

This local successor candidate is curated from tracked source without importing
predecessor Git objects. Historical test dependencies are separately byte-pinned
as documented in [HISTORICAL_RUNTIME.md](HISTORICAL_RUNTIME.md). This is not yet a
published repository; exact public-commit review and human approvals remain pending.

Corrections to naming are made forward, in the public documents, and recorded
under [VERSIONING.md](VERSIONING.md). Rewriting published history to hide a
naming or scoping correction is not acceptable: the history of a benchmark's
construct is part of its evidence.

## One-sentence descriptions

**OperateBench**

> OperateBench evaluates whether an AI agent can own a longitudinal,
> event-driven business operation as time, actors and authoritative state change.

**Boundary Track**

> The Boundary Track, inherited from BoundaryBench, evaluates whether an agent
> makes the right consequential decision at a specific operational boundary.
