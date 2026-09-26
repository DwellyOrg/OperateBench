"""Version identity for the OperateBench slice.

Two numbers live here, and they are two numbers on purpose.

``__version__`` is the **distribution** version — the wheel that ships, read by
Hatch's dynamic-version source and published in the installed distribution's
own metadata. No artefact carries it: which build produced a run is packaging
provenance, not the identity of the run. ``OPERATEBENCH_VERSION`` is the
**engine** version — the runtime and the evaluator together, recorded in every
artefact as ``engine_version`` and answering a different question: under which
execution and grading semantics was this result produced.

Deriving one from the other looks tidy and is wrong. A packaging fix moves the
distribution and must not invalidate comparability; a change to event dispatch
or to what a dimension asserts moves the engine and *does* invalidate it, with
no new wheel required. They are separate identities in
``docs/VERSIONING.md``, so they are separate literals here, and the release
scanner pins both by value rather than to each other.

A leaf module so any recorder can import it without pulling the package graph
in.
"""

from __future__ import annotations

#: Distribution version of the ``operatebench`` wheel. The one literal Hatch's
#: dynamic-version source reads out of this file. Provenance only: it is not the
#: identity of an operation, a runtime contract or an evaluator.
__version__ = "0.1.0"

#: Engine version: the runtime and the evaluator, together, as recorded in every
#: artefact's ``engine_version``. Moved to 0.2.0 when event delivery, the
#: ledger↔delivery binding and the result vector changed under distribution
#: 0.1.0, and to 0.3.0 when every accepted effect began recording the canonical
#: identities it established and the evaluator began re-deriving invoice
#: validation from them rather than from the final state. Both halves moved:
#: the runtime writes a field it did not write, and the evaluator refuses runs
#: it used to call reliable.
#:
#: 0.5.0 added tool-mediated reading: the runtime serves a published catalogue
#: through a batch object, records what it served with per-result provenance, and
#: invalidates every derived fact on a wake and on the agent's own accepted
#: effect. The normalized projection of the record was still handed over beside
#: it, so nothing about *what an agent was told* had changed yet.
#:
#: 0.6.0 is where it changes, and both halves move again. The runtime no longer
#: publishes the operation record to the agent at all — the observation carries a
#: coarse phase, the waking event as a claim, and the read affordances — and it
#: refuses a business proposal whose canonical required reads are absent, stale
#: or answered only by what a participant asserted. The evaluator grades a
#: dimension it did not grade, reconstructing the binding between each proposal
#: and the reads that established it from the record alone. Runs produced by
#: 0.5.0 and 0.6.0 are not comparable: the same agent is being asked a different
#: question.
#:
#: 0.4.0 moves both halves again, and changes what a result *means*. The runtime
#: no longer discloses the scenario identity to the agent — the observation
#: carries an opaque per-run operation instance identity instead — so a decision
#: is no longer made by something that could look the case up; and a WAIT is no
#: longer refused because nothing pending can deliver it, because what is pending
#: is authored-future truth the agent cannot see. A wait for a known event type
#: is accepted and time decides, so runs that used to end at the declaration now
#: end at the operational horizon. Artefacts produced by 0.1.0, 0.2.0, 0.3.0 and
#: 0.4.0 are not comparable with each other.
#: 0.8.0 corrects Boundary-arm evaluator ownership: once a subject starts the
#: target timestamp, only that subject's decisions can establish the point. The
#: operation, Core runtime, protocol and artefact shape are unchanged; the
#: combined preview identity still moves because it also names the evaluator.
#: 0.9.0 corrects Full-arm terminal suffix mapping: only a structurally valid,
#: digest-bound Engine outcome ending in deadlock or horizon can establish that
#: absent later boundaries were genuinely unreached. Raw or contradictory rows
#: fail closed as mapping errors instead.
#: 0.10.0 enforces invocation admission before event and fallback wakes.
#: Artifact 8 remains readable across engines, but replay requires exact engine
#: identity and refuses older trajectories before re-execution.
#: 0.13.0 repairs Maintenance unsolicited-wake grading and new completion duties.
#: Domain guidance moves with this identity; old reads preserve stored grades
#: and replay refuses foreign engines rather than silently regrading.
OPERATEBENCH_VERSION = "0.13.0"

#: Frozen legacy alias for the original packaged Card/Census contract.  It stays
#: at one so existing root imports and v1 callers remain byte- and API-stable;
#: additive v2 contracts use the card-family literals below instead.
CARD_SCHEMA_VERSION = 1

#: Internal module-level Card contract versions.  These are intentionally not
#: re-exported from :mod:`operatebench`: schema/version dispatch is per family,
#: while ``CARD_SCHEMA_VERSION`` remains the frozen public v1 compatibility alias.
OPERATION_CARD_V1_SCHEMA_VERSION = 1
SEMANTIC_SCENARIO_CARD_V1_SCHEMA_VERSION = 1
VARIANT_CARD_V1_SCHEMA_VERSION = 1
OPERATION_CARD_V2_SCHEMA_VERSION = 2
SEMANTIC_SCENARIO_CARD_V2_SCHEMA_VERSION = 2
VARIANT_CARD_V2_SCHEMA_VERSION = 2

__all__ = [
    "CARD_SCHEMA_VERSION",
    "OPERATEBENCH_VERSION",
    "__version__",
]
