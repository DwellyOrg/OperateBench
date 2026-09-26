"""The run artefact, and replay from it.

One file, canonical UTF-8 JSON, sorted keys, sufficient for exact replay. It
carries the identity of what was run (operation, spec digest, scenario, agent,
engine version), the semantic input as delivered (every event with its actor,
instant, cause and disposition), the trajectory (proposal, refusal, accepted
effect, side effect, checkpoint, obligation, terminal), the final canonical
state and the whole evaluation vector.

Three things this module treats as adversarial.

**The file is untrusted input.** :func:`read_artifact` holds it to an exact
top-level allowlist and a recursive shape: every field this build writes and no
field it does not, with its type, its range and — for instants, digests and
counters — its form. ``operation: []`` is a named refusal here rather than an
``AttributeError`` two frames later, and a lone surrogate that JSON can escape
but UTF-8 cannot encode is refused rather than hashed.

**Provenance is a cross-field claim, and it is checked as one.** Who ran, what
produced the decisions, how many provider calls were made and what each
attempt ended as are four statements about one execution, and a reader that
holds each to its own shape and never compares them will accept a record that
is well-formed and describes no run at all. So the execution record, the
decision tape and the agent identity are bound to each other here
(:func:`_validate_provenance`): a ``model`` record names a model and a protocol
this build speaks, its calls, attempts and decisions have the same cardinality
and the same per-decision invocation and turn, each attempt's classification
agrees with the kind of the decision beside it, and an in-process or recorded
execution carries no model, no protocol, no calls and no attempts. Relabelling
a deterministic run as a model run — the cheapest forgery there is — fails on
all of them at once, and each mismatch has its own named refusal.

**Replay compares the whole result.** :func:`replay_artifact` does not read the
recorded run back and agree with it, and it does not compare three fields of it.
It re-executes the spec, scenario and agent the artefact names, rebuilds the
*complete* artefact that rerun would have written, and compares every
result-bearing field — identity, metadata, delivered events, trajectory, final
state, digests and evaluation — reporting section by section so a divergence
says what diverged.

**The path is created, not written to.** :func:`write_artifact` refuses a final
symlink and every symlinked ancestor, then creates the file with
``O_CREAT|O_EXCL|O_NOFOLLOW``, so there is no window between deciding a path is
safe and using it, and no way for a link left at the output path to redirect the
write onto a file the operator never named.

**An excluded execution has no episode artefact.** A run that stopped at the
provider boundary produced no business outcome, so there is nothing for the
evaluator to score and nothing for a terminal to be. This build therefore
writes no artefact for one — :class:`~operatebench.agents.transport.ProviderFailure`
propagates out of the runner before an artefact exists — and, fail-closed, it
refuses to *read* one: ``excluded=True`` is a named refusal
(:data:`_NO_EXCLUDED_ARTEFACT`) rather than a shape that could carry a reliable,
legitimate, completed evaluation beside a provider fault. Recording an excluded
execution needs an execution ledger of its own, whose result-bearing fields are
the attempts and the fault rather than a terminal; until that exists the honest
answer is a refusal, not a permissive field nothing can reach.

What no digest here claims is immutability or tamper-evidence. These are content
digests: they answer "is this the same content?", and nothing at all about who
could have changed the file.

The provenance binding above inherits that limit exactly, and it is worth stating
rather than leaving to be discovered. What it establishes is *consistency*: no
record reaches a caller whose execution, tape and identity contradict each other,
and the provider identity a model run names is part of what its request digests
hash to rather than a label beside them. What it cannot establish is
*attestation*. A party who can run this build can compute the same digests from
the same observations and mint a record that is consistent throughout, because
nothing in a self-describing local file distinguishes a provider that answered
from a provider that was never asked. Closing that needs evidence the provider
signs, which is not something this format holds. So the claim here is the one
that is true — an inconsistent record is refused by name — and not the one that
would be more useful.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from pathlib import Path
from typing import Any

from boundarybench.jsonsafe import (
    JsonSafetyError,
    NestingDepthError,
    canonical_json_text,
    ensure_json_safe,
    ensure_raw_json_depth,
)
from operatebench._write_once import (
    _write_once_bytes,
    _write_once_bytes_at,
    _WriteOnceFailure,
)
from operatebench.agents.model import (
    MODEL_PROTOCOL_VERSION,
    MODEL_PROTOCOL_VERSION_V1,
    MODEL_PROTOCOL_VERSION_V2,
    MODEL_PROTOCOL_VERSION_V3,
    READABLE_CLASSIFICATIONS,
)
from operatebench.agents.playback import (
    DECISION_FIELDS,
    MALFORMED_KIND,
    RecordedModelAgent,
    RecordedOutcomeAgent,
    tape_from_records,
)
from operatebench.agents.transport import (
    FAULTS,
    OUTCOME_SOURCES,
    RETRY_COUNT,
    SOURCE_DETERMINISTIC,
    SOURCE_MODEL,
    SOURCE_RECORDED,
    execution_from_record,
)
from operatebench.core.clock import parse_timestamp
from operatebench.core.engine import EpisodeOutcome
from operatebench.core.errors import (
    ArtifactError,
    MalformedTimestampError,
    OperateBenchError,
)
from operatebench.core.instance import operation_instance_id_problem
from operatebench.core.ledger import RECORD_TYPES, RETRIEVAL_RECORD_TYPES
from operatebench.core.outcomes import OUTCOME_KINDS
from operatebench.core.retrieval import (
    INVALIDATION_CAUSES,
    RETRIEVAL_AUTHORITIES,
    RETRIEVAL_INITIATORS,
    RETRIEVAL_REFUSAL_CODES,
    RETRIEVE_KIND,
)
from operatebench.domains.lettings.maintenance.agents import AGENTS
from operatebench.domains.lettings.maintenance.evaluator import DIMENSIONS
from operatebench.domains.lettings.maintenance.spec import (
    TERMINAL_OUTCOMES,
    OperationSpec,
)
from operatebench.domains.lettings.maintenance.state import validate_canonical_state

# The ledger contract this build's bindings are evidence for, and the identity
# rule they are held to. Imported rather than restated: a binding that named a
# journal shape by a number kept in step by hand is a binding to whichever
# shape somebody last remembered. Nothing about this import reaches the sidecar
# — :mod:`operatebench.execution_ledger` opens no file until it is asked to,
# and this module never asks.
from operatebench.execution_ledger import (
    SUPPORTED_EXECUTION_LEDGER_VERSIONS,
    execution_run_id_problem,
)
from operatebench.providers.cost import usd_text

# ``_reproduce_episode`` is private to the runner and imported here on purpose.
# Reproducing a record means executing under the identity that record carries,
# and the public run surface has no way to say that — deliberately, because it
# is provider-capable. This module is the reproduction path the private surface
# exists for; it reaches no provider for a model record, and the identity it
# passes came out of a validated artefact.
from operatebench.runner import EpisodeRun, _reproduce_episode
from operatebench.version import OPERATEBENCH_VERSION

#: The artefact contract version this build *writes*. A reader refuses a version
#: it cannot honour; see :data:`SUPPORTED_ARTIFACT_VERSIONS` for what it reads.
ARTIFACT_VERSION = 8

#: The first contract. Still read, and byte-frozen in the suite: v2 adds fields,
#: and adding fields is not a licence to stop honouring records that were
#: written before they existed. A v1 artefact validates against the v1 field set
#: *exactly* — it does not gain optional v2 fields by being read by a v2 build,
#: so a document that carries one is refused rather than half-read.
#:
#: Replay is a separate question, and the compatibility ends there rather than
#: here. A v1 record is replayed by re-executing the operation, engine and
#: evaluator it names; where those semantics have moved, the spec-digest binding
#: in :func:`replay_artifact` refuses it by name rather than comparing two
#: different operations. The suite holds both halves: the frozen v1 fixture for
#: what this build still reads, and a v1-shaped record derived from a current
#: run for what it still replays.
ARTIFACT_VERSION_V1 = 1

#: The second contract: v1 plus the decision tape, its digest and the agent
#: execution record. Still read, and still read under *its own* row shapes.
#:
#: That last part is what makes v3 a contract bump rather than a widening. v3
#: changes no top-level field; it changes the durable body of one trajectory row,
#: requiring every ``effect_accepted`` to carry the identities it established.
#: A reader that accepted a row with or without ``bindings`` at any version would
#: be a reader for which "this build always writes them" is unenforceable — the
#: forgery this contract exists to refuse would simply omit the field. So the row
#: shapes are selected by the document's own version: a v1 or v2 record is held
#: to the body those contracts were written under, and a v3 record must carry the
#: bindings exactly.
ARTIFACT_VERSION_V2 = 2

#: The third contract: v2 with ``bindings`` required on every accepted effect.
ARTIFACT_VERSION_V3 = 3

#: The fourth contract: v3 with the run's operation instance identity, and the
#: wake-reachability claim retracted from a declared wait. Still read, under its
#: own field set and its own row shapes, and no longer reproduced — see
#: :data:`REPRODUCIBLE_ARTIFACT_VERSIONS`.
ARTIFACT_VERSION_V4 = 4

#: The fifth contract: v4 with tool-mediated read evidence beside the normalized
#: projection an agent was still handed unasked. Still read, under its own field
#: set and its own row shapes, and no longer reproduced — see
#: :data:`REPRODUCIBLE_ARTIFACT_VERSIONS`.
ARTIFACT_VERSION_V5 = 5

#: The sixth contract: v5 with reading as the only way an authoritative fact
#: reaches an agent. Still read, under its own field set and its own row shapes,
#: and no longer reproduced — see :data:`REPRODUCIBLE_ARTIFACT_VERSIONS`.
#:
#: It is also the last contract that carries **no provider execution evidence**.
#: A v6 model record states which model produced its decisions and what each
#: request hashed to, and states nothing at all about what the run cost, how
#: many attempts it made or which journal witnessed it — because no journal was
#: bound to it. That is why v7 is a contract bump rather than an optional field:
#: a reader cannot tell "this run had no provider evidence" from "this run's
#: provider evidence was left out" unless the two are different contracts.
ARTIFACT_VERSION_V6 = 6

#: The seventh contract: v6 plus the provider-execution ledger binding. Frozen
#: as its own literal; artifact 8 changes evaluation meaning, not this identity.
ARTIFACT_VERSION_V7 = 7

#: Every contract version this build can read.
SUPPORTED_ARTIFACT_VERSIONS: tuple[int, ...] = (
    ARTIFACT_VERSION_V1,
    ARTIFACT_VERSION_V2,
    ARTIFACT_VERSION_V3,
    ARTIFACT_VERSION_V4,
    ARTIFACT_VERSION_V5,
    ARTIFACT_VERSION_V6,
    ARTIFACT_VERSION_V7,
    ARTIFACT_VERSION,
)

#: Every contract version this build can *reproduce*. Only the current one, and
#: that is a deliberate narrowing rather than an omission.
#:
#: Contract 4 moved two things a replay depends on. The model-visible
#: observation dropped the scenario identity and gained the operation instance
#: identity, so every ``observation_digest_sha256`` in a pre-v4 tape was taken
#: over a projection this build no longer produces and cannot reconstruct
#: without re-introducing the leak. And the ``wait_declared`` row dropped
#: ``reachable``, a claim about the authored future that this build has retracted
#: — a rerun cannot write it back, and writing it back is exactly the guess this
#: contract exists to stop making.
#:
#: Contract 6 moves the same two things again, and further. The model-visible
#: observation no longer carries the normalized projection of the operation
#: record at all, so every ``observation_digest_sha256`` in a v5 tape was taken
#: over a projection this build cannot produce without putting the answer set
#: back; a business proposal is now bound to the reads that established it, so
#: the ``agent_invoked`` row states the waking event's authority and the strings
#: it carried, which no v5 row has; and the result vector carries a dimension v5
#: did not grade.
#:
#: Contract 7 moves one more. A model record now carries the binding to the
#: execution ledger that witnessed it, and a replay compares that binding like
#: every other field. A v6 model record carries no binding at all, so replaying
#: one would compare a field the record does not have against a field the rerun
#: does — and a v6 *deterministic* record is not exempt either, because the
#: reconstruction a rerun produces is a v7 document and the two field sets are
#: exact on both sides.
#:
#: So a pre-v7 record is still *read*: it is what it is, it validates under its
#: own contract, and refusing to read it would destroy evidence about how the
#: benchmark stood. It is not replayed, because a replay is a comparison and
#: there is nothing honest to compare it against. See :func:`_reproduce`.
REPRODUCIBLE_ARTIFACT_VERSIONS: tuple[int, ...] = (ARTIFACT_VERSION,)


class RetrievalProvenanceMismatchError(ArtifactError):
    """A re-served read did not come back as the read the record describes.

    Its own class rather than a difference in the replay report. Every other
    section difference is a comparison of two documents; this is a broken
    *binding* — the record says a decision rested on a record at a version, and
    the live operation says that record is at another — and an operator has to be
    able to tell "the run diverged" from "the evidence does not hold".
    """


#: Said in every artefact, because a digest is easy to over-read.
ARTIFACT_NOTE = (
    "Content digests over canonical UTF-8 JSON. They establish that two records "
    "carry the same content; they are not tamper-evidence and make no "
    "cryptographic claim about who could have written this file."
)

#: Exactly the top-level fields a **v1** artefact carries: all of them required,
#: none of them optional, and nothing else tolerated. An unknown field is a claim
#: this build cannot honour, and honouring it silently is how a reader ends up
#: comparing two records that mean different things.
#:
#: Frozen. This tuple is the v1 contract, and a v2 field added to it would
#: retroactively invalidate every artefact written under v1.
_TOP_LEVEL_FIELDS_V1: tuple[str, ...] = (
    "artifact_version",
    "engine_version",
    "operation",
    "scenario_id",
    "semantic_scenario_id",
    "scenario_label",
    "agent_id",
    "agent_kind",
    "expected_terminal",
    "status",
    "terminal_outcome",
    "replay_final",
    "started_at",
    "ended_at",
    "simulated_minutes",
    "agent_invocations",
    "events",
    "trajectory",
    "final_state",
    "final_state_digest_sha256",
    "trajectory_digest_sha256",
    "evaluation",
    "note",
)

#: What v2 adds, and the whole reason it exists: the decisions the agent made,
#: each bound to the digest of the observation that produced it, and the record
#: of how those decisions were produced. Together they are what lets a
#: stochastic run be reproduced without asking the provider anything twice.
_NEW_IN_V2: tuple[str, ...] = (
    "decisions",
    "decisions_digest_sha256",
    "agent_execution",
)

_TOP_LEVEL_FIELDS_V2: tuple[str, ...] = _TOP_LEVEL_FIELDS_V1 + _NEW_IN_V2

#: v3 adds no top-level field. What it adds is inside the trajectory, and the
#: alias says so rather than leaving a reader to compare two tuples and conclude
#: the contract did not move.
_TOP_LEVEL_FIELDS_V3: tuple[str, ...] = _TOP_LEVEL_FIELDS_V2

#: What v4 adds: the identity of *this run*, as distinct from the identity of the
#: case it ran. It is what the agent was told the operation instance was called,
#: and it is here because the observation digests in ``decisions`` were taken
#: over a projection that carries it — without it in the record, a replay cannot
#: offer a recorded decision the observation it was produced from.
#:
#: The scenario fields stay exactly where they were. Separating what the agent
#: may see from what the record must state is the point: the artefact keeps
#: naming the case, the semantic scenario and the expected terminal, because a
#: grader reads the artefact.
_NEW_IN_V4: tuple[str, ...] = ("operation_instance_id",)

_TOP_LEVEL_FIELDS_V4: tuple[str, ...] = _TOP_LEVEL_FIELDS_V3 + _NEW_IN_V4

#: v5 adds no top-level field. What it adds is inside the trajectory — three
#: record types the runtime now writes — and on the tape, which learns a sixth
#: decision kind. The alias says so rather than leaving a reader to compare two
#: tuples and conclude the contract did not move.
#:
#: **v5 is a draft within an unmerged branch.** It carries tool-mediated read
#: evidence and it does *not* carry provider telemetry; the spike's own
#: sequencing says those should land in one bump, and this phase deliberately
#: ships the first half so the vertical can be executed and measured. Phase 1B
#: has to finalise execution evidence before any live provider run is attributed
#: to this contract. No v5 record produced outside this branch exists.
_TOP_LEVEL_FIELDS_V5: tuple[str, ...] = _TOP_LEVEL_FIELDS_V4

#: v6 adds no top-level field either. What it changes is what the fields *mean*:
#: the observation the decision digests were taken over no longer carries the
#: normalized record, the trajectory's invocation row states the waking
#: authority, and the evaluation carries a dimension about reading. A contract
#: bump for a change of meaning rather than of shape is the honest one — the
#: alternative is two documents with the same fields that cannot be compared.
_TOP_LEVEL_FIELDS_V6: tuple[str, ...] = _TOP_LEVEL_FIELDS_V5

#: What v7 adds, and the whole reason it exists: the binding between an episode
#: and the execution ledger that witnessed the provider session behind it.
#:
#: One field, always present, and ``null`` for every run that reached no
#: provider. Always present rather than optional, because "absent" and "null"
#: would then be two ways to write the same document and a reader could not
#: refuse either. And a *binding* rather than the evidence itself: the ledger is
#: a separate file with its own chain, and copying its rows in here would make
#: the artefact a second, drifting source of truth for facts the journal already
#: states — and would drag provider spend into a document every field of which
#: :func:`replay_artifact` compares.
_NEW_IN_V7: tuple[str, ...] = ("provider_execution",)

_TOP_LEVEL_FIELDS_V7: tuple[str, ...] = _TOP_LEVEL_FIELDS_V6 + _NEW_IN_V7

#: The field set each contract version is held to, exactly.
_FIELDS_BY_VERSION: Mapping[int, tuple[str, ...]] = {
    ARTIFACT_VERSION_V1: _TOP_LEVEL_FIELDS_V1,
    ARTIFACT_VERSION_V2: _TOP_LEVEL_FIELDS_V2,
    ARTIFACT_VERSION_V3: _TOP_LEVEL_FIELDS_V3,
    ARTIFACT_VERSION_V4: _TOP_LEVEL_FIELDS_V4,
    ARTIFACT_VERSION_V5: _TOP_LEVEL_FIELDS_V5,
    ARTIFACT_VERSION_V6: _TOP_LEVEL_FIELDS_V6,
    ARTIFACT_VERSION_V7: _TOP_LEVEL_FIELDS_V7,
    ARTIFACT_VERSION: _TOP_LEVEL_FIELDS_V7,
}

#: Exactly what a provider execution binding states, and nothing else.
#:
#: Every entry is either an identity, a digest, a count or an exact decimal
#: amount. There is no field for a provider's prose, a credential, a raw
#: response, a header, an endpoint with a query, or a path on the machine that
#: ran it — and there is deliberately nowhere one could be put. The ledger's
#: *location* is not here either: a binding names a run and a digest, and a
#: reader that has the sidecar can find it by digest rather than by a path that
#: says where somebody's disk was laid out.
PROVIDER_EXECUTION_FIELDS: tuple[str, ...] = (
    "execution_run_id",
    "execution_ledger_version",
    "execution_ledger_digest_sha256",
    "provider",
    "api",
    "model",
    "settings_digest_sha256",
    "pricing_digest_sha256",
    "provider_calls",
    "attempts",
    "measured_cost_usd",
    "forfeited_reservation_usd",
    "input_tokens",
    "output_tokens",
    "decision_call_index",
)

#: Said wherever the binding is reported, because a digest is easy to over-read.
#: The artefact's own validator can establish that a binding is *coherent* —
#: that its cardinality, its mapping and its request digests agree with the rest
#: of the document it sits in. It cannot establish that the ledger holds what
#: the binding says it holds, because the ledger is not in the document. That
#: needs :func:`~operatebench.execution_bundle.audit_execution_bundle`, and even
#: that establishes agreement between two local files rather than anything the
#: provider signed.
PROVIDER_EXECUTION_NOTE = (
    "A provider execution binding names the execution ledger that witnessed a "
    "run and states its digest. Reading it establishes that this artefact is "
    "internally coherent about that run; it does not establish what the ledger "
    "contains, which requires auditing the sidecar, and neither establishes "
    "provider attestation."
)

_DECISION_FIELDS: tuple[str, ...] = DECISION_FIELDS

_EXECUTION_FIELDS: tuple[str, ...] = (
    "outcome_source",
    "model",
    "protocol_version",
    "max_output_tokens",
    "transport_calls",
    "retry_count",
    "attempts",
    "excluded",
    "exclusion_code",
    "exclusion_detail",
)

_ATTEMPT_FIELDS: tuple[str, ...] = (
    "invocation_index",
    "turn_index",
    "request_digest_sha256",
    "outcome",
    "fault",
)

#: What an attempt can have ended as. Closed, because "some other string" is how
#: an execution record grows a category nobody defined.
_ATTEMPT_OUTCOMES = frozenset({"decided", "classified", "failed"})

#: Which of those an *artefact* can carry, and the decision kind each one sits
#: beside. ``failed`` is absent on purpose: a call that failed at the provider
#: produced no decision and excluded the run, and an excluded run has no episode
#: artefact in this build — see :data:`_NO_EXCLUDED_ARTEFACT`.
#: A retrieval turn is ``decided``: it is a real call the provider answered with
#: a well-formed structured request, and it cost exactly what a business turn
#: costs. Recording it as anything else would make a run's provider cost
#: unreadable from its own attempts.
_ATTEMPT_OUTCOME_FOR_KIND: Mapping[str, str] = {
    **dict.fromkeys(OUTCOME_KINDS, "decided"),
    RETRIEVE_KIND: "decided",
    MALFORMED_KIND: "classified",
}

#: The kinds a recorded decision may be. Core's five, plus the one the tape
#: needs in order to be able to say "the agent produced something that was not a
#: decision" — a run whose model misbehaved has to be replayable too.
_DECISION_KINDS = frozenset({*OUTCOME_KINDS, MALFORMED_KIND, RETRIEVE_KIND})

#: The closed vocabulary a recorded ``MALFORMED`` decision's code is held to.
#: Taken from the model boundary rather than rebuilt here: it is every code this
#: build can *write*, which includes the base ``MODEL_OUTPUT_UNCLASSIFIED`` that
#: a response neither parsed nor classified is recorded as. Deriving it from the
#: six parsed classifications alone left this reader refusing a tape this build
#: had written itself.
_CLASSIFICATION_CODES = READABLE_CLASSIFICATIONS

#: The exact body of every recorded decision kind, keyed by kind. Written from
#: the same projections the tape is produced by —
#: :func:`~operatebench.core.outcomes.outcome_as_dict` and
#: :func:`~operatebench.agents.playback.decision_record` — because a reader that
#: checked only ``kind`` would accept an ``ACT`` carrying a provider's prose, an
#: invented argument or an ``evidence_refs`` that is a mapping, and playback
#: would then rebuild a decision out of fields nobody wrote.
#:
#: ``kind`` itself is implicit: it selects the shape, and the shape is exact, so
#: a field this build does not write is a refusal rather than a value carried
#: forward into durable evidence.
_OUTCOME_SHAPES: Mapping[str, Mapping[str, str]] = {
    "ACT": {
        "action_type": "text",
        "payload": "object",
        "evidence_refs": "text_list",
        "rationale": "string",
    },
    "WAIT": {
        "reason": "string",
        "wake_on": "text_list",
        "fallback_after_minutes": "optional_minutes",
    },
    "ASK": {
        "recipient_actor_id": "text",
        "message_fixture_id": "text",
        "correlation_id": "optional_text",
        "wait": "wait",
    },
    "ESCALATE": {
        "checkpoint_id": "text",
        "exception_type": "text",
        "evidence_refs": "text_list",
        "deadline_after_minutes": "minutes",
        "rationale": "string",
    },
    "COMPLETE": {"reason": "string", "evidence_refs": "text_list"},
    RETRIEVE_KIND: {"requests": "retrieval_requests"},
    MALFORMED_KIND: {"code": "classification"},
}

#: The agent kind an execution source implies, for the two sources no registry
#: can speak for. A registered in-process agent's kind comes from the registry;
#: a provider-backed or replayed one has no registry entry, so the source is the
#: only thing that can name it, and it names it exactly.
KIND_MODEL = "model"
KIND_RECORDED = "recorded"

#: The kinds the shipped agent registry declares. Derived, not restated: an
#: agent added under a new kind must be a kind an artefact can carry, and a
#: hand-written copy of this set would be the place the two silently diverge.
DETERMINISTIC_AGENT_KINDS: frozenset[str] = frozenset(
    entry.kind for entry in AGENTS.values()
)

#: The model protocol versions this build can *read* a recorded model run under.
#:
#: Two, and the older one is here for the same reason a v4 artefact is still
#: read: the record is what it is, and refusing to read it would destroy evidence
#: about how the benchmark stood. Reading is not reproducing. A ``v1`` record is
#: readable and — because it can only appear inside a contract this build no
#: longer reproduces — is never replayed, so nothing rebuilds a request under a
#: protocol this build does not speak.
SUPPORTED_MODEL_PROTOCOLS: tuple[str, ...] = (
    MODEL_PROTOCOL_VERSION_V1,
    MODEL_PROTOCOL_VERSION_V2,
    MODEL_PROTOCOL_VERSION_V3,
    MODEL_PROTOCOL_VERSION,
)

#: The protocol a *reproducible* record must name. A contract-5 record whose
#: decisions came off a provider was produced under this build's protocol, and a
#: record that says otherwise describes requests this build cannot rebuild.
REPRODUCIBLE_MODEL_PROTOCOLS: tuple[str, ...] = (MODEL_PROTOCOL_VERSION,)

#: Why an excluded execution is refused rather than read. Stated once, so the
#: refusal an operator meets and the contract this module documents are the same
#: sentence.
_NO_EXCLUDED_ARTEFACT = (
    "says the execution was excluded at the provider boundary. A run that "
    "stopped there produced no business outcome, so it has no terminal, no "
    "evaluation and no episode artefact in this build: the failure propagates "
    "out of the runner before an artefact exists. Recording one needs an "
    "execution ledger whose result is the attempts and the fault rather than a "
    "terminal, and until that exists this reader refuses the record instead of "
    "accepting a scored, reliable, completed episode beside a provider fault"
)

_OPERATION_FIELDS: tuple[str, ...] = (
    "operation_id",
    "operation_type",
    "operation_version",
    "semantic_scenario_id",
    "privacy_status",
    "spec_digest_sha256",
    "scenario_ids",
)

_EVENT_FIELDS: tuple[str, ...] = (
    "event_id",
    "event_type",
    "actor_id",
    "at",
    "sequence",
    "payload",
    "triggers_agent",
    "caused_by",
    "disposition",
    "verdict_code",
)

_EVALUATION_FIELDS: tuple[str, ...] = (
    "terminal_outcome",
    "legitimate_completion",
    "reliable",
    "failed_dimensions",
    "finding_codes",
    "dimensions",
)

_DIMENSION_FIELDS: tuple[str, ...] = ("name", "ok", "findings", "counts", "note")

_FINDING_FIELDS: tuple[str, ...] = ("code", "detail", "at")

#: The result vector contract 1 was written under, in report order. Frozen, and
#: restated rather than derived: the frozen v1 fixture carries exactly these nine
#: names, and a tuple computed by removing later additions from the current
#: vector would silently follow the current vector wherever it went.
_DIMENSIONS_V1: tuple[str, ...] = (
    "terminal_outcome",
    "critical_invariants",
    "temporal_correctness",
    "authority_boundaries",
    "action_validity",
    "human_checkpoints",
    "recovery",
    "obligations",
    "deterministic_replay",
)

#: The result vector contracts 2 to 5 were written under: contract 1's, plus the
#: dimension oracle 0.2.0 added. Contracts 3, 4 and 5 moved other things and left
#: the vector alone, so they share this tuple rather than each restating it.
_DIMENSIONS_V2: tuple[str, ...] = (*_DIMENSIONS_V1, "environment_integrity")

#: Which vector each contract version is held to, exactly. Version-aware for the
#: same reason the row shapes are: ``retrieval_discipline`` is graded by this
#: build and was graded by no build that wrote a v5 record, so a reader that
#: accepted the vector with or without it at every version would let a forged v5
#: document claim a dimension its vintage never measured — and would refuse the
#: frozen v1 fixture for not carrying two dimensions that did not exist.
#:
#: The current entry is read from the live evaluator rather than restated, so a
#: dimension added there and described nowhere is caught by
#: :func:`evaluation_dimension_coverage_problem` instead of being frozen out.
_DIMENSIONS_BY_VERSION: Mapping[int, tuple[str, ...]] = {
    ARTIFACT_VERSION_V1: _DIMENSIONS_V1,
    ARTIFACT_VERSION_V2: _DIMENSIONS_V2,
    ARTIFACT_VERSION_V3: _DIMENSIONS_V2,
    ARTIFACT_VERSION_V4: _DIMENSIONS_V2,
    ARTIFACT_VERSION_V5: _DIMENSIONS_V2,
    # v7 grades what v6 graded: the binding it adds is evidence about the
    # provider session, and no dimension scores a provider.
    ARTIFACT_VERSION_V6: DIMENSIONS,
    ARTIFACT_VERSION_V7: DIMENSIONS,
    ARTIFACT_VERSION: DIMENSIONS,
}


def evaluation_dimensions_for(version: int) -> tuple[str, ...]:
    """Exactly the dimensions contract ``version`` graded, in report order."""
    names = _DIMENSIONS_BY_VERSION.get(version)
    _require(
        names is not None,
        "artefact evaluation",
        f"contract {version!r} has no result-vector contract in this build; "
        f"described versions are {sorted(_DIMENSIONS_BY_VERSION)}",
    )
    assert names is not None
    return names


def evaluation_dimension_coverage_problem() -> str | None:
    """Why the per-version result vectors do not describe this build, or ``None``.

    Two properties, both of which a hand-maintained table loses quietly. Every
    version this build reads must have a vector, or an artefact would be held to
    no vector at all; and the current version's vector must *be* the evaluator's,
    or the reader and the writer would disagree about what a run produces.

    The third is the one that keeps the history honest: each contract's vector
    must be a prefix of the next. The vector has only ever gained a dimension at
    the end, and a table that reordered or dropped one would be describing a
    contract no build ever wrote.
    """
    missing = [
        version
        for version in SUPPORTED_ARTIFACT_VERSIONS
        if version not in _DIMENSIONS_BY_VERSION
    ]
    if missing:
        return f"contract version(s) {missing} have no result-vector contract"
    current = _DIMENSIONS_BY_VERSION[ARTIFACT_VERSION]
    if current != DIMENSIONS:
        return (
            f"the current contract grades {list(current)} and the evaluator "
            f"produces {list(DIMENSIONS)}"
        )
    ordered = sorted(SUPPORTED_ARTIFACT_VERSIONS)
    for earlier, later in pairwise(ordered):
        before = _DIMENSIONS_BY_VERSION[earlier]
        after = _DIMENSIONS_BY_VERSION[later]
        if after[: len(before)] != before:
            return (
                f"contract {later} does not extend contract {earlier}: "
                f"{list(before)} is not a prefix of {list(after)}"
            )
    return None


#: Every row is indexed, stamped and typed. What it carries beyond that is fixed
#: by its type, and stated below.
_TRAJECTORY_FIELDS: tuple[str, ...] = ("index", "at", "record_type")

#: The body an ``effect_accepted`` row carried under contracts 1 and 2. Frozen:
#: it is what those contracts were written under, and a record produced before
#: ``bindings`` existed does not acquire the field by being read by a v3 build.
_EFFECT_ACCEPTED_BEFORE_V3: Mapping[str, str] = {
    "proposal_id": "text",
    "action_type": "text",
    "code": "text",
    "cycle_id": "optional_text",
}

#: The body contract 3 writes: the same four fields, plus the canonical
#: identities the effect established, as a string→string mapping.
#:
#: Required, not optional, and that is the whole point of the version bump. The
#: evaluator re-derives which invoice a validation was about from this mapping
#: rather than from a final state assembled afterwards; a shape that accepted a
#: row with or without it would let a forgery drop the field and be graded on
#: hindsight again. So the two shapes live under two contract versions and a
#: document is held to its own.
_EFFECT_ACCEPTED_V3: Mapping[str, str] = {
    **_EFFECT_ACCEPTED_BEFORE_V3,
    "bindings": "bindings",
}

#: The exact body of every trajectory record type, keyed by type and given as the
#: tuple of shapes that type is written in. A row must match one shape exactly:
#: every field of it, with the right primitive kind, and nothing else.
#:
#: This table is the contract *this build writes* — see
#: :data:`_ROW_SHAPES_BY_VERSION` for the one an older document is held to.
#:
#: Checking only ``index``/``at``/``record_type`` was enough to let a row claim
#: to be an ``effect_accepted`` while carrying an action type that is a list, or
#: an ``action_proposed`` with the proposal identity removed — and the evaluator
#: reads exactly those fields. A record type with more than one shape has more
#: than one entry rather than a pile of optional fields, so "a wait that fired
#: its fallback" and "a wait that declared a wake set" cannot be confused for
#: each other by omission.
_ROW_SHAPES: Mapping[str, tuple[Mapping[str, str], ...]] = {
    "event_observed": (
        {
            "event_id": "text",
            "event_type": "text",
            "actor_id": "text",
            "code": "text",
            "observational": "flag",
        },
    ),
    "event_audit_only": (
        {
            "event_id": "text",
            "event_type": "text",
            "actor_id": "text",
            "code": "text",
            "observational": "flag",
        },
    ),
    "event_rejected": (
        {
            "event_id": "text",
            "event_type": "text",
            "actor_id": "text",
            "code": "text",
            "detail": "string",
        },
    ),
    "event_after_terminal": (
        {
            "event_id": "text",
            "event_type": "text",
            "actor_id": "text",
            "mutating": "flag",
        },
    ),
    # ``trigger_event_authority`` and ``trigger_claim_values`` arrive at contract
    # 6. They are what lets a reader ask, from the record alone, whether a
    # proposal cited what somebody *told* the agent rather than what the record
    # holds — the question ``retrieval_discipline`` answers, and it cannot be
    # answered by a document that only says which event type woke the agent.
    "agent_invoked": (
        {
            "invocation_index": "counter",
            "trigger_event_id": "optional_text",
            "trigger_event_type": "optional_text",
            "trigger_event_authority": "optional_text",
            "trigger_is_authoritative": "flag",
            "trigger_claim_values": "text_list",
        },
    ),
    # ``outcome`` here is the same projection the decision tape carries — the
    # engine writes both from ``outcome_as_dict`` — so it is held to the same
    # exact bodies. Checking it as a bare object accepted an ACT carrying the
    # prose that arrived beside a tool call, an ASK whose nested wait nothing
    # could honour and a MALFORMED naming a code this build never classifies,
    # and the evaluator reads these rows.
    "agent_outcome": (
        {"invocation_index": "counter", "turn_index": "counter", "outcome": "outcome"},
    ),
    "outcome_rejected": (
        {
            "invocation_index": "counter",
            "turn_index": "counter",
            "code": "text",
            "detail": "string",
        },
    ),
    "wait_declared": (
        {"fallback_fired": "flag"},
        {
            "reason": "string",
            "wake_on": "text_list",
            "fallback_at": "optional_instant",
        },
    ),
    "wait_unresolved_at_horizon": (
        {"reason": "string", "wake_on": "text_list", "horizon_at": "instant"},
    ),
    "wait_rejected": (
        {"code": "text", "detail": "string"},
        {"code": "text", "detail": "string", "event_id": "text"},
    ),
    "action_proposed": (
        {
            "proposal_id": "text",
            "action_type": "text",
            "payload": "object",
            "evidence_refs": "text_list",
        },
    ),
    "action_rejected": (
        {
            "proposal_id": "text",
            "action_type": "text",
            "code": "text",
            "detail": "string",
        },
    ),
    "effect_accepted": (_EFFECT_ACCEPTED_V3,),
    "side_effect": (
        {
            "channel": "text",
            "message_fixture_id": "text",
            "recipient_actor_id": "text",
            "status": "text",
        },
    ),
    "side_effect_failed": (
        {
            "channel": "text",
            "message_fixture_id": "text",
            "recipient_actor_id": "text",
            "committed_decision_preserved": "flag",
        },
    ),
    "checkpoint_opened": (
        {
            "checkpoint_id": "text",
            "checkpoint_type": "text",
            "assignee_actor_id": "text",
            "required": "flag",
            "deadline_at": "instant",
        },
        {
            "checkpoint_id": "text",
            "checkpoint_type": "text",
            "assignee_actor_id": "text",
            "required": "flag",
            "exception_type": "text",
            "evidence_refs": "text_list",
            "deadline_at": "instant",
        },
    ),
    "checkpoint_resolved": (
        {"checkpoint_id": "text", "checkpoint_type": "text", "resolution": "text"},
    ),
    "checkpoint_rejected": (
        {"checkpoint_id": "text", "code": "text", "detail": "string"},
    ),
    "obligation_created": (
        {"obligation_id": "text", "kind": "text", "due_at": "instant"},
    ),
    "obligation_discharged": ({"obligation_id": "text", "kind": "text"},),
    "obligation_cancelled": ({"obligation_id": "text", "kind": "text"},),
    "obligation_breached": ({"obligation_id": "text", "kind": "text"},),
    "timer_scheduled": ({"event_id": "text", "event_type": "text", "due_at": "instant"},),
    "timer_cancelled": ({"event_id": "text"},),
    "terminal_proposed": ({"reason": "string", "evidence_refs": "text_list"},),
    "terminal_rejected": ({"code": "text", "detail": "string"},),
    "provisional_close_entered": (
        {"cycle_id": "text", "reopen_until": "instant", "replay_final": "flag"},
    ),
    "operation_reopened": (
        {
            "reopened_cycle_id": "text",
            "warranty_cycle_id": "text",
            "reason": "string",
        },
    ),
    "terminal_accepted": (
        {"outcome": "text", "replay_final": "flag", "cycle_id": "optional_text"},
        {"outcome": "text", "replay_final": "flag", "operator_actor_id": "text"},
    ),
    "episode_ended": (
        {"phase": "text"},
        {"phase": "text", "status": "text", "invocations": "counter"},
    ),
    "critical_violation": ({"code": "text", "detail": "string"},),
    # One row per served result, not one per batch. A batch is an atom for
    # *serving* — one instant, one canonical order — and evidence about what a
    # decision rested on is per record: which record, at which version, from
    # which source, under whose authority. ``batch_index`` and ``request_index``
    # are what put the atom back together without collapsing the rows.
    "retrieval_served": (
        {
            "invocation_index": "counter",
            "turn_index": "counter",
            "batch_index": "counter",
            "request_index": "counter",
            "initiated_by": "retrieval_initiator",
            "tool": "text",
            "source": "text",
            "authority": "retrieval_authority",
            "record_id": "text",
            "record_version": "text",
            "as_of": "instant",
            "schema_id": "text",
            # What came back, not only that something did. Without the body,
            # ``record_version`` is a number only its own author can check: a
            # reader can compare it against a re-served read, which is the
            # reader's own machinery, and nothing else in the document
            # constrains it. Carrying the records makes the version a claim the
            # row can be held to by anyone holding the row.
            "records": "retrieval_records",
            "ok": "flag",
        },
    ),
    # A batch that was refused before anything was served. ``requested_tools``
    # holds only catalogue members: a tool name an agent invented is
    # provider-controlled text, and the count says how many there were rather
    # than quoting one into durable evidence.
    "retrieval_refused": (
        {
            "invocation_index": "counter",
            "turn_index": "counter",
            "request_count": "counter",
            "requested_tools": "text_list",
            "code": "retrieval_refusal",
            "detail": "string",
        },
    ),
    "derived_context_invalidated": (
        {
            "invocation_index": "counter",
            "turn_index": "counter",
            "cause": "invalidation_cause",
            "action_type": "optional_text",
            "cleared_tools": "text_list",
        },
    ),
}

#: The row shapes contract 5 was written under: this build's, with the
#: invocation row put back the way that contract wrote it. A v5 build did not
#: know the waking event's authority class and did not record the strings it
#: carried, so describing them for that contract would state a body no artefact
#: of that vintage could have — and would let a forged v5 record claim the
#: evidence contract 6 exists to carry.
_ROW_SHAPES_BEFORE_V6: Mapping[str, tuple[Mapping[str, str], ...]] = {
    **_ROW_SHAPES,
    "agent_invoked": (
        {
            "invocation_index": "counter",
            "trigger_event_id": "optional_text",
            "trigger_event_type": "optional_text",
        },
    ),
}

#: The row shapes contracts 1 to 4 were written under: contract 5's, without the
#: three record types the runtime learned to write at contract 5. A pre-v5
#: document cannot carry a retrieval row — the build that wrote it performed no
#: retrieval — so describing one for those contracts would state a body no
#: artefact of that vintage could have, and would let a forged v4 record claim
#: reads that never happened.
_ROW_SHAPES_BEFORE_V5: Mapping[str, tuple[Mapping[str, str], ...]] = {
    name: shapes
    for name, shapes in _ROW_SHAPES_BEFORE_V6.items()
    if name not in RETRIEVAL_RECORD_TYPES
}

#: The row shapes contracts 1 to 3 were written under: this build's, with the
#: wait declaration put back the way those contracts wrote it. ``reachable`` was
#: a claim about the authored future — whether anything pending could deliver the
#: wake condition — and contract 4 retracts it. A record written under an earlier
#: contract still carries it, and is still held to carrying it: a reader that
#: accepted the row with or without the field would enforce neither contract.
_ROW_SHAPES_BEFORE_V4: Mapping[str, tuple[Mapping[str, str], ...]] = {
    **_ROW_SHAPES_BEFORE_V5,
    "wait_declared": (
        {"fallback_fired": "flag"},
        {
            "reason": "string",
            "wake_on": "text_list",
            "fallback_at": "optional_instant",
            "reachable": "flag",
        },
    ),
}

#: The row shapes contracts 1 and 2 were written under: the above, with the one
#: row whose body v3 changed put back the way those contracts wrote it. Derived
#: rather than copied, so a row type added to the table above is described for
#: every version rather than silently undescribed for the older ones.
_ROW_SHAPES_BEFORE_V3: Mapping[str, tuple[Mapping[str, str], ...]] = {
    **_ROW_SHAPES_BEFORE_V4,
    "effect_accepted": (_EFFECT_ACCEPTED_BEFORE_V3,),
}

#: Which row-shape table each contract version is held to. Version-aware on
#: purpose: a reader that accepted either body at any version would enforce
#: neither, and "this build always writes bindings" would stop being a checkable
#: claim about a durable record.
_ROW_SHAPES_BY_VERSION: Mapping[int, Mapping[str, tuple[Mapping[str, str], ...]]] = {
    ARTIFACT_VERSION_V1: _ROW_SHAPES_BEFORE_V3,
    ARTIFACT_VERSION_V2: _ROW_SHAPES_BEFORE_V3,
    ARTIFACT_VERSION_V3: _ROW_SHAPES_BEFORE_V4,
    ARTIFACT_VERSION_V4: _ROW_SHAPES_BEFORE_V5,
    ARTIFACT_VERSION_V5: _ROW_SHAPES_BEFORE_V6,
    # v7 changes no trajectory row: what it adds is one top-level field, so it
    # shares v6's table rather than restating it.
    ARTIFACT_VERSION_V6: _ROW_SHAPES,
    ARTIFACT_VERSION_V7: _ROW_SHAPES,
    ARTIFACT_VERSION: _ROW_SHAPES,
}

#: Which record types each contract version's runtime wrote. The coverage check
#: below compares a version's row-shape table against *this*, not against the
#: current vocabulary: a type this build writes and an older contract did not is
#: correctly absent from that contract's table, and demanding it there would make
#: every contract bump either retroactive or unenforceable.
_RECORD_TYPES_BEFORE_V5: tuple[str, ...] = tuple(
    name for name in RECORD_TYPES if name not in RETRIEVAL_RECORD_TYPES
)


def _record_types_for(version: int) -> tuple[str, ...]:
    """Which record types the runtime of contract ``version`` wrote.

    The current contract's answer is read from the live vocabulary rather than
    from a snapshot, so a record type added to the ledger and described nowhere
    is caught by the coverage check instead of being frozen out of it.
    """
    if version >= ARTIFACT_VERSION_V5:
        return RECORD_TYPES
    return _RECORD_TYPES_BEFORE_V5


_DISPOSITIONS = frozenset({"accepted", "rejected", "audit", "post_terminal"})

#: Which operation type's canonical state validator to hand ``final_state`` to.
#:
#: The dependency direction is the reason this is a table rather than a chain of
#: ``isinstance`` checks: Core knows nothing about any domain, and *this* module
#: is the root reader, which is allowed to know which domains exist. What it is
#: not allowed to do is guess. An artefact naming an operation type with no
#: registered validator is refused, because the alternative is validating the
#: outer mapping, shrugging at everything inside it and calling that a check.
_STATE_VALIDATORS: Mapping[str, Callable[[Any, str], None]] = {
    "lettings.maintenance.synthetic": validate_canonical_state,
}

_HEX = frozenset("0123456789abcdef")

#: How the artefact's fields group into the sections a replay reports on. Every
#: top-level field is in exactly one section, which is what makes "the replay
#: compared everything" a checkable statement rather than a claim.
#:
#: The invariant is enforced, not documented: :func:`section_coverage_problem`
#: recomputes it from the field set for every supported version, and the suite
#: asserts it is clean. A field added to the artefact and not to a section would
#: otherwise be a field replay silently agrees about by never looking at it.
_SECTIONS_V1: Mapping[str, tuple[str, ...]] = {
    "identity": (
        "artifact_version",
        "engine_version",
        "operation",
        "scenario_id",
        "semantic_scenario_id",
        "scenario_label",
        "agent_id",
        "agent_kind",
        "expected_terminal",
    ),
    "metadata": (
        "status",
        "terminal_outcome",
        "replay_final",
        "started_at",
        "ended_at",
        "simulated_minutes",
        "agent_invocations",
        "note",
    ),
    "events": ("events",),
    "trajectory": ("trajectory", "trajectory_digest_sha256"),
    "final_state": ("final_state", "final_state_digest_sha256"),
    "evaluation": ("evaluation",),
}

#: v2's sections: v1's, plus one section per thing v2 added. Two new sections
#: rather than three new fields hidden inside an existing one, because a
#: divergence in *what the agent decided* and a divergence in *how it was
#: reached* are different findings and an operator has to be able to tell them
#: apart from the report alone.
_SECTIONS_V2: Mapping[str, tuple[str, ...]] = {
    **_SECTIONS_V1,
    "decisions": ("decisions", "decisions_digest_sha256"),
    "execution": ("agent_execution",),
}

#: v3's sections are v2's: the contract moved inside the trajectory, which is
#: already one section, so nothing new is compared and nothing stops being.
_SECTIONS_V3: Mapping[str, tuple[str, ...]] = _SECTIONS_V2

#: v4's sections: v3's, with the run's instance identity in ``identity``, where
#: the rest of what this run *was* already sits.
_SECTIONS_V4: Mapping[str, tuple[str, ...]] = {
    **_SECTIONS_V3,
    "identity": (*_SECTIONS_V3["identity"], "operation_instance_id"),
}

#: v5's sections are v4's: the contract moved inside the trajectory and the
#: decision tape, both of which are already sections, so nothing new is compared
#: and nothing stops being.
_SECTIONS_V5: Mapping[str, tuple[str, ...]] = _SECTIONS_V4

#: v6's sections are v5's, for the same reason: what moved sits inside the
#: trajectory, the decision tape and the evaluation, all of which are already
#: sections.
_SECTIONS_V6: Mapping[str, tuple[str, ...]] = _SECTIONS_V5

#: v7's sections: v6's, plus one of its own for the binding. Its own rather than
#: folded into ``execution``, because "this run reached a different provider
#: session" and "this run's decisions were produced differently" are different
#: findings, and a report an operator reads has to keep them apart.
_SECTIONS_V7: Mapping[str, tuple[str, ...]] = {
    **_SECTIONS_V6,
    "provider_execution": ("provider_execution",),
}

_SECTIONS_BY_VERSION: Mapping[int, Mapping[str, tuple[str, ...]]] = {
    ARTIFACT_VERSION_V1: _SECTIONS_V1,
    ARTIFACT_VERSION_V2: _SECTIONS_V2,
    ARTIFACT_VERSION_V3: _SECTIONS_V3,
    ARTIFACT_VERSION_V4: _SECTIONS_V4,
    ARTIFACT_VERSION_V5: _SECTIONS_V5,
    ARTIFACT_VERSION_V6: _SECTIONS_V6,
    ARTIFACT_VERSION_V7: _SECTIONS_V7,
    ARTIFACT_VERSION: _SECTIONS_V7,
}


def row_shape_coverage_problem() -> str | None:
    """Why the row-shape tables and :data:`RECORD_TYPES` disagree, or ``None``.

    A record type the ledger writes and a table does not describe would reach
    ``shapes[record_type]`` as a raw ``KeyError`` — a traceback out of a reader
    whose whole contract is that untrusted input produces named refusals. A
    shape for a type the ledger no longer writes is the opposite drift: a body
    this build claims to accept and can never meet. Both are computed rather
    than asserted in a comment, and the suite runs this.

    Every supported version's table is checked, not just the current one. A row
    type described for v3 and missing from the table an older document is read
    under would be a ``KeyError`` reachable from a v1 file, which is the same
    defect one contract version away.
    """
    for version, shapes in sorted(_ROW_SHAPES_BY_VERSION.items()):
        written = set(_record_types_for(version))
        undescribed = sorted(written - set(shapes))
        if undescribed:
            return (
                f"v{version}: trajectory record type(s) {undescribed} are written by "
                "the ledger and have no row shape here, so a row of that type would "
                "reach the reader as a KeyError rather than a named refusal"
            )
        unwritten = sorted(set(shapes) - written)
        if unwritten:
            return (
                f"v{version}: row shape(s) {unwritten} describe record type(s) this "
                "build's ledger does not write, so they state a body no artefact can "
                "carry"
            )
    missing = sorted(set(SUPPORTED_ARTIFACT_VERSIONS) - set(_ROW_SHAPES_BY_VERSION))
    if missing:
        return (
            f"contract version(s) {missing} are read by this build and have no row "
            "shape table, so a document at that version could not be held to any body"
        )
    return None


def outcome_shape_coverage_problem() -> str | None:
    """Why :data:`_OUTCOME_SHAPES` and :data:`_DECISION_KINDS` disagree, or ``None``.

    The same drift check one level down. A decision kind with no shape would be
    admitted on its ``kind`` alone, which is exactly the hole that let a
    recorded outcome carry provider prose beside a well-known kind.
    """
    undescribed = sorted(_DECISION_KINDS - set(_OUTCOME_SHAPES))
    if undescribed:
        return (
            f"decision kind(s) {undescribed} are recordable and have no outcome shape "
            "here, so a decision of that kind would be admitted on its kind alone"
        )
    unwritten = sorted(set(_OUTCOME_SHAPES) - _DECISION_KINDS)
    if unwritten:
        return (
            f"outcome shape(s) {unwritten} describe decision kind(s) this build does "
            "not record"
        )
    return None


def section_coverage_problem(version: int) -> str | None:
    """Why this version's sections do not partition its fields, or ``None``.

    The check the suite runs, living beside the tables it checks. A field in no
    section is a field replay never compares; a field in two is a difference
    reported twice under two names. Both are silent, so both are computed.
    """
    fields = _FIELDS_BY_VERSION.get(version)
    sections = _SECTIONS_BY_VERSION.get(version)
    if fields is None or sections is None:
        return f"artefact version {version!r} has no field set or no section table"
    seen: dict[str, list[str]] = {}
    for section, names in sections.items():
        for name in names:
            seen.setdefault(name, []).append(section)
    duplicated = sorted(name for name, places in seen.items() if len(places) > 1)
    if duplicated:
        return (
            f"v{version}: field(s) {duplicated} appear in more than one replay "
            "comparison section, so a difference in them would be reported twice "
            "under two names"
        )
    uncovered = sorted(set(fields) - set(seen))
    if uncovered:
        return (
            f"v{version}: field(s) {uncovered} belong to no replay comparison "
            "section, so a replay would agree about them by never looking"
        )
    unknown = sorted(set(seen) - set(fields))
    if unknown:
        return (
            f"v{version}: section(s) name field(s) {unknown} that this version's "
            "artefact does not carry"
        )
    return None


def project_trajectory(
    rows: Sequence[Mapping[str, Any]], version: int
) -> list[dict[str, Any]]:
    """The same trajectory, written the way contract ``version`` wrote one.

    Only ever *drops* row fields a later contract added; it never invents one
    and never rewrites a value. That direction is the whole safety property:
    projecting a current run back onto an older contract can lose evidence the
    older contract had no place for, and cannot manufacture evidence the run
    did not produce.

    Needed in two places, which is why it is here rather than in either of them.
    Replay reconstructs the artefact a rerun would have written and compares it
    with the recorded one; when the record is a v1 or v2 document, the
    reconstruction has to be the one *that* contract would have written, or
    every ``effect_accepted`` row would be reported as a difference an older
    artefact had no way to avoid. And the suite derives a current-contract run
    into an older-shaped document to keep the older reader's evidence
    executable.

    The digest is the caller's problem, deliberately: a projection changes
    content, so the content digest changes with it, and recomputing it silently
    here would hide that from a caller who did not intend a projection at all.

    It drops only fields it *owns* — the ones a named contract bump added to a
    named row — and never fields it merely fails to recognise. That is not
    tidiness. A projection that stripped anything unfamiliar would launder a
    forged row into a document the older reader accepts, and the reader is where
    an unknown field has to be refused, because only the reader knows which
    contract the document claims to be.
    """
    dropped_at_v3 = sorted(set(_EFFECT_ACCEPTED_V3) - set(_EFFECT_ACCEPTED_BEFORE_V3))
    dropped_at_v6 = sorted(
        set(_ROW_SHAPES["agent_invoked"][0])
        - set(_ROW_SHAPES_BEFORE_V6["agent_invoked"][0])
    )
    projected: list[dict[str, Any]] = []
    for row in rows:
        body = dict(row)
        record_type = body.get("record_type")
        if version < ARTIFACT_VERSION_V3 and record_type == "effect_accepted":
            for name in dropped_at_v3:
                body.pop(name, None)
        # Contract 6 bound a business proposal to the reads that established it,
        # so the invocation row states the waking event's authority class and the
        # strings it carried. No build that wrote a pre-v6 record knew either.
        if version < ARTIFACT_VERSION_V6 and record_type == "agent_invoked":
            for name in dropped_at_v6:
                body.pop(name, None)
        projected.append(body)
    return projected


def project_top_level(payload: Mapping[str, Any], version: int) -> dict[str, Any]:
    """The same document, written the way contract ``version`` wrote its top level.

    The top-level counterpart of :func:`project_trajectory`, and it obeys the
    same direction: it only ever *drops* fields a later contract added, never
    invents one and never rewrites a value. Projecting a v7 record onto contract
    6 removes ``provider_execution`` and nothing else — the binding is the only
    thing v7 added — so what is left is a document a v6 reader can hold to its
    own exact field set.

    It does not rewrite ``artifact_version``, deliberately. A projection is a
    *view*; relabelling the document as a v6 record is a claim about what wrote
    it, and the caller who wants to make that claim should have to make it.
    """
    fields = _FIELDS_BY_VERSION.get(version)
    _require(
        fields is not None,
        "artefact projection",
        f"contract {version!r} has no field set in this build; described versions "
        f"are {sorted(_FIELDS_BY_VERSION)}",
    )
    assert fields is not None
    source_version = payload.get("artifact_version")
    _require(
        isinstance(source_version, int) and not isinstance(source_version, bool),
        "artefact projection",
        "the document being projected states no contract version",
    )
    assert isinstance(source_version, int)
    _require(
        version <= source_version,
        "artefact projection",
        f"a contract-{source_version} document cannot be projected onto contract "
        f"{version}: a projection drops fields a later contract added and never "
        "invents ones the record does not carry",
    )
    return {name: value for name, value in payload.items() if name in set(fields)}


def content_digest(payload: Any) -> str:
    """The digest an artefact section is recorded under. One implementation."""
    return _content_digest(payload)


#: Why a model run with no complete scored ledger has no episode artefact.
#: Stated once, so the refusal an operator meets at build time and the one a
#: reader meets are the same sentence.
_NO_UNWITNESSED_MODEL_ARTEFACT = (
    "says its decisions came off a provider and carries no provider_execution "
    "binding. Contract 7 requires one for a model run: the binding names the "
    "execution ledger that witnessed the session and states its digest, and a "
    "model artefact without one would assert a provider session with nothing "
    "durable behind it. A run whose ledger is missing, incomplete or not scored "
    "writes no episode artefact rather than one with a null binding"
)


def build_artifact(run: EpisodeRun) -> dict[str, Any]:
    """The complete, detached record of one episode, at the current contract.

    A run whose decisions came off a provider is refused here unless it carries
    the binding to the journal that witnessed it. The refusal is at *build*
    time and not only at read time on purpose: an artefact this build would
    refuse to read is an artefact it will not write, and a caller that ran a
    model without an evidence recorder should be told so before a file exists.
    """
    scenario = run.spec.scenario(run.scenario_id)
    decisions = run.tape.as_list()
    binding = run.provider_execution
    if run.execution.outcome_source == SOURCE_MODEL and binding is None:
        raise ArtifactError(f"this run {_NO_UNWITNESSED_MODEL_ARTEFACT}")
    if run.execution.outcome_source != SOURCE_MODEL and binding is not None:
        raise ArtifactError(
            f"this run carries a provider execution binding for a "
            f"{run.execution.outcome_source!r} execution; neither an in-process "
            "agent nor a recorded playback reaches a provider, so a binding on one "
            "attributes a provider session to a run that made none"
        )
    payload: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "engine_version": OPERATEBENCH_VERSION,
        "operation": run.spec.identity_payload(),
        "operation_instance_id": run.operation_instance_id,
        "scenario_id": run.scenario_id,
        "semantic_scenario_id": scenario.semantic_scenario_id,
        "scenario_label": scenario.label,
        "agent_id": run.agent_id,
        "agent_kind": run.identity()["agent_kind"],
        "expected_terminal": scenario.expected_terminal,
        "status": run.outcome.status,
        "terminal_outcome": run.outcome.terminal_outcome,
        "replay_final": run.outcome.replay_final,
        "started_at": run.outcome.started_at,
        "ended_at": run.outcome.ended_at,
        "simulated_minutes": run.outcome.simulated_minutes,
        "agent_invocations": run.outcome.invocations,
        "events": [dict(event) for event in run.outcome.events],
        "trajectory": [dict(record) for record in run.outcome.trajectory],
        "final_state": dict(run.outcome.final_state),
        "final_state_digest_sha256": run.outcome.final_state_digest_sha256,
        "trajectory_digest_sha256": run.outcome.trajectory_digest_sha256,
        "evaluation": run.evaluation.as_dict(),
        "decisions": decisions,
        "decisions_digest_sha256": _content_digest(decisions),
        "agent_execution": run.execution.as_dict(),
        "provider_execution": None if binding is None else dict(binding),
        "note": ARTIFACT_NOTE,
    }
    return payload


def artifact_text(run: EpisodeRun) -> str:
    """The exact bytes an artefact is written as, as text."""
    try:
        return canonical_json_text(build_artifact(run), "operatebench run artefact")
    except JsonSafetyError as exc:
        raise ArtifactError(
            f"this run cannot be written as canonical JSON: {exc}"
        ) from exc


# ------------------------------------------------------------------ safe writing


def _artifact_write_error(failure: _WriteOnceFailure) -> ArtifactError:
    target = failure.path
    if failure.reason == "parent_symlink":
        return ArtifactError(
            f"{failure.component} is a symlink on the path to {target}; a run "
            "artefact is reachable through real directories only, so it cannot be "
            "redirected onto a file nobody named"
        )
    if failure.reason == "final_symlink":
        return ArtifactError(
            f"{target} is a symlink; a run artefact is written in place, never "
            "through a link — following one would create or overwrite its target "
            "instead of the path that was asked for"
        )
    if failure.reason == "exists":
        return ArtifactError(
            f"{target} already exists; a run artefact is written once so a replay "
            "cannot silently compare against a rewritten record"
        )
    if failure.reason in {"parent_create", "parent_open", "anchor"}:
        return ArtifactError(f"cannot create the directory for run artefact {target}")
    if failure.reason == "unsupported":
        return ArtifactError(
            "cannot write a run artefact safely on this platform: descriptor-anchored "
            "no-follow filesystem primitives are unavailable"
        )
    return ArtifactError(f"cannot write run artefact {target}")


def write_artifact(
    run: EpisodeRun, path: str | Path, *, dir_fd: int | None = None
) -> Path:
    """Write one run to ``path``, exactly once, without following anything.

    The bytes are produced before the file exists, so a run that cannot be
    represented is refused without leaving anything behind. Every directory is
    opened without following links and kept open; missing parents are created
    against those held descriptors. The final basename is created exclusively
    against the verified parent descriptor, written in full, and both file and
    parent are fsynced before success is returned.
    """
    target = Path(path)
    rendered = (artifact_text(run) + "\n").encode("utf-8")
    if dir_fd is None:
        return _write_once_bytes(rendered, target, error_adapter=_artifact_write_error)
    raw_name = os.fspath(path)
    return _write_once_bytes_at(
        rendered,
        raw_name,
        dir_fd=dir_fd,
        display_path=target,
        error_adapter=_artifact_write_error,
    )


# ------------------------------------------------------------------ validation


def _require(condition: bool, where: str, message: str) -> None:
    if not condition:
        raise ArtifactError(f"{where}: {message}")


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    _require(
        isinstance(value, Mapping),
        where,
        f"must be a JSON object, got {type(value).__name__}",
    )
    assert isinstance(value, Mapping)
    return value


def _listing(value: Any, where: str) -> Sequence[Any]:
    _require(
        isinstance(value, list),
        where,
        f"must be a JSON array, got {type(value).__name__}",
    )
    assert isinstance(value, list)
    return value


def _exact_fields(mapping: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    unknown = sorted(set(mapping) - set(allowed))
    _require(
        not unknown,
        where,
        f"unknown field(s) {unknown}; this build refuses an artefact it cannot "
        f"fully honour. Known fields are {sorted(allowed)}",
    )
    missing = sorted(set(allowed) - set(mapping))
    _require(not missing, where, f"missing required field(s) {missing}")


def _text(value: Any, where: str) -> str:
    _require(isinstance(value, str) and bool(value), where, "must be a non-empty string")
    assert isinstance(value, str)
    return value


def _optional_text(value: Any, where: str) -> str | None:
    if value is None:
        return None
    return _text(value, where)


def _flag(value: Any, where: str) -> bool:
    _require(isinstance(value, bool), where, "must be true or false")
    assert isinstance(value, bool)
    return value


def _counter(value: Any, where: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        where,
        "must be an integer (a boolean is not a quantity)",
    )
    assert isinstance(value, int)
    _require(value >= 0, where, f"must not be negative, got {value}")
    return value


def _instant(value: Any, where: str) -> str:
    try:
        parse_timestamp(value, where)
    except MalformedTimestampError as exc:
        raise ArtifactError(str(exc)) from exc
    assert isinstance(value, str)
    return value


def _digest_text(value: Any, where: str) -> str:
    text = _text(value, where)
    _require(
        len(text) == 64 and set(text) <= _HEX,
        where,
        "must be a 64-character lowercase hexadecimal SHA-256 digest",
    )
    return text


def _validate_operation(payload: Any, where: str) -> None:
    body = _mapping(payload, where)
    _exact_fields(body, _OPERATION_FIELDS, where)
    for name in (
        "operation_id",
        "operation_type",
        "operation_version",
        "semantic_scenario_id",
        "privacy_status",
    ):
        _text(body[name], f"{where}.{name}")
    _digest_text(body["spec_digest_sha256"], f"{where}.spec_digest_sha256")
    scenario_ids = _listing(body["scenario_ids"], f"{where}.scenario_ids")
    for position, scenario_id in enumerate(scenario_ids):
        _text(scenario_id, f"{where}.scenario_ids[{position}]")


def _validate_events(payload: Any, where: str) -> None:
    for position, event in enumerate(_listing(payload, where)):
        at = f"{where}[{position}]"
        body = _mapping(event, at)
        _exact_fields(body, _EVENT_FIELDS, at)
        for name in ("event_id", "event_type", "actor_id", "verdict_code"):
            _text(body[name], f"{at}.{name}")
        _instant(body["at"], f"{at}.at")
        _counter(body["sequence"], f"{at}.sequence")
        _mapping(body["payload"], f"{at}.payload")
        _flag(body["triggers_agent"], f"{at}.triggers_agent")
        _optional_text(body["caused_by"], f"{at}.caused_by")
        _require(
            body["disposition"] in _DISPOSITIONS,
            f"{at}.disposition",
            f"must be one of {sorted(_DISPOSITIONS)}, got {body['disposition']!r}",
        )


def _positive_counter(value: Any, where: str, what: str) -> int:
    """A count this build only ever writes above zero."""
    count = _counter(value, where)
    _require(
        count > 0,
        where,
        f"must be a positive number of {what}: a bound of zero is one nothing in "
        "this build could have run under",
    )
    return count


def _retrieval_records(value: Any, where: str) -> None:
    """Hold a served row's ``records`` body to what canonical JSON can carry.

    Recursive rather than "is a mapping", and refusing rather than repairing.
    This body is hashed into the trajectory digest, compared field by field
    against a re-served read, and handed to an evaluator that recomputes a
    version over it. A value canonical UTF-8 JSON cannot round-trip — a lone
    surrogate, a non-string key, an infinity, a structure that contains itself —
    would make all three of those answer differently depending on which one ran,
    and the difference would look like a disagreement about the *run*.

    The safety walk already answers exactly this question for every value this
    build persists, so it is reused rather than restated; what is added here is
    the two shapes it cannot see from inside — that the root is an object, and
    that a failure is an :class:`ArtifactError` about a document rather than a
    generic encoding complaint.
    """
    body = _mapping(value, where)
    try:
        ensure_json_safe(body, where)
    except JsonSafetyError as exc:
        raise ArtifactError(
            f"{where}: this retrieved record body cannot be carried by canonical "
            f"UTF-8 JSON, so nothing can hash or compare it — {exc}"
        ) from exc


def _row_field(value: Any, kind: str, where: str) -> None:
    if kind == "text":
        _text(value, where)
    elif kind == "optional_text":
        _optional_text(value, where)
    elif kind == "string":
        _require(isinstance(value, str), where, "must be a string")
    elif kind == "counter":
        _counter(value, where)
    elif kind == "flag":
        _flag(value, where)
    elif kind == "instant":
        _instant(value, where)
    elif kind == "optional_instant":
        if value is not None:
            _instant(value, where)
    elif kind == "object":
        _mapping(value, where)
    elif kind == "bindings":
        # A name and a canonical identity, both strings. Not "an object": the
        # evaluator compares these against a proposal's payload, and a binding
        # whose value is a list or a number is one no comparison can be made
        # from — which is a refusal at the reader, not a mismatch downstream.
        body = _mapping(value, where)
        for name, identity in body.items():
            _text(identity, f"{where}.{name}")
    elif kind == "minutes":
        _positive_counter(value, where, "simulated minutes")
    elif kind == "optional_minutes":
        if value is not None:
            _positive_counter(value, where, "simulated minutes")
    elif kind == "retrieval_authority":
        code = _text(value, where)
        _require(
            code in RETRIEVAL_AUTHORITIES,
            where,
            f"{code!r} is not an authority class this build writes; known classes "
            f"are {list(RETRIEVAL_AUTHORITIES)}",
        )
    elif kind == "retrieval_initiator":
        code = _text(value, where)
        _require(
            code in RETRIEVAL_INITIATORS,
            where,
            f"{code!r} is not a retrieval initiator this build writes; known "
            f"initiators are {list(RETRIEVAL_INITIATORS)}. A record naming another "
            "one claims a read this build has no path to perform",
        )
    elif kind == "retrieval_refusal":
        code = _text(value, where)
        _require(
            code in RETRIEVAL_REFUSAL_CODES,
            where,
            f"{code!r} is not a retrieval refusal this build names; known codes are "
            f"{list(RETRIEVAL_REFUSAL_CODES)}",
        )
    elif kind == "invalidation_cause":
        code = _text(value, where)
        _require(
            code in INVALIDATION_CAUSES,
            where,
            f"{code!r} is not a reason this build invalidates derived context for; "
            f"known causes are {list(INVALIDATION_CAUSES)}",
        )
    elif kind == "retrieval_records":
        _retrieval_records(value, where)
    elif kind == "retrieval_requests":
        # The requests a recorded RETRIEVE decision carried. Bounds are the
        # *environment's* contract, not the tape's: a batch refused for being
        # empty or oversize has to be recordable, or the replay could not
        # reproduce the refusal it caused.
        for position, entry in enumerate(_listing(value, where)):
            body = _mapping(entry, f"{where}[{position}]")
            _exact_fields(body, ("tool", "arguments"), f"{where}[{position}]")
            _text(body["tool"], f"{where}[{position}].tool")
            _mapping(body["arguments"], f"{where}[{position}].arguments")
    elif kind == "classification":
        code = _text(value, where)
        _require(
            code in _CLASSIFICATION_CODES,
            where,
            f"{code!r} is not a model output classification this build writes; "
            f"known codes are {sorted(_CLASSIFICATION_CODES)}",
        )
    elif kind == "wait":
        # The one nested outcome. Held to the WAIT shape rather than to
        # "is a mapping", so an ASK cannot carry a wait nobody could honour.
        _validate_outcome_body(_mapping(value, where), where, expected_kind="WAIT")
    elif kind == "outcome":
        # A decision recorded in the trajectory, held to the same exact body as
        # the same decision recorded on the tape. One table, both places: a row
        # the evaluator reads cannot carry a kind, a field or a classification
        # code the tape would have been refused for.
        _validate_outcome_body(_mapping(value, where), where)
    else:  # "text_list"
        for position, item in enumerate(_listing(value, where)):
            _text(item, f"{where}[{position}]")


def _validate_outcome_body(
    outcome: Mapping[str, Any], where: str, *, expected_kind: str | None = None
) -> str:
    """Hold one recorded decision to the exact body its kind is written in.

    Returns the kind, so a caller can bind it to the attempt that produced it.
    Every field is named by :data:`_OUTCOME_SHAPES` and nothing else is
    tolerated: an unknown key in a recorded outcome is provider-controlled
    content this build never wrote, whether it is an invented argument or the
    prose that came beside the tool call.
    """
    kind = outcome.get("kind")
    _require(
        kind in _DECISION_KINDS,
        f"{where}.kind",
        f"{kind!r} is not a decision kind this build records; known kinds are "
        f"{sorted(_DECISION_KINDS)}",
    )
    assert isinstance(kind, str)
    if expected_kind is not None:
        _require(
            kind == expected_kind,
            f"{where}.kind",
            f"must be {expected_kind!r} here, got {kind!r}",
        )
    shape = _OUTCOME_SHAPES.get(kind)
    _require(
        shape is not None,
        f"{where}.kind",
        f"{kind!r} is a recordable decision kind with no body shape in this build, "
        "so its body cannot be held to anything",
    )
    assert shape is not None
    _exact_fields(outcome, ("kind", *shape), where)
    for name, field_kind in shape.items():
        _row_field(outcome[name], field_kind, f"{where}.{name}")
    return kind


def _validate_row_body(
    body: Mapping[str, Any], record_type: str, where: str, version: int
) -> None:
    """Hold one row to exactly one of the shapes its record type is written in.

    ``version`` selects the table, so a row is held to the body the contract
    *this document declares* was written under — not to the union of every body
    this build has ever written, which would enforce none of them.
    """
    shapes = _ROW_SHAPES_BY_VERSION[version].get(record_type)
    # Two different refusals wearing one shape check. A row type this build's
    # runtime writes and an *earlier* contract's did not is a document claiming
    # evidence its own contract could not have produced — the exact forgery a
    # version-aware reader exists to refuse — and it is named as that rather
    # than as an internal coverage gap. The second arm is the coverage gap, kept
    # because :func:`row_shape_coverage_problem` and the suite are what hold the
    # tables in step and an untrusted document must still meet a named refusal
    # if they ever drift.
    _require(
        shapes is not None or record_type not in _record_types_for(ARTIFACT_VERSION),
        where,
        f"a {record_type!r} record is written by contract {ARTIFACT_VERSION} and not "
        f"by contract {version}, which this document declares; a record carrying it "
        "describes a run its own contract had no way to produce",
    )
    _require(
        shapes is not None,
        where,
        f"a {record_type!r} record is a type this build's ledger writes and this "
        "reader has no shape for, so its body cannot be held to anything",
    )
    assert shapes is not None
    carried = set(body) - set(_TRAJECTORY_FIELDS)
    for shape in shapes:
        if carried != set(shape):
            continue
        for name, kind in shape.items():
            _row_field(body[name], kind, f"{where}.{name}")
        return
    _require(
        False,
        where,
        f"a {record_type!r} record carries {sorted(carried)}, which is none of the "
        f"shapes contract {version} is written in: "
        f"{[sorted(shape) for shape in shapes]}",
    )


def _validate_trajectory(payload: Any, where: str, version: int) -> None:
    rows = _listing(payload, where)
    previous: int | None = None
    for position, row in enumerate(rows):
        at = f"{where}[{position}]"
        body = _mapping(row, at)
        missing = sorted(set(_TRAJECTORY_FIELDS) - set(body))
        _require(not missing, at, f"missing required field(s) {missing}")
        _require(
            body["index"] == position,
            f"{at}.index",
            f"must be {position}: a trajectory is an append-only sequence and its "
            f"indices are assigned by the ledger, got {body['index']!r}",
        )
        moment = parse_timestamp(body["at"], f"{at}.at")
        _require(
            previous is None or moment >= previous,
            f"{at}.at",
            "is before the row that precedes it; simulated time never moves backwards",
        )
        previous = moment
        record_type = body["record_type"]
        _require(
            record_type in RECORD_TYPES,
            f"{at}.record_type",
            f"{record_type!r} is not a record type this build writes",
        )
        _validate_row_body(body, str(record_type), at, version)


def _validate_evaluation(payload: Any, where: str, version: int) -> None:
    """Hold the result vector to its shape *and* to what it claims.

    Three of the four fields beside ``dimensions`` are not independent data. The
    writer derives ``failed_dimensions``, ``finding_codes`` and ``reliable`` from
    the dimensions — see :class:`~operatebench.core.evaluation.OperationEvaluation`
    — so a record whose derived fields disagree with the vector beside them
    describes no run this build could have produced. Checking each field's type
    and never comparing them accepted exactly that: a green ``reliable`` beside a
    failing dimension, an empty ``failed_dimensions`` beside a vector that
    failed, a finding code no finding carries.

    The dimension names are held to *this contract's* vector, exactly and in
    order. Missing, unknown, duplicated and reordered are one check, because they
    are one question — "is this the vector contract ``version`` graded?" — and
    four separate checks would each answer part of it.

    Deliberately absent: any rule tying ``ok`` to ``findings``. The writer's
    contract does not state one; ``Dimension`` carries the two independently, and
    a reader that inferred the rule would refuse records the writer is free to
    produce.
    """
    body = _mapping(payload, where)
    _exact_fields(body, _EVALUATION_FIELDS, where)
    _text(body["terminal_outcome"], f"{where}.terminal_outcome")
    legitimate = _flag(body["legitimate_completion"], f"{where}.legitimate_completion")
    reliable = _flag(body["reliable"], f"{where}.reliable")
    for name in ("failed_dimensions", "finding_codes"):
        for position, item in enumerate(_listing(body[name], f"{where}.{name}")):
            _text(item, f"{where}.{name}[{position}]")
    dimensions = _listing(body["dimensions"], f"{where}.dimensions")
    names: list[str] = []
    failed: list[str] = []
    codes: list[str] = []
    for position, dimension in enumerate(dimensions):
        at = f"{where}.dimensions[{position}]"
        entry = _mapping(dimension, at)
        _exact_fields(entry, _DIMENSION_FIELDS, at)
        names.append(_text(entry["name"], f"{at}.name"))
        ok = _flag(entry["ok"], f"{at}.ok")
        if not ok:
            failed.append(str(entry["name"]))
        _require(isinstance(entry["note"], str), f"{at}.note", "must be a string")
        counts = _mapping(entry["counts"], f"{at}.counts")
        for key, value in counts.items():
            _require(
                isinstance(value, int) and not isinstance(value, bool),
                f"{at}.counts.{key}",
                "must be an integer",
            )
        for index, finding in enumerate(_listing(entry["findings"], f"{at}.findings")):
            place = f"{at}.findings[{index}]"
            record = _mapping(finding, place)
            _exact_fields(record, _FINDING_FIELDS, place)
            codes.append(_text(record["code"], f"{place}.code"))
            _require(
                version >= ARTIFACT_VERSION
                or record["code"]
                not in {
                    "REQUIRED_EVIDENCE_REF_NOT_CITED",
                    "REQUIRED_EVIDENCE_SELECTOR_UNRESOLVED",
                },
                f"{place}.code",
                f"{record['code']} belongs to artifact contract 8; "
                f"contract {version} did not own that evaluation semantic",
            )
            _require(
                isinstance(record["detail"], str), f"{place}.detail", "must be a string"
            )
            if record["at"] is not None:
                _text(record["at"], f"{place}.at")

    expected = evaluation_dimensions_for(version)
    _require(
        tuple(names) == expected,
        f"{where}.dimensions",
        f"is not the result vector contract {version} graded. That contract's "
        f"vector is exactly {list(expected)}, in that order; this record names "
        f"{names}. Missing {sorted(set(expected) - set(names))}, unknown "
        f"{sorted(set(names) - set(expected))}, duplicated "
        f"{sorted({name for name in names if names.count(name) > 1})}",
    )
    _require(
        list(body["failed_dimensions"]) == failed,
        f"{where}.failed_dimensions",
        f"must be exactly the dimensions that did not hold, in dimension order: "
        f"{failed}, not {list(body['failed_dimensions'])}",
    )
    _require(
        list(body["finding_codes"]) == codes,
        f"{where}.finding_codes",
        f"must be exactly the code of every finding, in dimension and finding "
        f"order: {codes}, not {list(body['finding_codes'])}",
    )
    _require(
        reliable == (legitimate and not failed),
        f"{where}.reliable",
        f"is the strict conjunction of legitimate completion with every "
        f"dimension holding, and this record says legitimate_completion="
        f"{legitimate} beside failed dimension(s) {failed}, so it must be "
        f"{legitimate and not failed}",
    )


def _validate_decisions(payload: Any, where: str) -> list[str]:
    """Hold the decision tape to its shape, its order and its own vocabulary.

    Two conditions here are not shape checks and are the point of the tape.
    ``(invocation_index, turn_index)`` must be strictly increasing in the order
    the engine produces them, so a record cannot claim two decisions were made
    at one moment or that time ran backwards through the invocations; and every
    outcome must be one this build records — *body and all*, not kind alone, so
    a decision carrying an argument nobody wrote or the prose that arrived
    beside a tool call is refused before anything replays it.

    Hands back the kind of each decision in order, which is what binds the tape
    to the attempts that produced it in :func:`_validate_provenance`.
    """
    kinds: list[str] = []
    previous: tuple[int, int] | None = None
    for position, decision in enumerate(_listing(payload, where)):
        at = f"{where}[{position}]"
        body = _mapping(decision, at)
        _exact_fields(body, _DECISION_FIELDS, at)
        invocation = _counter(body["invocation_index"], f"{at}.invocation_index")
        turn = _counter(body["turn_index"], f"{at}.turn_index")
        _require(
            invocation >= 1,
            f"{at}.invocation_index",
            "must be at least 1: invocations are counted from one, and a decision "
            "made on invocation zero was made before the agent was invoked",
        )
        _require(
            previous is None or (invocation, turn) > previous,
            at,
            f"records invocation {invocation} turn {turn} after {previous}; decisions "
            "are appended in the order the engine asked for them, so a tape whose "
            "order moved describes a different run",
        )
        previous = (invocation, turn)
        _digest_text(body["observation_digest_sha256"], f"{at}.observation_digest_sha256")
        outcome = _mapping(body["outcome"], f"{at}.outcome")
        kinds.append(_validate_outcome_body(outcome, f"{at}.outcome"))
    return kinds


def _validate_execution(payload: Any, where: str) -> Mapping[str, Any]:
    """Hold the execution record to its shape, and to its own consistency.

    The consistency is the part that matters. ``excluded`` and
    ``exclusion_code`` move together in both directions, so there is no shape of
    this record that reports a fault without excluding the run, or excludes a
    run without saying which fault ended it — and then an excluded record is
    refused outright, because this build writes no artefact for a run that
    stopped at the provider. ``retry_count`` is pinned to the build's value: a
    record claiming retries this build does not perform would describe requests
    that were never sent.

    What this function deliberately does *not* do is compare the execution to
    the rest of the artefact. That is :func:`_validate_provenance`, which needs
    the tape and the identity as well, and keeping the two apart is what lets a
    refusal say whether the execution record is malformed or merely describes a
    different run from the one recorded beside it.
    """
    body = _mapping(payload, where)
    _exact_fields(body, _EXECUTION_FIELDS, where)
    source = _text(body["outcome_source"], f"{where}.outcome_source")
    _require(
        source in OUTCOME_SOURCES,
        f"{where}.outcome_source",
        f"{source!r} is not an outcome source this build writes; known sources are "
        f"{list(OUTCOME_SOURCES)}",
    )
    _optional_text(body["model"], f"{where}.model")
    _optional_text(body["protocol_version"], f"{where}.protocol_version")
    if body["max_output_tokens"] is not None:
        _positive_counter(
            body["max_output_tokens"], f"{where}.max_output_tokens", "output tokens"
        )
    calls = _counter(body["transport_calls"], f"{where}.transport_calls")
    retries = _counter(body["retry_count"], f"{where}.retry_count")
    _require(
        retries == RETRY_COUNT,
        f"{where}.retry_count",
        f"is {retries} and this build performs {RETRY_COUNT}; a record claiming "
        "retries that were never made describes requests that were never sent",
    )
    excluded = _flag(body["excluded"], f"{where}.excluded")
    code = _optional_text(body["exclusion_code"], f"{where}.exclusion_code")
    _require(
        isinstance(body["exclusion_detail"], str),
        f"{where}.exclusion_detail",
        "must be a string",
    )
    _require(
        excluded == (code is not None),
        where,
        f"says excluded={excluded} with exclusion_code={code!r}; a run is excluded "
        "with a named fault or it is not excluded at all — there is no third state",
    )
    if code is not None:
        _require(
            code in FAULTS,
            f"{where}.exclusion_code",
            f"{code!r} is not a fault this build classifies; known faults are "
            f"{list(FAULTS)}",
        )
    # Fail closed, and only after the pairing above, so a half-stated exclusion
    # is named as a half-stated exclusion rather than as this refusal.
    _require(not excluded, where, _NO_EXCLUDED_ARTEFACT)

    if source in (SOURCE_DETERMINISTIC, SOURCE_RECORDED):
        _require(
            calls == 0,
            f"{where}.transport_calls",
            f"is {calls} for a {source!r} run; neither an in-process agent nor a "
            "recorded playback has a provider to call, so any count above zero "
            "means this record describes a different execution",
        )
    attempts = _listing(body["attempts"], f"{where}.attempts")
    _require(
        len(attempts) <= calls,
        f"{where}.attempts",
        f"records {len(attempts)} attempt(s) against {calls} transport call(s); a "
        "record cannot hold more attempts than calls were made",
    )
    for position, attempt in enumerate(attempts):
        at = f"{where}.attempts[{position}]"
        entry = _mapping(attempt, at)
        _exact_fields(entry, _ATTEMPT_FIELDS, at)
        _counter(entry["invocation_index"], f"{at}.invocation_index")
        _counter(entry["turn_index"], f"{at}.turn_index")
        _digest_text(entry["request_digest_sha256"], f"{at}.request_digest_sha256")
        result = _text(entry["outcome"], f"{at}.outcome")
        _require(
            result in _ATTEMPT_OUTCOMES,
            f"{at}.outcome",
            f"{result!r} is not how an attempt ends in this build; known endings are "
            f"{sorted(_ATTEMPT_OUTCOMES)}",
        )
        fault = _optional_text(entry["fault"], f"{at}.fault")
        _require(
            (fault is not None) == (result == "failed"),
            at,
            f"ended {result!r} with fault {fault!r}; an attempt that failed names the "
            "fault that ended it, and one that did not names none",
        )
        if fault is not None:
            _require(
                fault in FAULTS,
                f"{at}.fault",
                f"{fault!r} is not a fault this build classifies; known faults are "
                f"{list(FAULTS)}",
            )
    return body


def _amount_text(value: Any, where: str) -> str:
    """One exact USD amount, as a decimal string and never as a JSON number.

    The same rule the execution ledger holds its amounts to, restated here for
    the same reason: a JSON number is a binary float to most readers, and the
    amount a run was authorised for is not a number that survives being one.
    """
    _require(
        isinstance(value, str),
        where,
        f"a USD amount is an exact decimal string, not a {type(value).__name__}; "
        "a JSON number is a binary float to most readers and is not the amount "
        "that was authorised",
    )
    text = str(value)
    try:
        parsed = Decimal(text)
    except InvalidOperation as exc:
        raise ArtifactError(f"{where}: {text!r} is not a decimal amount") from exc
    _require(
        parsed.is_finite() and parsed >= 0,
        where,
        "a USD amount is finite and non-negative",
    )
    _require(
        usd_text(parsed) == text,
        where,
        f"{text!r} is not the canonical rendering of the amount it names, which is "
        f"{usd_text(parsed)!r}",
    )
    return text


def _validate_provider_execution(payload: Any, where: str) -> Mapping[str, Any]:
    """Hold the binding to its own shape. Cross-field work is provenance's.

    Everything here is answerable from the binding alone: the exact field set,
    the form of each identity, digest, count and amount, and the shape of the
    decision-to-call mapping. Whether those values agree with the execution
    record and the tape beside them is :func:`_validate_provenance`'s question,
    and whether they agree with the *ledger* is nobody's question here — that
    file is not in this document, and this reader does not open it.
    """
    body = _mapping(payload, where)
    _exact_fields(body, PROVIDER_EXECUTION_FIELDS, where)

    identity = body["execution_run_id"]
    problem = execution_run_id_problem(identity)
    _require(problem is None, f"{where}.execution_run_id", str(problem))

    version = body["execution_ledger_version"]
    _require(
        version in SUPPORTED_EXECUTION_LEDGER_VERSIONS,
        f"{where}.execution_ledger_version",
        f"{version!r} is not an execution ledger contract this build reads, "
        f"which are {list(SUPPORTED_EXECUTION_LEDGER_VERSIONS)}; a binding to a "
        "journal shape this "
        "build cannot read is a binding nothing can check",
    )
    for name in (
        "execution_ledger_digest_sha256",
        "settings_digest_sha256",
        "pricing_digest_sha256",
    ):
        _digest_text(body[name], f"{where}.{name}")

    provider = _text(body["provider"], f"{where}.provider")
    _require(
        provider in BINDABLE_PROVIDERS,
        f"{where}.provider",
        f"{provider!r} is not a provider this build binds evidence for; it binds "
        f"{list(BINDABLE_PROVIDERS)}",
    )
    api = _text(body["api"], f"{where}.api")
    _require(
        api in BINDABLE_PROVIDER_APIS,
        f"{where}.api",
        f"{api!r} is not a provider API this build binds evidence for; it binds "
        f"{list(BINDABLE_PROVIDER_APIS)}",
    )
    _require(
        (provider, api) in BINDABLE_PROVIDER_API_PAIRS,
        where,
        f"provider {provider!r} cannot bind API {api!r}; allowed pairs are "
        f"{list(BINDABLE_PROVIDER_API_PAIRS)}",
    )
    _text(body["model"], f"{where}.model")

    calls = _positive_counter(body["provider_calls"], f"{where}.provider_calls", "calls")
    attempts = _positive_counter(body["attempts"], f"{where}.attempts", "attempts")
    _require(
        attempts >= calls,
        f"{where}.attempts",
        f"records {attempts} attempt(s) against {calls} provider call(s); every "
        "call made at least one attempt, so fewer attempts than calls describes "
        "requests that were never dispatched",
    )
    for name in ("input_tokens", "output_tokens"):
        _counter(body[name], f"{where}.{name}")
    for name in ("measured_cost_usd", "forfeited_reservation_usd"):
        _amount_text(body[name], f"{where}.{name}")

    indexes = _listing(body["decision_call_index"], f"{where}.decision_call_index")
    for position, index in enumerate(indexes):
        at = f"{where}.decision_call_index[{position}]"
        _counter(index, at)
        _require(
            int(index) < calls,
            at,
            f"names call {index} against {calls} recorded provider call(s); a "
            "decision cannot have come from a call this run never made",
        )
    return body


#: The providers and APIs a binding may name. Closed, because "some other
#: string" is how a binding grows a provider nobody wired: this build reaches
#: exactly four, through exactly three APIs, and a record naming another describes a
#: session no code here can have produced.
BINDABLE_PROVIDERS: tuple[str, ...] = ("openai", "anthropic", "xai", "mistral")
BINDABLE_PROVIDER_APIS: tuple[str, ...] = ("responses", "messages", "chat_completions")
BINDABLE_PROVIDER_API_PAIRS: tuple[tuple[str, str], ...] = (
    ("openai", "responses"),
    ("anthropic", "messages"),
    ("xai", "chat_completions"),
    ("xai", "responses"),
    ("mistral", "chat_completions"),
)


def _validate_binding_coherence(body: Mapping[str, Any], where: str) -> None:
    """Bind the provider execution binding to the rest of the document.

    Six clauses, and each one is a forgery that passed the shape check above:

    * a binding on a run that reached no provider, or none on a run that did;
    * a binding naming a model the execution record does not name;
    * a call or attempt count the execution record's own attempts contradict;
    * a decision-to-call mapping that is not one index per recorded decision;
    * a mapping that is not the exact contiguous run ``0 … n-1`` — a model run
      makes one call per decision in the order the engine asked for them, so
      any other mapping describes a run whose calls and decisions were paired
      by something other than the order they happened in;
    * a binding on a record whose execution is excluded, which cannot exist,
      because an excluded run has no episode artefact at all.
    """
    binding = body["provider_execution"]
    execution = body["agent_execution"]
    source_kind = str(execution["outcome_source"])
    at = f"{where}.provider_execution"

    if source_kind != SOURCE_MODEL:
        _require(
            binding is None,
            at,
            f"is stated for a {source_kind!r} execution; neither an in-process "
            "agent nor a recorded playback reaches a provider, so a binding on one "
            "attributes a provider session to a run that made none",
        )
        return
    _require(binding is not None, at, f"{_NO_UNWITNESSED_MODEL_ARTEFACT}")
    assert binding is not None

    _require(
        not bool(execution["excluded"]),
        at,
        "sits on a record whose execution is excluded; an excluded run has no "
        "episode artefact at all, so there is nothing for a binding to bind",
    )
    _require(
        binding["model"] == execution["model"],
        f"{at}.model",
        f"names {binding['model']!r} and the execution record names "
        f"{execution['model']!r}; one run asked one model, and a record naming two "
        "describes two",
    )

    attempts = list(execution["attempts"])
    decisions = list(body["decisions"])
    _require(
        int(binding["provider_calls"]) == len(attempts),
        f"{at}.provider_calls",
        f"is {binding['provider_calls']} against {len(attempts)} recorded "
        "attempt(s) in the execution record; this build appends exactly one "
        "attempt per provider call, so the two are one number",
    )
    _require(
        int(binding["attempts"]) == len(attempts),
        f"{at}.attempts",
        f"is {binding['attempts']} against {len(attempts)} recorded attempt(s) in "
        "the execution record",
    )
    indexes = [int(index) for index in binding["decision_call_index"]]
    _require(
        len(indexes) == len(decisions),
        f"{at}.decision_call_index",
        f"names {len(indexes)} call(s) against {len(decisions)} recorded "
        "decision(s); the mapping states which call produced each decision, so a "
        "mapping of another length describes a different run",
    )
    _require(
        indexes == list(range(len(decisions))),
        f"{at}.decision_call_index",
        "is not the exact contiguous mapping this build writes. A model run makes "
        f"one call per decision, in order, so the mapping is 0 to {len(decisions) - 1}"
        f"; a repeated, reordered or gapped index pairs a decision with a call it "
        "did not come from",
    )
    protocol = execution["protocol_version"]
    _require(
        protocol in REPRODUCIBLE_MODEL_PROTOCOLS,
        f"{where}.agent_execution.protocol_version",
        f"is {protocol!r} on a record carrying a provider execution binding; a "
        f"binding is evidence about a session this build can rebuild, and this "
        f"build rebuilds {list(REPRODUCIBLE_MODEL_PROTOCOLS)}",
    )


def derived_agent_kind(agent_id: str, outcome_source: str) -> str:
    """What kind of agent produced this run, derived rather than believed.

    The registry answers for an in-process agent, and the execution source
    answers for the two kinds no registry can speak for. Nothing here reads the
    ``agent_kind`` an artefact carries: a field a record supplies about itself
    cannot also be the check on that record, and re-deriving it is what makes
    "this deterministic control's run was relabelled as a model's" a difference
    a replay reports rather than a claim it repeats back.
    """
    if outcome_source == SOURCE_MODEL:
        return KIND_MODEL
    if outcome_source == SOURCE_RECORDED:
        return KIND_RECORDED
    entry = AGENTS.get(agent_id)
    if entry is None:
        raise ArtifactError(
            f"agent_id {agent_id!r} is not an agent this build registers, so the "
            "kind of a deterministic run it produced cannot be derived; registered "
            f"agents are {sorted(AGENTS)}"
        )
    return entry.kind


def _validate_provenance(
    body: Mapping[str, Any], kinds: Sequence[str], where: str
) -> None:
    """Bind identity, execution and tape to each other, or refuse by name.

    Each of the three is well-formed on its own by the time this runs. What is
    checked here is whether they describe *one execution*, and every clause
    below is a forgery that passed a shape check:

    * a deterministic control's run relabelled ``model`` — a registered agent
      has no provider to have called;
    * a model run naming no model, or a protocol version this build cannot
      parse a response under;
    * a transport-call count nobody made — calls, attempts and decisions have
      one cardinality between them, not three;
    * an attempt reordered, removed or added — attempt *i* and decision *i* are
      the same invocation and turn or they are not the same run;
    * an attempt that says the provider answered with a decision sitting beside
      a decision that says it did not, in either direction;
    * a tape claiming more invocations than the episode records.

    The one binding deliberately left to replay is the exact agent kind of a
    deterministic run. :func:`derived_agent_kind` re-derives it from the
    registry and the rebuilt artefact carries the derived value, so a
    mislabelled control is a *reported difference* — which says which field
    diverged — rather than a refusal to read the file at all.
    """
    execution = body["agent_execution"]
    source = str(execution["outcome_source"])
    agent_id = str(body["agent_id"])
    agent_kind = str(body["agent_kind"])
    registered = AGENTS.get(agent_id)

    if source == SOURCE_DETERMINISTIC:
        _require(
            registered is not None,
            f"{where}.agent_id",
            f"{agent_id!r} is not an agent this build registers, and a deterministic "
            "run's decisions came from an in-process agent that has to exist to have "
            f"produced them; registered agents are {sorted(AGENTS)}",
        )
        _require(
            agent_kind in DETERMINISTIC_AGENT_KINDS,
            f"{where}.agent_kind",
            f"{agent_kind!r} is not a kind this build's agent registry declares "
            f"({sorted(DETERMINISTIC_AGENT_KINDS)}), and the execution record says "
            "these decisions came from an in-process agent",
        )
    else:
        _require(
            registered is None,
            f"{where}.agent_id",
            f"{agent_id!r} is a registered in-process agent, and this record says "
            f"its decisions came from a {source!r} execution; a deterministic "
            "control has no provider to have called, so this record relabels one "
            "run as another",
        )
        expected_kind = KIND_MODEL if source == SOURCE_MODEL else KIND_RECORDED
        _require(
            agent_kind == expected_kind,
            f"{where}.agent_kind",
            f"is {agent_kind!r} for a {source!r} execution, which is recorded as "
            f"{expected_kind!r}; the kind of an agent no registry declares is "
            "derived from what produced its decisions, and nothing else",
        )

    model = execution["model"]
    protocol = execution["protocol_version"]
    ceiling = execution["max_output_tokens"]
    calls = int(execution["transport_calls"])
    attempts = list(execution["attempts"])
    decisions = list(body["decisions"])
    at = f"{where}.agent_execution"

    if source == SOURCE_MODEL:
        _require(
            isinstance(model, str) and bool(model),
            f"{at}.model",
            "must name the model a model run's decisions came from; an unnamed "
            "model cannot be the recorded identity of anything it produced",
        )
        _require(
            isinstance(protocol, str) and protocol in SUPPORTED_MODEL_PROTOCOLS,
            f"{at}.protocol_version",
            f"{protocol!r} is not a model protocol this build speaks; it speaks "
            f"{list(SUPPORTED_MODEL_PROTOCOLS)}, and a response parsed under one "
            "protocol version is not evidence about another",
        )
        _require(
            ceiling is not None,
            f"{at}.max_output_tokens",
            "must state the output ceiling a model run was requested under; it is "
            "part of every request's identity, so a record without it cannot have "
            "its attempts rebuilt and its provider identity would be a label "
            "nothing reads",
        )
        _require(
            calls == len(attempts),
            at,
            f"records {calls} transport call(s) against {len(attempts)} attempt(s); "
            "this build appends exactly one attempt per call it makes, so a count "
            "that is not the number of attempts describes calls nobody made",
        )
        _require(
            len(attempts) == len(decisions),
            at,
            f"records {len(attempts)} attempt(s) against {len(decisions)} recorded "
            "decision(s); every decision in a model run came back from exactly one "
            "provider call, so an attempt with no decision or a decision with no "
            "attempt is a tape and an execution that describe different runs",
        )
    else:
        _require(
            model is None and protocol is None and ceiling is None,
            at,
            f"names model {model!r}, protocol {protocol!r} and output ceiling "
            f"{ceiling!r} for a {source!r} execution; neither an in-process agent "
            "nor a recorded playback reaches a model, so a record naming one "
            "attributes these decisions to a provider that was never called",
        )
        # Nothing here about attempts: an in-process or replayed execution is
        # already held to zero calls, and a record cannot hold more attempts
        # than calls were made, so "no attempts" is stated by those two together
        # rather than a third time in a clause nothing could reach.

    for position, (attempt, kind) in enumerate(zip(attempts, kinds, strict=False)):
        place = f"{at}.attempts[{position}]"
        decision = decisions[position]
        _require(
            (attempt["invocation_index"], attempt["turn_index"])
            == (decision["invocation_index"], decision["turn_index"]),
            place,
            f"was made on invocation {attempt['invocation_index']} turn "
            f"{attempt['turn_index']} and sits beside the decision recorded for "
            f"invocation {decision['invocation_index']} turn "
            f"{decision['turn_index']}; attempts and decisions are appended together "
            "in the order the engine asked for them, so a pair that does not line up "
            "means one of the two was reordered, removed or added",
        )
        expected = _ATTEMPT_OUTCOME_FOR_KIND.get(kind)
        _require(
            expected is not None and attempt["outcome"] == expected,
            place,
            f"ended {attempt['outcome']!r} beside a {kind!r} decision, which this "
            f"build records as {expected!r}; an attempt that failed at the provider "
            "produced no decision at all and excludes the run, so it cannot appear "
            "in an episode artefact",
        )

    digests = [str(attempt["request_digest_sha256"]) for attempt in attempts]
    _require(
        len(set(digests)) == len(digests),
        f"{at}.attempts",
        "record two calls whose request identities are the same digest; a request "
        "carries the invocation and turn it was made on, so two identical ones are "
        "one call written down twice",
    )

    invocations = int(body["agent_invocations"])
    for position, decision in enumerate(decisions):
        _require(
            int(decision["invocation_index"]) <= invocations,
            f"{where}.decisions[{position}].invocation_index",
            f"is {decision['invocation_index']} and the episode records "
            f"{invocations} agent invocation(s); a decision cannot have been made on "
            "an invocation the episode never reached",
        )


def validate_artifact(payload: Any, source: str = "run artefact") -> dict[str, Any]:
    """Hold a run artefact to its whole shape, and hand back a plain copy.

    Every field is validated here, before anything downstream reads one. That
    ordering is the point: an artefact whose identity happens to match must
    still be a well-formed artefact, or a replay would re-execute on the
    strength of one field of a document it never checked.
    """
    try:
        ensure_json_safe(payload, source)
    except JsonSafetyError as exc:
        raise ArtifactError(f"{source}: this is not a JSON-safe record: {exc}") from exc

    body = _mapping(payload, source)
    version = body.get("artifact_version")
    _require(
        isinstance(version, int)
        and not isinstance(version, bool)
        and version in SUPPORTED_ARTIFACT_VERSIONS,
        f"{source}.artifact_version",
        f"{version!r} is not a contract version this build reads (it reads "
        f"{list(SUPPORTED_ARTIFACT_VERSIONS)} and writes {ARTIFACT_VERSION})",
    )
    assert isinstance(version, int)
    # Held to *this version's* field set, exactly. A v1 record does not gain the
    # v2 fields by being read by a v2 build, and a v2 record does not get to
    # omit them: both directions of "unknown field" and "missing field" are
    # refusals, so an artefact cannot be half-read under a contract it does not
    # satisfy.
    _exact_fields(body, _FIELDS_BY_VERSION[version], source)

    _text(body["engine_version"], f"{source}.engine_version")
    _validate_operation(body["operation"], f"{source}.operation")
    if version >= ARTIFACT_VERSION_V6:
        # Held to its form, not merely to being text. The instance identity is
        # what a recorded decision's observation digest was taken over, so a
        # value that is not one this build mints would replay as a mismatch two
        # frames later instead of as a refusal here.
        instance_problem = operation_instance_id_problem(body["operation_instance_id"])
        _require(
            instance_problem is None,
            f"{source}.operation_instance_id",
            str(instance_problem),
        )
    for name in (
        "scenario_id",
        "semantic_scenario_id",
        "scenario_label",
        "agent_id",
        "agent_kind",
        "status",
    ):
        _text(body[name], f"{source}.{name}")
    expected_terminal = _text(body["expected_terminal"], f"{source}.expected_terminal")
    _require(
        expected_terminal in TERMINAL_OUTCOMES,
        f"{source}.expected_terminal",
        f"{expected_terminal!r} is not a terminal class; known classes are "
        f"{list(TERMINAL_OUTCOMES)}",
    )
    _optional_text(body["terminal_outcome"], f"{source}.terminal_outcome")
    _flag(body["replay_final"], f"{source}.replay_final")
    started = _instant(body["started_at"], f"{source}.started_at")
    ended = _instant(body["ended_at"], f"{source}.ended_at")
    _require(
        parse_timestamp(ended, "ended_at") >= parse_timestamp(started, "started_at"),
        f"{source}.ended_at",
        "is before the episode started; simulated time never moves backwards",
    )
    _counter(body["simulated_minutes"], f"{source}.simulated_minutes")
    _counter(body["agent_invocations"], f"{source}.agent_invocations")
    _validate_events(body["events"], f"{source}.events")
    _validate_trajectory(body["trajectory"], f"{source}.trajectory", version)

    operation_type = str(body["operation"]["operation_type"])
    state_validator = _STATE_VALIDATORS.get(operation_type)
    _require(
        state_validator is not None,
        f"{source}.operation.operation_type",
        f"{operation_type!r} has no canonical-state validator in this build, so the "
        "final state it records cannot be held to any shape; known operation types "
        f"are {sorted(_STATE_VALIDATORS)}",
    )
    assert state_validator is not None
    if operation_type == "lettings.maintenance.synthetic":
        validate_canonical_state(
            body["final_state"],
            f"{source}.final_state",
            reporting_actor=body["engine_version"] in {"0.11.0", "0.12.0", "0.13.0"},
        )
    else:
        state_validator(body["final_state"], f"{source}.final_state")

    _digest_text(body["final_state_digest_sha256"], f"{source}.final_state_digest_sha256")
    _digest_text(body["trajectory_digest_sha256"], f"{source}.trajectory_digest_sha256")
    _validate_evaluation(body["evaluation"], f"{source}.evaluation", version)
    _text(body["note"], f"{source}.note")

    digested: list[tuple[str, str]] = [
        ("final_state_digest_sha256", "final_state"),
        ("trajectory_digest_sha256", "trajectory"),
    ]
    if version >= ARTIFACT_VERSION_V2:
        kinds = _validate_decisions(body["decisions"], f"{source}.decisions")
        _validate_execution(body["agent_execution"], f"{source}.agent_execution")
        digested.append(("decisions_digest_sha256", "decisions"))
        # Cross-field, and here rather than at replay. Splitting the two
        # questions — validation answers "is this well-formed?", replay answers
        # "is this one run?" — works only for a field replay re-derives. A model
        # run's execution record is not one: it describes a provider session a
        # replay is required not to repeat, so a contradiction between that
        # record, the tape and the identity is one no reproduction can report.
        # Anything a reproduction cannot check, the reader has to; what is left
        # to replay is what replay genuinely re-derives. See
        # :func:`_validate_provenance`.
        _validate_provenance(body, kinds, source)
        if version >= ARTIFACT_VERSION_V7:
            # Shape first, then coherence, in the same order and for the same
            # reason as the execution record above: a malformed binding is named
            # as a malformed binding rather than as a run it disagrees with.
            if body["provider_execution"] is not None:
                _validate_provider_execution(
                    body["provider_execution"], f"{source}.provider_execution"
                )
            _validate_binding_coherence(body, source)
    else:
        # A v1 artefact carries no execution record, and every v1 run was an
        # in-process one: the model boundary did not exist under that contract.
        # So the identity binding applies with the source implied.
        _require(
            str(body["agent_id"]) in AGENTS,
            f"{source}.agent_id",
            f"{body['agent_id']!r} is not an agent this build registers; every v1 "
            "run's decisions came from an in-process agent, which has to exist to "
            f"have produced them. Registered agents are {sorted(AGENTS)}",
        )

    # Last, and before any caller acts on the payload: a record whose own digest
    # does not describe the payload written beside it is not this run's record,
    # whatever else about it is well-formed. Recomputed here rather than at
    # replay, so a reader that never replays still cannot be handed a document
    # whose state, trajectory or decisions were edited after they were hashed.
    for name, payload_field in digested:
        _require(
            _content_digest(body[payload_field]) == body[name],
            f"{source}.{name}",
            f"does not describe the {payload_field} recorded beside it; the content "
            "and the digest of that content disagree",
        )
    return dict(body)


def _read_artifact_text_at(name: str, dir_fd: int, source: Path) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name in (os.curdir, os.pardir)
        or "/" in name
        or "\\" in name
        or "\x00" in name
    ):
        raise ArtifactError(f"cannot read run artefact {source.name}")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dir_fd
        )
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise OSError(errno.EPERM, "artifact identity or mode changed")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        descriptor_to_close = descriptor
        descriptor = None
        os.close(descriptor_to_close)
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactError(
            f"{source.name} is not valid UTF-8: a run artefact is a UTF-8 text file"
        ) from exc
    except OSError as exc:
        raise ArtifactError(f"cannot read run artefact {source.name}") from exc
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)


def read_artifact(path: str | Path, *, dir_fd: int | None = None) -> dict[str, Any]:
    """Read and fully validate a run artefact."""
    source = Path(path)
    if dir_fd is not None:
        text = _read_artifact_text_at(os.fspath(path), dir_fd, source)
    else:
        try:
            text = source.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ArtifactError(
                f"{source} is not valid UTF-8: a run artefact is a UTF-8 text file "
                f"({exc.reason} at byte {exc.start})"
            ) from exc
        except OSError as exc:
            raise ArtifactError(f"cannot read run artefact {source}: {exc}") from exc
    try:
        ensure_raw_json_depth(text, str(source))
        payload = json.loads(text)
    except NestingDepthError as exc:
        raise ArtifactError(str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{source} is not valid JSON: {exc}") from exc
    return validate_artifact(payload, str(source))


# --------------------------------------------------------------------- replay


@dataclass(frozen=True)
class ReplayReport:
    """What a re-execution found when it compared itself to the record.

    ``sections`` carries a per-field verdict for every result-bearing field of
    the artefact, grouped the way an operator reads them. ``ok`` is the
    conjunction over all of them: there is no field a replay agrees about by
    not looking.

    ``reproduction`` says how the agent's decisions were obtained — ``rerun``
    means the agent was executed again, ``playback`` means they came off the
    recorded tape — and ``carried_forward`` names *every* field the replay took
    from the record instead of re-deriving.

    That list is empty for a rerun, which derives everything it compares. It is
    not empty for a playback, and what it holds is the honest extent of the
    asymmetry: the decisions came off the tape, the execution record of a model
    run describes a provider session a replay is required not to repeat, and
    the agent's identity is a label no registry can answer for once the
    decisions did not come from a registered in-process agent. A field in that
    list is a field :attr:`ok` says nothing about, so
    :attr:`identity_matches` on a playback is a verdict on the identity fields
    that were re-derived and not a claim that the whole identity was compared —
    :attr:`total_comparison` is the flag that says which of the two you are
    holding. An asymmetry a report states is one a reader can weigh; one it
    hides is a comparison that looks total and is not.

    What keeps a carried field from being unchecked is not this report. The
    reader has already bound the execution record, the decision tape and the
    identity to each other before any of this ran — see
    :func:`_validate_provenance` — so a carried record that contradicts the run
    it sits beside never reaches a replay.
    """

    scenario_id: str
    agent_id: str
    sections: Mapping[str, Mapping[str, bool]]
    recorded_final_state_digest: str
    replayed_final_state_digest: str
    recorded_trajectory_digest: str
    replayed_trajectory_digest: str
    differences: tuple[str, ...]
    artifact_version: int = ARTIFACT_VERSION_V1
    reproduction: str = "rerun"
    carried_forward: tuple[str, ...] = ()
    _replayed_outcome: EpisodeOutcome | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def _section_ok(self, name: str) -> bool:
        return all(self.sections[name].values())

    @property
    def identity_matches(self) -> bool:
        return self._section_ok("identity")

    @property
    def metadata_matches(self) -> bool:
        return self._section_ok("metadata")

    @property
    def events_match(self) -> bool:
        return self._section_ok("events")

    @property
    def trajectory_matches(self) -> bool:
        return self._section_ok("trajectory")

    @property
    def final_state_matches(self) -> bool:
        return self._section_ok("final_state")

    @property
    def evaluation_matches(self) -> bool:
        return self._section_ok("evaluation")

    @property
    def ok(self) -> bool:
        return all(self._section_ok(name) for name in self.sections)

    @property
    def total_comparison(self) -> bool:
        """Whether every compared field was re-derived rather than carried."""
        return not self.carried_forward

    @property
    def validated_episode_outcome(self) -> EpisodeOutcome | None:
        """Return the authenticated re-execution only after exact replay succeeds.

        The value is a current-process Engine outcome, not reconstructed or
        deserialized evidence.  A failing report withholds it, so diagnostics
        cannot accidentally become authority to rescore a recorded trajectory.
        """
        if not self.ok:
            return None
        return self._replayed_outcome

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "agent_id": self.agent_id,
            "artifact_version": self.artifact_version,
            "reproduction": self.reproduction,
            "carried_forward": list(self.carried_forward),
            "total_comparison": self.total_comparison,
            "ok": self.ok,
            "sections": {
                name: {"matches": self._section_ok(name), "fields": dict(fields)}
                for name, fields in self.sections.items()
            },
            "identity_matches": self.identity_matches,
            "metadata_matches": self.metadata_matches,
            "events_match": self.events_match,
            "final_state_matches": self.final_state_matches,
            "trajectory_matches": self.trajectory_matches,
            "evaluation_matches": self.evaluation_matches,
            "recorded_final_state_digest_sha256": self.recorded_final_state_digest,
            "replayed_final_state_digest_sha256": self.replayed_final_state_digest,
            "recorded_trajectory_digest_sha256": self.recorded_trajectory_digest,
            "replayed_trajectory_digest_sha256": self.replayed_trajectory_digest,
            "differences": list(self.differences),
        }


#: How a replay obtained the agent's decisions.
REPRODUCTION_RERUN = "rerun"
REPRODUCTION_PLAYBACK = "playback"


#: What a playback takes from the record instead of re-deriving, whatever the
#: source. The decisions are the record's by construction — that is what
#: playback *is* — and so is the agent's identity: no registry declares a
#: provider-backed or replayed agent, so ``agent_id`` cannot be checked against
#: anything and ``agent_kind`` follows from the execution source the record
#: itself states. Named here rather than inferred, because a comparison that
#: quietly includes fields it took from the thing it is comparing against is a
#: comparison that always passes.
_CARRIED_BY_PLAYBACK: tuple[str, ...] = (
    "agent_id",
    "agent_kind",
    "decisions",
    "decisions_digest_sha256",
)


def _not_reproducible(version: int) -> str:
    """Why a record outside the reproducible contract is refused whole.

    Contract 4 moved the model-visible observation — the projection every
    recorded decision's digest was taken over — and retracted the wait
    declaration's reachability claim. This build can neither rebuild the older
    projection without re-introducing the disclosure it removed, nor write back
    a claim about the authored future it no longer makes. A comparison assembled
    out of either would report agreement it did not establish.
    """
    return (
        f"this record was written under artefact contract {version}, and this "
        f"build reproduces contract(s) {list(REPRODUCIBLE_ARTIFACT_VERSIONS)}. "
        "Contract 6 made reading the only way an authoritative fact reaches an "
        "agent: the observation a decision is bound to carries the operation's "
        "coarse phase, the waking event's authority envelope and the read "
        "affordances, and no projection of the operation record at all. The "
        "agent returns batches of reads, the trajectory carries a provenance row "
        "for every record served and the authority every invocation was woken "
        "under, and a business proposal is refused unless its canonical required "
        "reads were established. A run under this build therefore produces "
        "neither the observation digests nor the trajectory rows an earlier "
        "record carries. It is still readable — it is what it is — but replaying "
        "it would compare two different contracts and call the difference a "
        "divergence. Contract 7 then bound an episode to the execution ledger "
        "that witnessed its provider session, so a record written before it "
        "carries no binding for a replay to compare and a rerun of one would "
        "produce a document with a field the record does not have"
    )


def _reproduce(
    spec: OperationSpec,
    recorded: Mapping[str, Any],
    scenario_id: str,
    agent_id: str,
) -> tuple[EpisodeRun, str, tuple[str, ...]]:
    """Produce the run this artefact claims to describe, without paying twice.

    A v1 artefact and a v2 one whose decisions came from an in-process agent
    re-execute the agent: that is free, and it checks the agent as well as the
    environment. Anything whose decisions came off a provider is reproduced from
    its own tape, because the alternative is a second provider run whose answers
    a stochastic agent is under no obligation to repeat.

    Either way the agent's kind is *derived* — from the registry for an
    in-process agent, from the execution source otherwise — and never read out
    of the record being checked. The tape is bound to the run all the same:
    playback refuses any decision offered an observation it was not produced
    from, so what is carried is the decision, not the state it answered.
    """
    version = int(recorded["artifact_version"])
    if version not in REPRODUCIBLE_ARTIFACT_VERSIONS:
        # Also here, and not only in :func:`replay_artifact`. This function is
        # the one that would execute something, and a guard that lives only at
        # the caller is a guard the next caller does not have.
        raise ArtifactError(_not_reproducible(version))

    execution = recorded["agent_execution"]
    source = str(execution["outcome_source"])
    instance_id = str(recorded["operation_instance_id"])
    if source == SOURCE_DETERMINISTIC:
        return (
            _reproduce_episode(
                spec, scenario_id, agent_id, operation_instance_id=instance_id
            ),
            REPRODUCTION_RERUN,
            (),
        )

    tape = tape_from_records(recorded["decisions"])

    def build_playback_agent() -> Any:
        if source != SOURCE_MODEL:
            return RecordedOutcomeAgent(tape, agent_id=agent_id)
        # A model artefact's playback rebuilds each request from the provider
        # identity the record names and the observation the episode offers, and
        # requires it to hash to the attempt recorded beside that decision. It
        # is what makes ``model`` and ``max_output_tokens`` part of the run
        # rather than a label on it — and it reaches no provider: the rebuilt
        # agent holds a ForbiddenTransport.
        return RecordedModelAgent(
            tape,
            agent_id=agent_id,
            model=str(execution["model"]),
            max_output_tokens=int(execution["max_output_tokens"]),
            request_digests=[
                str(attempt["request_digest_sha256"]) for attempt in execution["attempts"]
            ],
        )

    run = _reproduce_episode(
        spec,
        scenario_id,
        agent_id,
        operation_instance_id=instance_id,
        agent_factory=build_playback_agent,
        agent_kind=derived_agent_kind(agent_id, source),
    )
    if source == SOURCE_RECORDED:
        # A record of a playback replays to a playback. The execution record is
        # re-derived — a playback's execution is "no provider, no attempts", and
        # the reproduction reaches that itself — so only the tape and the
        # identity it names are carried.
        return run, REPRODUCTION_PLAYBACK, _CARRIED_BY_PLAYBACK
    # A model run's execution record describes the provider session that
    # produced these decisions. A replay is required not to repeat that session,
    # so it cannot re-derive the record — it carries it, and says so. What keeps
    # that from being a blank cheque is the reader: :func:`_validate_provenance`
    # has already held the carried record to the tape and the identity it sits
    # beside, so the fields carried here are consistent with the run even though
    # this reproduction did not produce them.
    #
    # The provider execution binding travels with it, for exactly the same
    # reason and with exactly the same limit. A replay reaches no provider, so
    # it cannot re-derive which journal witnessed the original session; carrying
    # the binding is what lets the rebuilt document be compared field for field
    # with the record instead of diverging on a field no reproduction could
    # produce. **Nothing here opens the ledger.** The binding is copied as an
    # opaque block of already-validated values, and whether the journal it names
    # says what it says is
    # :func:`~operatebench.execution_bundle.audit_execution_bundle`'s question,
    # asked against the sidecar and never during a replay.
    return (
        replace(
            run,
            execution=execution_from_record(execution),
            provider_execution=recorded["provider_execution"],
        ),
        REPRODUCTION_PLAYBACK,
        tuple(sorted((*_CARRIED_BY_PLAYBACK, "agent_execution", "provider_execution"))),
    )


def replay_artifact(spec: OperationSpec, artifact: Mapping[str, Any]) -> ReplayReport:
    """Re-execute what an artefact records and compare the whole result.

    The rerun is turned back into the artefact it would have written, and that
    reconstruction is what the record is compared against. Nothing is compared
    by summary and nothing is left out: a field that is not in a section is a
    field the artefact does not have.

    How the agent's decisions are obtained depends on what the record says
    produced them. A v1 artefact, and a v2 one whose decisions came from an
    in-process agent, re-execute that agent — the check it has always been. A
    v2 artefact whose decisions came from a model replays them from its own
    tape, because re-executing the agent would be a second provider run against
    an agent that need not agree with itself. Either way the *environment* is
    re-executed in full: the clock, the events, the domain validators, the
    ledger and the evaluator all run again.
    """
    recorded = validate_artifact(artifact)
    version = int(recorded["artifact_version"])
    if version not in REPRODUCIBLE_ARTIFACT_VERSIONS:
        # Asked before the operation binding, because it is the wider statement:
        # this build cannot reproduce a record of that contract against *any*
        # spec, so which operation the caller passed does not come into it.
        raise ArtifactError(_not_reproducible(version))
    if recorded["engine_version"] != OPERATEBENCH_VERSION:
        raise ArtifactError(
            f"this artefact records engine version {recorded['engine_version']!r}, "
            f"but this runtime is {OPERATEBENCH_VERSION!r}; replay requires the "
            "original engine contract, not execution under a different runtime"
        )
    recorded_digest = recorded["operation"]["spec_digest_sha256"]
    if recorded_digest != spec.spec_digest_sha256:
        raise ArtifactError(
            f"this artefact was produced from spec digest {recorded_digest!r}, but "
            f"{spec.source} has digest {spec.spec_digest_sha256!r}; a replay against "
            "a different operation would compare two different runs"
        )
    scenario_id = str(recorded["scenario_id"])
    agent_id = str(recorded["agent_id"])
    try:
        run, reproduction, carried = _reproduce(spec, recorded, scenario_id, agent_id)
    except OperateBenchError as exc:
        raise ArtifactError(f"this artefact cannot be replayed: {exc}") from exc
    # No projection onto an older contract happens here any more, and the
    # absence is the contract: :func:`_reproduce` refuses a record this build
    # cannot reproduce before anything is executed, so everything that reaches
    # this point is a current-contract record compared against a
    # current-contract reconstruction, field for field, with nothing dropped to
    # make the two agree.
    expected = build_artifact(run)
    _require_retrieval_provenance(recorded, expected)

    sections: dict[str, dict[str, bool]] = {}
    differences: list[str] = []
    for section, field_names in _SECTIONS_BY_VERSION[version].items():
        verdicts: dict[str, bool] = {}
        for name in field_names:
            matched = recorded[name] == expected[name]
            verdicts[name] = matched
            if not matched:
                differences.append(
                    f"{section}: {name} differs — recorded "
                    f"{_summarise(recorded[name])}, replayed "
                    f"{_summarise(expected[name])}"
                )
        sections[section] = verdicts

    # Self-consistency is not checked again here: :func:`validate_artifact` above
    # recomputes both content digests and refuses a record whose payload and
    # digest disagree, so anything that reaches this point already describes
    # itself. What is compared below is a different question — whether the run it
    # describes is the run that happens when it is executed again.
    report = ReplayReport(
        scenario_id=scenario_id,
        agent_id=agent_id,
        sections=sections,
        recorded_final_state_digest=str(recorded["final_state_digest_sha256"]),
        replayed_final_state_digest=run.outcome.final_state_digest_sha256,
        recorded_trajectory_digest=str(recorded["trajectory_digest_sha256"]),
        replayed_trajectory_digest=run.outcome.trajectory_digest_sha256,
        differences=tuple(differences),
        artifact_version=version,
        reproduction=reproduction,
        carried_forward=carried,
    )
    # This is the independently executed Engine result, never reconstructed from
    # recorded fields. Store it only when the report already passed: a failed
    # report cannot be turned into authority by mutating a nested verdict mapping.
    if report.ok:
        object.__setattr__(report, "_replayed_outcome", run.outcome)
    return report


#: What a replay requires a re-served read to reproduce exactly: *which record,
#: at which version, from which source, under whose authority, at which instant —
#: and saying what*. The body is in the comparison at contract 5 because the row
#: now carries it. A version equal to a re-served version over a body nobody
#: compared would be a digest agreeing with itself, and a records body edited
#: with its version recomputed is exactly the forgery a grammar reading one row
#: cannot see: it is internally consistent, and only the live operation
#: disagrees with it.
_RETRIEVAL_PROVENANCE_FIELDS: tuple[str, ...] = (
    "invocation_index",
    "turn_index",
    "batch_index",
    "request_index",
    "initiated_by",
    "tool",
    "source",
    "authority",
    "record_id",
    "record_version",
    "as_of",
    "schema_id",
    "records",
    "ok",
)


def _retrieval_provenance(
    payload: Mapping[str, Any],
) -> list[tuple[Any, ...]]:
    return [
        tuple(row.get(name) for name in _RETRIEVAL_PROVENANCE_FIELDS)
        for row in payload["trajectory"]
        if row.get("record_type") == "retrieval_served"
    ]


def _require_retrieval_provenance(
    recorded: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    """Refuse a replay whose re-served reads are not the reads that were recorded.

    Playback returns the recorded ``RETRIEVE`` batch and the **live domain**
    serves it again — never the artefact — so the versions come back from the
    operation rather than from the document being checked. Requiring them to be
    equal is what makes a recorded read a claim the record cannot make about
    itself: edit one ``record_version`` and the replay refuses by name instead of
    listing it among the differences, where it would read as one disagreement
    among many rather than as a broken binding.
    """
    was = _retrieval_provenance(recorded)
    now = _retrieval_provenance(expected)
    if was == now:
        return
    if len(was) != len(now):
        raise RetrievalProvenanceMismatchError(
            f"this record carries {len(was)} served retrieval row(s) and re-serving "
            f"its batches against the live operation produces {len(now)}; a record "
            "that read a different number of times describes a different run"
        )
    for position, (recorded_row, replayed_row) in enumerate(zip(was, now, strict=True)):
        if recorded_row == replayed_row:
            continue
        differing = sorted(
            name
            for name, left, right in zip(
                _RETRIEVAL_PROVENANCE_FIELDS,
                recorded_row,
                replayed_row,
                strict=True,
            )
            if left != right
        )
        raise RetrievalProvenanceMismatchError(
            f"served retrieval row {position} differs in {differing} between this "
            "record and the same read re-served from the live operation; a "
            "retrieved record's provenance is what a decision rested on, and a "
            "replay that accepted a different version would be checking the "
            "decision against evidence it never saw"
        )


def replay_file(spec: OperationSpec, path: str | Path) -> ReplayReport:
    """Read an artefact and replay it. The CLI's whole ``replay`` command."""
    return replay_artifact(spec, read_artifact(path))


def _summarise(value: Any) -> str:
    """A short, safe rendering of a field for a difference line."""
    if isinstance(value, Mapping):
        return f"a record of {len(value)} field(s)"
    if isinstance(value, list):
        return f"{len(value)} entr(y/ies)"
    return repr(value)


def _content_digest(payload: Any) -> str:
    return hashlib.sha256(
        canonical_json_text(payload, "operatebench artefact section").encode("utf-8")
    ).hexdigest()


__all__ = [
    "ARTIFACT_NOTE",
    "ARTIFACT_VERSION",
    "ARTIFACT_VERSION_V1",
    "ARTIFACT_VERSION_V2",
    "ARTIFACT_VERSION_V4",
    "ARTIFACT_VERSION_V5",
    "ARTIFACT_VERSION_V6",
    "ARTIFACT_VERSION_V7",
    "BINDABLE_PROVIDERS",
    "BINDABLE_PROVIDER_APIS",
    "BINDABLE_PROVIDER_API_PAIRS",
    "DETERMINISTIC_AGENT_KINDS",
    "KIND_MODEL",
    "KIND_RECORDED",
    "PROVIDER_EXECUTION_FIELDS",
    "PROVIDER_EXECUTION_NOTE",
    "REPRODUCIBLE_ARTIFACT_VERSIONS",
    "REPRODUCIBLE_MODEL_PROTOCOLS",
    "REPRODUCTION_PLAYBACK",
    "REPRODUCTION_RERUN",
    "SUPPORTED_ARTIFACT_VERSIONS",
    "SUPPORTED_MODEL_PROTOCOLS",
    "ReplayReport",
    "RetrievalProvenanceMismatchError",
    "artifact_text",
    "build_artifact",
    "content_digest",
    "derived_agent_kind",
    "evaluation_dimension_coverage_problem",
    "evaluation_dimensions_for",
    "outcome_shape_coverage_problem",
    "project_top_level",
    "project_trajectory",
    "read_artifact",
    "replay_artifact",
    "replay_file",
    "row_shape_coverage_problem",
    "section_coverage_problem",
    "validate_artifact",
    "write_artifact",
]
