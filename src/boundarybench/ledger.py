"""The application-level append-only episode ledger.

One completed or terminally failed episode is one line of canonical JSON, and a
line this build has written is never rewritten by it. That is what makes a run
resumable without re-executing work and checkable without trusting the process
that wrote it: the file is the record, and everything needed to bind a row back
to its run, its suite, its scaffold and its variant travels in the row.

Reading is deliberately unforgiving. A malformed row, a truncated tail, a
duplicate, a row for an episode this run never planned, a row whose
classification does not follow from its own recorded cause, or one whose
provenance disagrees with the manifest all stop the resume. None of them is
repaired or skipped: each one means the evidence on disk is not what it claims,
and quietly continuing would produce a run that looks complete and is not.

What that is *not* is tamper-proofing, and the distinction is stated in
:data:`LEDGER_INTEGRITY_NOTE` and carried in every run report rather than left
to a reader's charity. Append-only here is a property of this application, not
of the medium: the file is unsigned and sits in a writable directory, so an
editor who rewrites it into a self-consistent whole produces something this
build accepts. Detecting that needs a commitment kept somewhere this build
cannot write, and no such external commitment is shipped.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any

from boundarybench.adapter import (
    AdapterError,
    TurnRequest,
    build_turn_request,
)
from boundarybench.budget import (
    CostControls,
    usd_text,
)
from boundarybench.compiler import Variant
from boundarybench.environment import (
    EnvironmentError,
    replay_trajectory,
    replay_turn_states,
)
from boundarybench.evaluator import evaluate
from boundarybench.freezing import deep_freeze, to_json
from boundarybench.jsonsafe import (
    JsonSafetyError,
    NestingDepthError,
    canonical_json_text,
    ensure_json_safe,
    ensure_raw_json_depth,
    sanitized_text,
)
from boundarybench.pricing import ModelPrice
from boundarybench.queries import QUERY_ACTIONS, QUERY_RESOLUTION_CONTRACT
from boundarybench.runmanifest import (
    RunManifest,
    RunManifestError,
    canonical_json,
    check_utc_timestamp,
)
from boundarybench.scaffold import (
    ACTION_SURFACE_CONTRACT,
    FACT_PARAMETER,
    STANDARD_SCAFFOLD,
    Scaffold,
    ScaffoldError,
    load_scaffold,
)
from boundarybench.trajectory import Step, TerminalDecision, Trajectory

#: The ledger row format this build reads and writes.
#:
#: ``2`` split ``run_id`` into ``configuration_id`` and ``execution_id``, added
#: the episode's own wall-clock window and locally measured duration, added
#: structured provider-attempt evidence and its measurement coverage, and made
#: every row state whether it is a completed semantic sample. Version-1 rows do
#: not carry those facts and are refused rather than read with blank answers.
#:
#: ``3`` changed what two of a row's fields mean. ``outcome`` gained
#: :data:`OUTCOME_PROVIDER_CONFIGURATION_FAILURE`, which a version-2 reader
#: would refuse; and ``completed_at_utc`` became the recorded start plus the
#: measured monotonic duration rather than a second wall-clock reading, so the
#: same field is now a derived instant rather than an observed one. A row is
#: refused rather than reinterpreted across that line, in either direction.
#:
#: ``5`` made a capped run's money re-derivable from the row rather than read
#: back from it. Every recorded provider attempt now states the reservation that
#: authorised it and how that reservation was resolved, so an episode's
#: ``cost_exposure_usd`` is checked against the attempts it is the sum of; a
#: row's measured ``usage.cost_usd`` is recomputed from its own token counts at
#: the manifest's pinned pricing policy rather than trusted; and ``outcome``
#: gained :data:`OUTCOME_COST_RESERVATION_BREACHED`. A version-4 row states none
#: of the fields the derivation needs, so it cannot be checked and is refused by
#: version rather than read with the checks skipped. A zeroed cost or exposure
#: must be contradicted by independently derived values.
#:
#: ``6`` changed what ``cost_reservation_usd`` has to be, without changing which
#: fields a row states. Under ``5`` it was checked against a range — the cost of
#: a bodiless request at the floor, the run's cap at the ceiling — and the row's
#: exposure was summed from the same field, so an editor who lowered every
#: attempt of a turn to the floor and adjusted the total to match produced a row
#: that passed both checks and a resume that forgot the difference. Under ``6``
#: each attempt's reservation is *rebuilt*: the request that turn sent is a
#: projection of the manifest, the scaffold it pins by digest, the compiled
#: variant and the replayed trajectory, and its conservative input bound at the
#: run's pinned output ceiling and pricing policy is the only amount the row may
#: state. A version-5 capped row is a claim only a range ever checked, so it is
#: refused by version rather than re-judged under a rule its writer was not held
#: to; and a version-6 row cannot be read by a version-5 reader that would
#: accept the forgery. See :class:`_RequestProjection`.
#:
#: ``7`` added the one field that makes a turn's attempts a *sequence* rather
#: than a bag, and with it three rules ``6`` had no way to state. Under ``6`` the
#: reader checked each attempt's index, each summary's arithmetic and each
#: reservation's derivation, and nothing about how many attempts a turn was
#: permitted, what order they may occur in, or how many turns of an episode
#: reached the provider at all. So a genuine three-attempt turn could be
#: shortened to two, a fourth attempt appended beyond the manifest's pinned
#: retry policy, a ``[fault, fault, response]`` turn reordered to
#: ``[response, fault, fault]``, or a whole provider turn deleted — each with
#: every count, tally, coverage summary and exposure total adjusted to agree,
#: and each accepted, with the run's own exposure reduced to match. Under ``7``
#: every recorded turn also states ``terminal_reason``: why its retry loop
#: stopped, from schema 7's locally owned closed terminal-reason vocabulary. A
#: ``6`` row states no reason, so what its attempts were permitted to be cannot
#: it, and it is refused by version rather than read under rules its writer was
#: never held to. See :func:`_check_turn_sequence`.
LEDGER_SCHEMA_VERSION = 7

# Schema 7 owns every closed provider-attempt and retry value that its reader
# interprets. The unsuffixed names below are local conveniences only; none is
# imported from the mutable adapter/provider kernel.
ATTEMPT_OUTCOME_FAULT = "fault"
ATTEMPT_OUTCOME_RESPONSE = "response"
ATTEMPT_OUTCOME_UNCLASSIFIED = "unclassified_error"
ATTEMPT_OUTCOMES_V7: tuple[str, ...] = (
    "fault",
    "response",
    "unclassified_error",
)
ATTEMPT_OUTCOMES = ATTEMPT_OUTCOMES_V7
ATTEMPT_SETTLEMENT_FORFEITED = "forfeited"
ATTEMPT_SETTLEMENT_MEASURED = "measured"
ATTEMPT_SETTLEMENTS_V7: tuple[str, ...] = (
    "forfeited",
    "measured",
)
ATTEMPT_SETTLEMENTS = ATTEMPT_SETTLEMENTS_V7
TURN_END_BACKOFF_UNAFFORDABLE = "backoff_unaffordable"
TURN_END_COST_CAP_EXHAUSTED = "cost_cap_exhausted"
TURN_END_DEADLINE_EXCEEDED = "deadline_exceeded"
TURN_END_FAULT_NOT_RETRYABLE = "fault_not_retryable"
TURN_END_PRE_DISPATCH_REFUSED = "pre_dispatch_refused"
TURN_END_RESPONSE = "response"
TURN_END_RETRIES_EXHAUSTED = "retries_exhausted"
TURN_END_UNCLASSIFIED = "unclassified_error"
TURN_TERMINAL_REASONS_V7: tuple[str, ...] = (
    "backoff_unaffordable",
    "cost_cap_exhausted",
    "deadline_exceeded",
    "fault_not_retryable",
    "pre_dispatch_refused",
    "response",
    "retries_exhausted",
    "unclassified_error",
)
TURN_TERMINAL_REASONS = TURN_TERMINAL_REASONS_V7
TURN_END_EARLY_REASONS_V7: tuple[str, ...] = (
    "backoff_unaffordable",
    "cost_cap_exhausted",
    "deadline_exceeded",
    "pre_dispatch_refused",
)
TURN_END_EARLY_REASONS = TURN_END_EARLY_REASONS_V7

#: The exact fault vocabulary owned by schema 7. Literals rather than an alias
#: to the live Lifecycle provider taxonomy keep later provider additions from
#: silently changing what this frozen historical journal accepts.
PROVIDER_FAULTS_V7: tuple[str, ...] = (
    "provider_authentication",
    "provider_configuration",
    "provider_network_error",
    "provider_rate_limited",
    "provider_request_rejected",
    "provider_response_invalid",
    "provider_server_error",
    "provider_timeout",
)

#: The exact retryable subset owned by schema 7. This is frozen beside the
#: vocabulary rather than imported from the provider kernel for the same reason:
#: historical row admissibility cannot move when the live taxonomy does.
RETRYABLE_PROVIDER_FAULTS_V7: tuple[str, ...] = (
    "provider_network_error",
    "provider_rate_limited",
    "provider_server_error",
    "provider_timeout",
)

#: What this ledger is, and — as importantly — what it is not. Stated in the
#: run report so the limitation travels with the evidence rather than living in
#: a document somebody has to remember to read.
LEDGER_INTEGRITY_NOTE = (
    "This ledger is application-level append-only/write-once: rows are appended "
    "under an exclusive lock, fsynced before an episode counts as done, never "
    "rewritten by this build, and validated on every read by replaying each "
    "trajectory through the real environment and regrading it. That is an "
    "internal-consistency guarantee, not a cryptographic one. The file is "
    "unsigned and lives in a writable directory, so an editor who replaces the "
    "whole file with self-consistent rows — or who edits one row and recomputes "
    "everything bound to it — produces a ledger this build accepts. Malicious "
    "whole-file replacement is not independently detectable locally. External "
    "immutable storage, or a signed and published commitment to the run's "
    "digests, is required before any release-grade immutability claim."
)

#: One turn is one request out and one response back. Defined here, not in the
#: runner, because the ledger is what enforces the relation between the two;
#: the runner imports it back so the two can never drift apart.
MESSAGES_PER_TURN = 2

#: Every way an episode can end, and the only values ``outcome`` may take.
#:
#: ``success`` means the loop ran to a terminal decision through the real
#: environment. It says nothing about whether the decision was *correct* — that
#: is the evaluator's verdict, recorded separately in the same row.
OUTCOME_SUCCESS = "success"
#: The model said something the scaffold does not define: a malformed response,
#: an unknown action, arguments that do not match the schema it was shown.
OUTCOME_MODEL_PROTOCOL_FAILURE = "model_protocol_failure"
#: The model said something schema-valid that it was not admissible to say: a
#: fact requested through a channel this case does not deliver it on, a workflow
#: action the case does not permit, an action taken out of order. The call
#: parsed; the *choice* was wrong. This is the model's doing, which is why it is
#: not an environment failure — the environment worked exactly as specified when
#: it refused.
OUTCOME_MODEL_ACTION_FAILURE = "model_action_failure"
OUTCOME_SCAFFOLD_FAILURE = "scaffold_failure"
#: Reserved for the environment machinery itself breaking — the case fixture
#: raising something that is not an :class:`EnvironmentError`. A refusal *by*
#: the environment is the environment working, and is classified as the model
#: action failure it is.
OUTCOME_ENVIRONMENT_FAILURE = "environment_failure"
OUTCOME_EVALUATOR_FAILURE = "evaluator_failure"
#: The integration itself broke: it raised, or reported a measurement that is
#: not one. Distinct from every ``provider_*`` outcome below, which are the
#: *service* failing rather than the code that calls it.
OUTCOME_ADAPTER_FAILURE = "adapter_failure"
#: The credential was refused, or is not permitted to run this model. An
#: operator action, and the one provider fault that is never worth retrying.
OUTCOME_PROVIDER_AUTH_FAILURE = "provider_auth_failure"
#: The provider says this run's pinned model or resource does not exist, or is
#: not available to this account. An operator action, never retried, and — with
#: a refused credential — one of the two failures that condemn the whole run:
#: the model id is fixed by configuration identity and is sent on every turn, so
#: the answer cannot come out differently for a later episode.
OUTCOME_PROVIDER_CONFIGURATION_FAILURE = "provider_configuration_failure"
#: The provider rejected the request itself — a payload it will not accept, a
#: body too large, a field it refuses. Never retried, because an invalid request
#: is invalid the second time too, and deliberately *not* run-terminal: it is a
#: fact about the request this episode happened to send, and the next episode
#: sends a different one.
OUTCOME_PROVIDER_REQUEST_FAILURE = "provider_request_failure"
#: Rate limited, and still rate limited when the adapter's bounded retries ran
#: out. Its own outcome because the remedy — slow down, or raise the limit — is
#: not the remedy for anything else here.
OUTCOME_PROVIDER_RATE_LIMITED = "provider_rate_limited"
#: The service or the path to it failed: a 5xx, a dropped connection, a request
#: timeout, an answer that does not parse. Nothing an operator can fix in the
#: run, which is why these four share one outcome — and why ``error_class``
#: keeps naming which of them it was.
OUTCOME_PROVIDER_FAILURE = "provider_failure"
#: The model's answer was cut off by the run's own pinned output-token ceiling.
#: Its own outcome, and deliberately not ``model_protocol_failure``: the model
#: was still speaking when a limit *this run chose* stopped it, so blaming the
#: model for it would both misattribute the fault and bury the one thing an
#: operator could change. Also not ``limit_exhausted``, which is the episode's
#: turn and message budget — a different budget, spent by a different actor.
OUTCOME_MODEL_OUTPUT_LIMIT = "model_output_limit"
OUTCOME_LIMIT_EXHAUSTED = "limit_exhausted"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_RUNNER_FAILURE = "runner_failure"
#: The run's authorised cost budget could not cover the next provider request,
#: so the request was not made. Not a provider fault and not a model failure:
#: nothing was asked of anyone. Its own outcome because the remedy — authorise
#: more, or accept a shorter run — is not the remedy for anything else here, and
#: because it is the one row in a ledger that records a request that *did not*
#: happen.
OUTCOME_COST_CAP_REACHED = "cost_cap_reached"
#: A provider response measured more than the reservation that authorised the
#: request that produced it. The mirror image of the outcome above: there the
#: request was refused before it was sent, and here it was sent, answered and
#: measured beyond what this run's own arithmetic had authorised. Its own
#: outcome because it is the one row that records the guarantee being *reached*
#: rather than held — the spend is real, it is recorded at its measured value,
#: and the run stops rather than continuing under a cap it has already passed.
OUTCOME_COST_RESERVATION_BREACHED = "cost_reservation_breached"

OUTCOMES: tuple[str, ...] = (
    OUTCOME_SUCCESS,
    OUTCOME_MODEL_PROTOCOL_FAILURE,
    OUTCOME_MODEL_ACTION_FAILURE,
    OUTCOME_MODEL_OUTPUT_LIMIT,
    OUTCOME_SCAFFOLD_FAILURE,
    OUTCOME_ENVIRONMENT_FAILURE,
    OUTCOME_EVALUATOR_FAILURE,
    OUTCOME_ADAPTER_FAILURE,
    OUTCOME_PROVIDER_AUTH_FAILURE,
    OUTCOME_PROVIDER_CONFIGURATION_FAILURE,
    OUTCOME_PROVIDER_REQUEST_FAILURE,
    OUTCOME_PROVIDER_RATE_LIMITED,
    OUTCOME_PROVIDER_FAILURE,
    OUTCOME_LIMIT_EXHAUSTED,
    OUTCOME_TIMEOUT,
    OUTCOME_RUNNER_FAILURE,
    OUTCOME_COST_CAP_REACHED,
    OUTCOME_COST_RESERVATION_BREACHED,
)

# -- what a row is a sample *of* ---------------------------------------------
#
# A ledger row can record a refused credential, an incomplete model attempt or a
# graded decision. A raw row count therefore mixes distinct evidence classes.
# An episode that never reached the model is not a sample of model behaviour,
# and an episode that reached it without finishing is not a scorable sample.
#
# So every row states which of three things it is, the rule is a closed function
# of the outcome, and the rule is re-derived when the row is read.

#: The loop reached a terminal decision through the real environment and the
#: evaluator graded it. This is the only status that is a completed semantic
#: sample, and the only one a score could ever be computed over.
SAMPLE_COMPLETED = "completed_semantic_sample"

#: The model acted, and what it did ended the episode short of a decision: it
#: answered off-protocol, took an action the case does not admit, spent its turn
#: or message budget, or ran into the pinned output ceiling. These are real
#: observations of model behaviour and they are *not* completed samples — there
#: is no terminal decision, so there is no verdict and nothing to score. They
#: are counted and reported separately, and never folded into either the
#: completed count or the infrastructure count.
SAMPLE_BEHAVIORAL_ATTEMPT = "behavioral_attempt_not_completed"

#: The infrastructure failed: the provider, the network, the credential, the
#: request, this build's adapter, its scaffold contract, the environment
#: machinery, the evaluator, the runner, or the episode's wall-clock budget.
#: Nothing about the model can be read from these, so they are excluded from
#: every semantic count. They are still preserved, in full, as attempts: the
#: evidence that a run met an outage is evidence.
SAMPLE_INFRASTRUCTURE_INELIGIBLE = "infrastructure_failure_ineligible"

SAMPLE_STATUSES: tuple[str, ...] = (
    SAMPLE_BEHAVIORAL_ATTEMPT,
    SAMPLE_COMPLETED,
    SAMPLE_INFRASTRUCTURE_INELIGIBLE,
)

#: The closed rule, as a total function of the outcome. Total on purpose: a new
#: outcome added without a decision about what it is a sample of would otherwise
#: default to something, and the safe-looking default — "count it" — is the one
#: that inflates an n.
SAMPLE_STATUS_BY_OUTCOME: Mapping[str, str] = MappingProxyType(
    {
        OUTCOME_SUCCESS: SAMPLE_COMPLETED,
        # Model-originated, and short of a decision.
        OUTCOME_MODEL_PROTOCOL_FAILURE: SAMPLE_BEHAVIORAL_ATTEMPT,
        OUTCOME_MODEL_ACTION_FAILURE: SAMPLE_BEHAVIORAL_ATTEMPT,
        OUTCOME_MODEL_OUTPUT_LIMIT: SAMPLE_BEHAVIORAL_ATTEMPT,
        OUTCOME_LIMIT_EXHAUSTED: SAMPLE_BEHAVIORAL_ATTEMPT,
        # Infrastructure. ``timeout`` is here rather than with the model's own
        # budget failures because the episode's wall-clock is dominated by
        # provider latency, which is not the model's doing and not something
        # this build can separate out from it.
        OUTCOME_TIMEOUT: SAMPLE_INFRASTRUCTURE_INELIGIBLE,
        OUTCOME_SCAFFOLD_FAILURE: SAMPLE_INFRASTRUCTURE_INELIGIBLE,
        OUTCOME_ENVIRONMENT_FAILURE: SAMPLE_INFRASTRUCTURE_INELIGIBLE,
        OUTCOME_EVALUATOR_FAILURE: SAMPLE_INFRASTRUCTURE_INELIGIBLE,
        OUTCOME_ADAPTER_FAILURE: SAMPLE_INFRASTRUCTURE_INELIGIBLE,
        OUTCOME_PROVIDER_AUTH_FAILURE: SAMPLE_INFRASTRUCTURE_INELIGIBLE,
        OUTCOME_PROVIDER_CONFIGURATION_FAILURE: SAMPLE_INFRASTRUCTURE_INELIGIBLE,
        OUTCOME_PROVIDER_REQUEST_FAILURE: SAMPLE_INFRASTRUCTURE_INELIGIBLE,
        OUTCOME_PROVIDER_RATE_LIMITED: SAMPLE_INFRASTRUCTURE_INELIGIBLE,
        OUTCOME_PROVIDER_FAILURE: SAMPLE_INFRASTRUCTURE_INELIGIBLE,
        OUTCOME_RUNNER_FAILURE: SAMPLE_INFRASTRUCTURE_INELIGIBLE,
        # Not a failure of anything, and still not a sample: the plan slot was
        # spent on a request this run was not authorised to make, so nothing
        # about the model can be read from it. It is grouped with the
        # infrastructure-ineligible rows because the only property that matters
        # downstream is the one they share — never a denominator.
        OUTCOME_COST_CAP_REACHED: SAMPLE_INFRASTRUCTURE_INELIGIBLE,
        # A real provider turn happened here, and it is still not a sample: the
        # episode ended on this build's own accounting rather than on anything
        # the model did, and the turn it ended on never reached a decision.
        OUTCOME_COST_RESERVATION_BREACHED: SAMPLE_INFRASTRUCTURE_INELIGIBLE,
    }
)

#: Said in every report, because "12 planned, 12 completed" is the sentence a
#: reader most wants to be simple and most often is not.
SAMPLE_ELIGIBILITY_NOTE = (
    "Every row states what it is a sample of. Only 'completed_semantic_sample' "
    "rows reached a terminal decision and were graded; only they could ever be a "
    "denominator. 'behavioral_attempt_not_completed' rows are real observations "
    "of model behaviour that ended short of a decision — off-protocol answers, "
    "inadmissible actions, an exhausted turn or message budget, an answer cut "
    "off at the pinned output ceiling — and they are reported separately and "
    "never counted as completed samples. "
    "'infrastructure_failure_ineligible' rows are the provider, the network, the "
    "credential, the request, the adapter, the scaffold, the environment, the "
    "evaluator, the runner or the episode timeout failing; nothing about the "
    "model can be read from them and they are excluded from every semantic "
    "count. Ineligible rows are durable and are never silently re-run: a plan "
    "slot they consumed stays consumed. Scheduling an explicit replacement "
    "sample for one is not implemented in this build, so a plan whose slots were "
    "spent on infrastructure failures yields fewer completed samples than it "
    "planned, and the counts above say so rather than hiding it."
)


def sample_status_for(outcome: str) -> str:
    """What a row with this outcome is a sample of. Total over :data:`OUTCOMES`."""
    return SAMPLE_STATUS_BY_OUTCOME[outcome]


#: The outcomes that condemn the whole run rather than one episode.
#:
#: Exactly two, and the test each has to pass is *proof*, not plausibility: the
#: failing input must be something every remaining episode would send
#: identically. A refused credential qualifies — one credential is resolved once
#: for the run — and so does a provider that cannot find the pinned model or
#: resource, because the model id is fixed by configuration identity and goes
#: out on every turn. For those two the thirteenth attempt fails exactly as the
#: first did, so the first is recorded durably and the run stops rather than
#: spending the plan and eleven more round-trips to learn nothing. Because
#: whatever an operator changes to fix them (a credential, an account's model
#: access) is by design *not* part of configuration identity, the same directory
#: can never continue afterwards; see
#: :class:`~boundarybench.runner.RunTerminatedError`.
#:
#: Everything else the provider refuses stays one episode's failure. A rejected
#: *request* — an oversized body, a payload field the API will not take — is a
#: fact about what this episode sent; the next episode sends a different
#: trajectory-dependent payload, and stopping the plan on one would destroy
#: eleven samples to record a fault that had not been shown to apply to them.
#: Those rows are durable, infrastructure-ineligible and never silently re-run,
#: which is the honest treatment of a spent plan slot.
#: A spent cost budget is the third, and it meets the same test: what is left of
#: an authorised budget does not grow, and every remaining episode would be
#: refused by the same arithmetic. Unlike the other two it is not fixed by an
#: operator action outside configuration identity — raising the cap *is* a change
#: to configuration identity — so a new authorisation is a new run by
#: construction, and this directory keeps its evidence.
RUN_TERMINAL_OUTCOMES: frozenset[str] = frozenset(
    {
        OUTCOME_PROVIDER_AUTH_FAILURE,
        OUTCOME_PROVIDER_CONFIGURATION_FAILURE,
        OUTCOME_COST_CAP_REACHED,
        OUTCOME_COST_RESERVATION_BREACHED,
    }
)

#: Measurement coverage, as a closed three-valued answer. ``partial`` exists
#: because a single boolean "measured" flag on an episode hid exactly this: an
#: episode where one turn of four reported a measurement read as measured.
MEASUREMENT_FULL = "full"
MEASUREMENT_PARTIAL = "partial"
MEASUREMENT_UNMEASURED = "unmeasured"
MEASUREMENT_STATUSES: tuple[str, ...] = (
    MEASUREMENT_FULL,
    MEASUREMENT_PARTIAL,
    MEASUREMENT_UNMEASURED,
)

_ROW_KEYS: tuple[str, ...] = (
    "schema_version",
    "configuration_id",
    "execution_id",
    "episode_id",
    "variant_id",
    "trial_index",
    "outcome",
    "sample_status",
    "error_class",
    "error_detail",
    "failure_event",
    "turns_used",
    "messages_used",
    "started_at_utc",
    "completed_at_utc",
    "elapsed_seconds",
    "trajectory",
    "evaluation",
    "usage",
    "provider_telemetry",
    "measurement",
    "cost_exposure_usd",
    "suite_content_digest",
    "scaffold_content_digest",
    "variant_content_digest",
    "adapter",
    "contract_versions",
)
_USAGE_KEYS = ("input_tokens", "output_tokens", "cost_usd", "latency_seconds")
_TELEMETRY_KEYS = ("turns", "attempt_count", "fault_counts", "measurement_status")
_TELEMETRY_TURN_KEYS = (
    "turn_index",
    "attempts",
    "attempt_count",
    "fault_counts",
    "response_received",
    "usage_reported",
    "turn_latency_seconds",
    "terminal_reason",
)
_ATTEMPT_KEYS = (
    "index",
    "outcome",
    "fault",
    "http_status",
    "latency_seconds",
    "response_received",
    "usage_reported",
    "cost_reservation_usd",
    "cost_settlement",
)
_MEASUREMENT_KEYS = (
    "turns_total",
    "turns_measured",
    "attempts_total",
    "attempts_measured",
    "status",
)
_ADAPTER_KEYS = ("provider", "model", "implementation", "version")
_CONTRACT_KEYS = ("package", "runner", "evaluator", "adapter")
_FAILURE_EVENT_KEYS = (
    "phase",
    "kind",
    "attempted_call",
    "response_type",
    "exception_type",
    "detail",
)
_ATTEMPTED_CALL_KEYS = ("action", "arguments")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


# -- the failure taxonomy ----------------------------------------------------
#
# ``outcome``, ``error_class`` and ``error_detail`` are a *summary* of why an
# episode ended. They are bound to the recorded event so a genuine
# ``model_protocol_failure`` cannot
# be rewritten to an ``environment_failure`` with an invented class and detail
# and still satisfy every check, because there was nothing left for them to
# disagree with. A failed row therefore carries a typed ``failure_event`` naming
# the phase the episode died in, the kind of fault, and the evidence that kind
# is defined by — and the whole summary, including the counters, is derived from
# it rather than stored alongside it.
#
# This is deterministic internal consistency, not tamper-proofing. The ledger is
# unsigned and lives in a writable directory: an editor who rewrites the event
# *and* every field bound to it still produces a file this build accepts. What
# is now impossible is a row whose classification does not follow from its own
# recorded cause.

#: Where in one episode a fault was classified.
PHASE_SCAFFOLD_CONTRACT = "scaffold_contract"
PHASE_TURN_START = "turn_start"
PHASE_ADAPTER_CALL = "adapter_call"
PHASE_USAGE = "usage"
PHASE_RESPONSE_VALIDATION = "response_validation"
PHASE_DISPATCH = "dispatch"
PHASE_EVALUATION = "evaluation"
PHASE_RUNNER = "runner"

#: What kind of fault it was. Every kind but the four ``*_exception`` ones names
#: its own ``error_class``; those four take the class from the exception raised.
KIND_INCOMPLETE_SCAFFOLD = "incomplete_scaffold"
KIND_EPISODE_TIMEOUT = "episode_timeout"
KIND_MAX_TURNS = "max_turns"
KIND_MAX_MESSAGES = "max_messages"
KIND_PROTOCOL_ERROR = "protocol_error"
#: The provider's answer was truncated by the run's pinned output ceiling. Its
#: own kind because it is its own cause: the answer parsed as far as it went and
#: stopped, which is not the same fault as an answer that was never well formed.
KIND_OUTPUT_TOKEN_LIMIT = "output_token_limit"
KIND_ADAPTER_EXCEPTION = "adapter_exception"
KIND_USAGE_EXCEPTION = "usage_exception"
KIND_MALFORMED_USAGE = "malformed_usage"
KIND_MALFORMED_RESPONSE = "malformed_response"
KIND_UNKNOWN_ACTION = "unknown_action"
KIND_MALFORMED_ARGUMENTS = "malformed_arguments"
#: The adapter's answer carried text canonical UTF-8 JSON cannot represent. It
#: is classified rather than sanitised: what a model produced is evidence, and
#: silently rewriting evidence to make it storable is worse than refusing it.
KIND_UNREPRESENTABLE_TEXT = "unrepresentable_text"
KIND_WRONG_CHANNEL = "wrong_channel"
KIND_ACTION_REJECTED = "action_rejected"
#: The environment raised something that is not an ``EnvironmentError`` — the
#: fixture itself is broken, not the model's choice.
KIND_ENVIRONMENT_EXCEPTION = "environment_exception"
KIND_EVALUATOR_EXCEPTION = "evaluator_exception"
KIND_RUNNER_EXCEPTION = "runner_exception"
#: The cost guard refused to authorise the request this turn was about to make.
KIND_COST_CAP_EXHAUSTED = "cost_cap_exhausted"
#: A response measured beyond the reservation that authorised its request. Not a
#: provider fault: the service answered and the counts it reported are the ones
#: this build priced. What failed is this build's bound on them.
KIND_COST_RESERVATION_BREACHED = "cost_reservation_breached"

#: Schema-7 provider failure kinds are durable literals, not aliases to the
#: adapter's current exception taxonomy.
KIND_PROVIDER_AUTHENTICATION = "provider_authentication"
KIND_PROVIDER_CONFIGURATION = "provider_configuration"
KIND_PROVIDER_REQUEST_REJECTED = "provider_request_rejected"
KIND_PROVIDER_RATE_LIMITED = "provider_rate_limited"
KIND_PROVIDER_SERVER_ERROR = "provider_server_error"
KIND_PROVIDER_NETWORK_ERROR = "provider_network_error"
KIND_PROVIDER_TIMEOUT = "provider_timeout"
KIND_PROVIDER_RESPONSE_INVALID = "provider_response_invalid"

#: Which outcome schema 7 assigns each historical provider fault.
PROVIDER_OUTCOMES_V7: Mapping[str, str] = {
    "provider_authentication": "provider_auth_failure",
    "provider_configuration": "provider_configuration_failure",
    "provider_request_rejected": "provider_request_failure",
    "provider_rate_limited": "provider_rate_limited",
    "provider_server_error": "provider_failure",
    "provider_network_error": "provider_failure",
    "provider_timeout": "provider_failure",
    "provider_response_invalid": "provider_failure",
}


@dataclass(frozen=True)
class FailureSpec:
    """Everything one ``(phase, kind)`` pair fixes about the row that records it.

    Read as a rule rather than a description: given the pair, the outcome, the
    error class, which evidence fields must be present, and how the counters
    must add up are all determined. Anything else in the row is a disagreement.
    """

    outcome: str
    #: ``error_class`` is the exception's type name rather than the kind's name.
    class_from_exception: bool = False
    #: The call the loop refused, recorded because the fault *is* about the call.
    attempted_call: bool = False
    #: What the adapter returned instead of a call.
    response_type: bool = False
    exception_type: bool = False
    #: A request that was sent and never answered, so the message count is odd.
    unanswered_request: bool = False
    #: How far ``turns_used`` may run ahead of the trajectory's step count.
    turns_beyond_steps: tuple[int, ...] = (1,)
    #: Whether the trajectory this fault leaves behind reached a terminal
    #: decision. ``False`` for every fault the runner can only raise *before* a
    #: terminal dispatch, ``True`` for the two it can only raise after one, and
    #: ``None`` where the runner genuinely reaches it from either side.
    #:
    #: This is what stops a genuine success being relabelled. Rewriting a
    #: completed episode's outcome to ``adapter_failure`` and nulling its
    #: verdict must be refused by binding the *class* of failure to the *shape*
    #: of the episode. An adapter call
    #: that raised cannot have happened after the case was already closed: the
    #: loop breaks at the terminal dispatch and never asks the adapter again.
    terminal_trajectory: bool | None = False


_FAILURE_SPECS: Mapping[tuple[str, str], FailureSpec] = {
    (PHASE_SCAFFOLD_CONTRACT, KIND_INCOMPLETE_SCAFFOLD): FailureSpec(
        OUTCOME_SCAFFOLD_FAILURE, turns_beyond_steps=(0,)
    ),
    (PHASE_TURN_START, KIND_EPISODE_TIMEOUT): FailureSpec(
        OUTCOME_TIMEOUT, turns_beyond_steps=(0,)
    ),
    (PHASE_TURN_START, KIND_MAX_TURNS): FailureSpec(
        OUTCOME_LIMIT_EXHAUSTED, turns_beyond_steps=(0,)
    ),
    (PHASE_TURN_START, KIND_MAX_MESSAGES): FailureSpec(
        OUTCOME_LIMIT_EXHAUSTED, turns_beyond_steps=(0,)
    ),
    # Both adapter-call kinds are raised *out of* the call, so no response ever
    # arrived and the outbound request is left unmatched. They are the only two
    # shapes an odd message count can have.
    (PHASE_ADAPTER_CALL, KIND_PROTOCOL_ERROR): FailureSpec(
        OUTCOME_MODEL_PROTOCOL_FAILURE, exception_type=True, unanswered_request=True
    ),
    # Same shape as the protocol error above — raised out of the call, no step
    # recorded, the outbound request left unmatched because nothing usable came
    # back — and a different outcome, because a truncated answer and a malformed
    # one are different facts with different remedies.
    (PHASE_ADAPTER_CALL, KIND_OUTPUT_TOKEN_LIMIT): FailureSpec(
        OUTCOME_MODEL_OUTPUT_LIMIT, exception_type=True, unanswered_request=True
    ),
    (PHASE_ADAPTER_CALL, KIND_ADAPTER_EXCEPTION): FailureSpec(
        OUTCOME_ADAPTER_FAILURE,
        class_from_exception=True,
        exception_type=True,
        unanswered_request=True,
    ),
    (PHASE_ADAPTER_CALL, KIND_EPISODE_TIMEOUT): FailureSpec(OUTCOME_TIMEOUT),
    # Shaped like an adapter call that raised, because that is where it is
    # raised from: the loop has already counted the turn and the outbound
    # message when the guard refuses, so exactly one message is unmatched. The
    # difference from every other unanswered-request fault is that here the
    # request was never sent at all — which is the point of the outcome's name
    # and is stated in the detail the row carries.
    (PHASE_ADAPTER_CALL, KIND_COST_CAP_EXHAUSTED): FailureSpec(
        OUTCOME_COST_CAP_REACHED, unanswered_request=True
    ),
    # Shaped like every other fault raised out of the call: the answer arrived
    # and was measured, and then the turn raised instead of returning it, so no
    # step was recorded and the outbound message is left unmatched. The row is
    # not empty of evidence, though — its usage carries what the response cost,
    # which is the number the whole outcome exists to preserve.
    (PHASE_ADAPTER_CALL, KIND_COST_RESERVATION_BREACHED): FailureSpec(
        OUTCOME_COST_RESERVATION_BREACHED, unanswered_request=True
    ),
    (PHASE_USAGE, KIND_USAGE_EXCEPTION): FailureSpec(
        OUTCOME_ADAPTER_FAILURE, class_from_exception=True, exception_type=True
    ),
    (PHASE_USAGE, KIND_MALFORMED_USAGE): FailureSpec(OUTCOME_ADAPTER_FAILURE),
    (PHASE_RESPONSE_VALIDATION, KIND_MALFORMED_RESPONSE): FailureSpec(
        OUTCOME_MODEL_PROTOCOL_FAILURE, response_type=True
    ),
    (PHASE_RESPONSE_VALIDATION, KIND_UNKNOWN_ACTION): FailureSpec(
        OUTCOME_MODEL_PROTOCOL_FAILURE, attempted_call=True
    ),
    (PHASE_RESPONSE_VALIDATION, KIND_MALFORMED_ARGUMENTS): FailureSpec(
        OUTCOME_MODEL_PROTOCOL_FAILURE, attempted_call=True
    ),
    # The answer parsed as a call, but its text cannot be written to a row, so
    # the call itself cannot be recorded as evidence of what was said.
    (PHASE_RESPONSE_VALIDATION, KIND_UNREPRESENTABLE_TEXT): FailureSpec(
        OUTCOME_MODEL_PROTOCOL_FAILURE
    ),
    # Both of these are schema-valid calls the case does not admit. The
    # environment refusing them is the environment working correctly.
    (PHASE_DISPATCH, KIND_WRONG_CHANNEL): FailureSpec(
        OUTCOME_MODEL_ACTION_FAILURE, attempted_call=True
    ),
    (PHASE_DISPATCH, KIND_ACTION_REJECTED): FailureSpec(
        OUTCOME_MODEL_ACTION_FAILURE, attempted_call=True
    ),
    # The environment machinery itself failing, which is not the model's doing
    # and can leave a half-recorded step behind.
    (PHASE_DISPATCH, KIND_ENVIRONMENT_EXCEPTION): FailureSpec(
        OUTCOME_ENVIRONMENT_FAILURE,
        class_from_exception=True,
        exception_type=True,
        attempted_call=True,
        turns_beyond_steps=(0, 1),
        terminal_trajectory=None,
    ),
    # The terminal step was recorded before the budget was found to be spent, so
    # this is one of exactly two faults that follow a terminal decision.
    (PHASE_DISPATCH, KIND_EPISODE_TIMEOUT): FailureSpec(
        OUTCOME_TIMEOUT, turns_beyond_steps=(0,), terminal_trajectory=True
    ),
    (PHASE_EVALUATION, KIND_EVALUATOR_EXCEPTION): FailureSpec(
        OUTCOME_EVALUATOR_FAILURE,
        class_from_exception=True,
        exception_type=True,
        turns_beyond_steps=(0,),
        terminal_trajectory=True,
    ),
    # The loop's own machinery, which can break at either end of a turn — and,
    # through the post-dispatch clock read, on either side of a terminal step.
    (PHASE_RUNNER, KIND_RUNNER_EXCEPTION): FailureSpec(
        OUTCOME_RUNNER_FAILURE,
        class_from_exception=True,
        exception_type=True,
        turns_beyond_steps=(0, 1),
        terminal_trajectory=None,
    ),
}

_FAILURE_SPECS = {
    **_FAILURE_SPECS,
    # Every provider fault has the same *shape* as an adapter exception — raised
    # out of the call, no step recorded, one message unmatched — and differs
    # only in what it is called and what outcome it implies. Generated from one
    # table rather than written out seven times, so the shape cannot drift
    # between them and adding a provider fault to the contract cannot leave a
    # hole here.
    **{
        (PHASE_ADAPTER_CALL, kind): FailureSpec(
            outcome, unanswered_request=True, turns_beyond_steps=(1,)
        )
        for kind, outcome in PROVIDER_OUTCOMES_V7.items()
    },
}

#: Every classification this build can produce, as ``phase/kind``.
FAILURE_CLASSIFICATIONS: tuple[str, ...] = tuple(
    f"{phase}/{kind}" for phase, kind in sorted(_FAILURE_SPECS)
)


def classify_failure(
    *,
    phase: str,
    kind: str,
    detail: str = "",
    exception: BaseException | None = None,
    attempted_call: Mapping[str, Any] | None = None,
    response_type: str | None = None,
) -> tuple[str, str, str, dict[str, Any]]:
    """Turn one observed fault into the outcome, class, detail and event of a row.

    The single place a classification is decided. The three summary fields are
    returned together so they cannot drift apart from each other or from the
    cause; they are returned as
    of the same table, and the event that produced them travels with them.

    An exception's ``str()`` may be empty — ``RuntimeError()`` is a real thing a
    provider integration can raise — and a row with a blank ``error_detail`` is
    refused. The detail therefore falls back to something stable and non-empty
    rather than being written out as a row nothing can read back.

    The detail is also *diagnostic prose*, not evidence, and it comes from
    outside: a provider integration's exception message is arbitrary text and
    may hold code points UTF-8 cannot encode. The explicit rule is that such
    text is repaired — each lone surrogate becomes U+FFFD — so that externally
    sourced text can never stop a failure being recorded. Evidence of what a
    model actually said is handled the other way round, by classification; see
    :data:`KIND_UNREPRESENTABLE_TEXT`.
    """
    spec = _FAILURE_SPECS[(phase, kind)]
    exception_type = None if exception is None else type(exception).__name__
    text = sanitized_text(detail).strip() or (
        f"{exception_type} was raised with no message"
        if exception_type is not None
        else f"{kind} in {phase}"
    )
    attempted_payload: dict[str, Any] | None = None
    if spec.attempted_call:
        if attempted_call is None:
            raise ValueError(f"{phase}/{kind} requires the attempted call")
        attempted_payload = dict(attempted_call)
    event = {
        "phase": phase,
        "kind": kind,
        "attempted_call": attempted_payload,
        "response_type": response_type if spec.response_type else None,
        "exception_type": exception_type if spec.exception_type else None,
        "detail": text,
    }
    error_class = exception_type if spec.class_from_exception and exception_type else kind
    return spec.outcome, error_class, text, event


# -- unmatched-query audit ---------------------------------------------------
#
# A model that names an acquisition key outside the set its case offered is a
# sanitized ``model_protocol_failure/malformed_arguments`` and stays one: nothing
# in this build maps such a key onto one of the four authored outcomes, because
# a runtime that guessed what an unoffered question meant would be inventing
# semantics the Cube never authored.
#
# It is still evidence, and it is the evidence a Cube author most needs: it says
# the model expected this workflow to be able to answer something the Cube does
# not resolve. So it is *countable*, from rows this build already writes, and the
# counting lives here rather than in the runner because it is analysis over a
# finished ledger rather than a decision inside an episode.


def _unmatched_query_event(row: Mapping[str, Any]) -> dict[str, Any] | None:
    event = row.get("failure_event")
    if not isinstance(event, Mapping):
        return None
    if event.get("phase") != PHASE_RESPONSE_VALIDATION:
        return None
    if event.get("kind") != KIND_MALFORMED_ARGUMENTS:
        return None
    call = event.get("attempted_call")
    if not isinstance(call, Mapping):
        return None
    action = call.get("action")
    if action not in QUERY_ACTIONS:
        return None
    arguments = call.get("arguments")
    if not isinstance(arguments, Mapping):
        return None
    query_key = arguments.get(FACT_PARAMETER)
    if not isinstance(query_key, str) or not query_key:
        return None
    return {
        "variant_id": row.get("variant_id"),
        "action": action,
        "query_key": query_key,
    }


def unmatched_query_events(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Every recorded turn that asked for a query key its case does not offer.

    Derived from the rows themselves — the phase, the kind and the attempted call
    the runner already records — so an audit reads what happened rather than a
    counter something had to remember to increment. Rows in file order, because
    the order a run met these in is part of what an auditor is reading.
    """
    found = [_unmatched_query_event(row) for row in rows]
    return tuple(event for event in found if event is not None)


def unmatched_query_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """How many turns asked for each unoffered query key, by key.

    Sorted by key, so two audits of the same ledger produce the same report.
    """
    counts: dict[str, int] = {}
    for event in unmatched_query_events(rows):
        key = str(event["query_key"])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


class LedgerError(ValueError):
    """Base class for every ledger failure."""


class LedgerCorruptionError(LedgerError):
    """The file on disk is not a sequence of well-formed rows."""


class LedgerIOError(LedgerError):
    """The ledger file itself cannot be opened, read or appended to.

    Named rather than left as an ``OSError`` so an unwritable run directory or a
    ledger that has been replaced by a link is reported as a run failure instead
    of a traceback.
    """


@dataclass(frozen=True)
class EpisodeRecord:
    """One episode, exactly as it is persisted."""

    configuration_id: str
    execution_id: str
    episode_id: str
    variant_id: str
    trial_index: int
    outcome: str
    sample_status: str
    error_class: str | None
    error_detail: str | None
    failure_event: Mapping[str, Any] | None
    turns_used: int
    messages_used: int
    #: One wall-clock reading, taken as the episode began.
    started_at_utc: str
    #: ``started_at_utc`` plus ``elapsed_seconds`` — derived arithmetic, not a
    #: second wall-clock observation, so a clock stepped backwards mid-episode
    #: cannot record a completion before the start. Read it as an estimate of
    #: the completion instant whose offset from the start is exactly the
    #: duration that was measured.
    completed_at_utc: str
    #: The monotonic duration this build measured locally.
    elapsed_seconds: float
    trajectory: Mapping[str, Any]
    evaluation: Mapping[str, Any] | None
    usage: Mapping[str, Any]
    #: Structured provider-attempt evidence, or ``None`` when no turn of this
    #: episode reached a provider at all. ``None`` is not an empty attempt list:
    #: an in-process fake makes no requests, and saying it made none and failed
    #: would be a different and false claim.
    provider_telemetry: Mapping[str, Any] | None
    #: How much of this episode was actually measured, as a closed three-valued
    #: answer over turns and attempts.
    measurement: Mapping[str, Any]
    suite_content_digest: str
    scaffold_content_digest: str
    variant_content_digest: str
    adapter: Mapping[str, Any]
    contract_versions: Mapping[str, Any]
    #: The conservative upper bound this run reserved for attempts in this
    #: episode that returned no readable usage, in USD. ``None`` means the run
    #: had no cost cap and therefore did no cost accounting at all; ``0`` means
    #: it did, and every attempt this episode made was measured. The two are
    #: different facts and a resume reads them differently, so they are never
    #: collapsed. See :mod:`boundarybench.budget`.
    cost_exposure_usd: float | None = None
    schema_version: int = LEDGER_SCHEMA_VERSION

    #: Every field that carries nested evidence rather than a scalar. Frozen on
    #: the way in, projected back to plain built-ins on the way out.
    _EVIDENCE_FIELDS = (
        "failure_event",
        "trajectory",
        "evaluation",
        "usage",
        "provider_telemetry",
        "measurement",
        "adapter",
        "contract_versions",
    )

    def __post_init__(self) -> None:
        """Freeze the evidence, recursively.

        A frozen dataclass holding live dicts is not immutable, and this
        particular object is the *parsed form of what is on disk*: reading a
        ledger and then editing ``records[0].evaluation['passed']`` changed the
        report's verdict counts while the bytes it claims to summarise stayed
        exactly as they were. Evidence that can drift from its own source is not
        evidence, so every nested structure becomes read-only here — which also
        means the trajectory, the usage and the adapter identity a report shows
        are the ones the ledger holds.
        """
        for name in self._EVIDENCE_FIELDS:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, deep_freeze(value))

    @property
    def succeeded(self) -> bool:
        return self.outcome == OUTCOME_SUCCESS

    @property
    def is_completed_sample(self) -> bool:
        """Whether this row may be counted as one semantic observation."""
        return self.sample_status == SAMPLE_COMPLETED

    def as_dict(self) -> dict[str, Any]:
        """The stored form: plain, detached, deterministic JSON values.

        ``to_json`` rather than ``dict``: the evidence is deep-frozen, so a
        shallow copy would hand read-only proxies and tuples to the encoder.
        The projection shares no state with this record, so a caller may edit
        the payload freely without reaching back into parsed evidence.
        """
        return {
            "schema_version": self.schema_version,
            "configuration_id": self.configuration_id,
            "execution_id": self.execution_id,
            "episode_id": self.episode_id,
            "variant_id": self.variant_id,
            "trial_index": self.trial_index,
            "outcome": self.outcome,
            "sample_status": self.sample_status,
            "error_class": self.error_class,
            "error_detail": self.error_detail,
            "failure_event": (
                None if self.failure_event is None else to_json(self.failure_event)
            ),
            "turns_used": self.turns_used,
            "messages_used": self.messages_used,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "elapsed_seconds": self.elapsed_seconds,
            "trajectory": to_json(self.trajectory),
            "evaluation": None if self.evaluation is None else to_json(self.evaluation),
            "usage": to_json(self.usage),
            "provider_telemetry": (
                None
                if self.provider_telemetry is None
                else to_json(self.provider_telemetry)
            ),
            "measurement": to_json(self.measurement),
            "cost_exposure_usd": self.cost_exposure_usd,
            "suite_content_digest": self.suite_content_digest,
            "scaffold_content_digest": self.scaffold_content_digest,
            "variant_content_digest": self.variant_content_digest,
            "adapter": to_json(self.adapter),
            "contract_versions": to_json(self.contract_versions),
        }


def append_episode(path: str | Path, record: EpisodeRecord) -> None:
    """Append one row, then flush and fsync before returning.

    The durability barrier is the point: the caller may only treat an episode as
    done once it survives a crash, otherwise a resume would re-execute work the
    ledger appears to have recorded.
    """
    path = Path(path)
    try:
        line = (
            canonical_json_text(record.as_dict(), f"episode {record.episode_id}") + "\n"
        )
    except JsonSafetyError as exc:
        raise LedgerError(
            f"episode {record.episode_id} cannot be written to the ledger: {exc}"
        ) from exc
    # O_NOFOLLOW so a ledger swapped for a symlink between validation and this
    # append is refused rather than redirected; O_APPEND so every write lands at
    # the end of the file the descriptor names, whatever else is happening.
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
    except OSError as exc:
        raise LedgerIOError(
            f"cannot append to ledger {path}: {exc}. A ledger is a regular file "
            "written in place; it is never written through a link."
        ) from exc
    try:
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:  # pragma: no cover - full or failing device
        raise LedgerIOError(f"cannot append to ledger {path}: {exc}") from exc


# -- strict reading ----------------------------------------------------------


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    for key, _value in pairs:
        if key in seen:
            raise LedgerCorruptionError(
                f"duplicate JSON key {key!r}; a ledger row states each key once"
            )
        seen.add(key)
    return dict(pairs)


def _reject_constant(name: str) -> Any:
    raise LedgerCorruptionError(f"{name} is not a finite JSON value")


def _positive_int(row: Mapping[str, Any], key: str, context: str) -> int:
    value = row[key]
    # bool is an int subclass, so `true` must not be read as 1.
    if type(value) is not int or value < 0:
        raise LedgerError(
            f"{context}: {key!r} must be a non-negative integer, got {value!r}"
        )
    return value


def _text(row: Mapping[str, Any], key: str, context: str) -> str:
    value = row[key]
    if not isinstance(value, str) or not value.strip():
        raise LedgerError(f"{context}: {key!r} must be a non-empty string")
    return value


def _optional_text(row: Mapping[str, Any], key: str, context: str) -> str | None:
    value = row[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise LedgerError(f"{context}: {key!r} must be a string or null, got {value!r}")
    return value


def _mapping(row: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = row[key]
    if not isinstance(value, Mapping):
        raise LedgerError(f"{context}: {key!r} must be a JSON object")
    return value


def _exact_keys(row: Mapping[str, Any], allowed: tuple[str, ...], context: str) -> None:
    missing = [key for key in allowed if key not in row]
    if missing:
        raise LedgerError(f"{context}: missing required field(s) {missing}")
    unknown = sorted(set(map(str, row)) - set(allowed))
    if unknown:
        raise LedgerError(
            f"{context}: unknown field(s) {unknown}; allowed: {sorted(allowed)}"
        )


def _digest(row: Mapping[str, Any], key: str, expected: str, context: str) -> str:
    value = row[key]
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise LedgerError(
            f"{context}: {key!r} must be a lowercase 64-character SHA-256 hex "
            f"digest, got {value!r}"
        )
    if value != expected:
        raise LedgerError(
            f"{context}: {key} {value} does not match this run's {expected}; the "
            "row was written against different artefacts"
        )
    return value


_TRAJECTORY_KEYS = (
    "variant_id",
    "steps",
    "terminal_decision",
    "observed_ids",
    "mutation_history",
)
_STEP_KEYS = ("index", "action", "arguments", "revealed_observation_ids", "mutations")
_TERMINAL_KEYS = (
    "disposition",
    "primary_reason_code",
    "secondary_reason_codes",
    "evidence_refs",
)
_EVALUATION_KEYS = ("variant_id", "passed", "failed_predicates", "predicates")
_PREDICATE_KEYS = ("name", "passed", "detail")
_TOKEN_KEYS = ("input_tokens", "output_tokens")
_AMOUNT_KEYS = ("cost_usd", "latency_seconds")


def _list_of(value: Any, key: str, context: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise LedgerError(f"{context}: {key!r} must be a JSON array, got {value!r}")
    return value


def _strings(value: Any, key: str, context: str) -> tuple[str, ...]:
    items = _list_of(value, key, context)
    if not all(isinstance(item, str) for item in items):
        raise LedgerError(f"{context}: {key!r} must be an array of strings")
    return tuple(items)


def _step(raw: Any, context: str) -> Step:
    if not isinstance(raw, Mapping):
        raise LedgerError(f"{context} must be a JSON object")
    _exact_keys(raw, _STEP_KEYS, context)
    index = raw["index"]
    if type(index) is not int or index < 1:
        raise LedgerError(f"{context}: 'index' must be a positive integer, got {index!r}")
    arguments = raw["arguments"]
    if not isinstance(arguments, Mapping):
        raise LedgerError(f"{context}: 'arguments' must be a JSON object")
    return Step(
        index=index,
        action=_text(raw, "action", context),
        # Free-form, but already proven representable: the whole row went
        # through the shared JSON-safety walk before any field was read, so a
        # second structural check here would be a duplicate validator, not a
        # second guarantee.
        arguments=dict(arguments),
        revealed_observation_ids=_strings(
            raw["revealed_observation_ids"], "revealed_observation_ids", context
        ),
        mutations=_strings(raw["mutations"], "mutations", context),
    )


def _terminal_decision(raw: Any, context: str) -> TerminalDecision | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise LedgerError(f"{context}: 'terminal_decision' must be a JSON object or null")
    inner = f"{context}: terminal_decision"
    _exact_keys(raw, _TERMINAL_KEYS, inner)
    return TerminalDecision(
        disposition=_text(raw, "disposition", inner),
        primary_reason_code=_text(raw, "primary_reason_code", inner),
        secondary_reason_codes=_strings(
            raw["secondary_reason_codes"], "secondary_reason_codes", inner
        ),
        evidence_refs=_strings(raw["evidence_refs"], "evidence_refs", inner),
    )


def _mutation_history(raw: Any, context: str) -> tuple[tuple[int, str], ...]:
    items = _list_of(raw, "mutation_history", context)
    history: list[tuple[int, str]] = []
    for position, item in enumerate(items):
        entry = f"{context}: mutation_history[{position}]"
        pair = _list_of(item, "mutation_history entry", entry)
        if len(pair) != 2 or type(pair[0]) is not int or not isinstance(pair[1], str):
            raise LedgerError(
                f"{entry}: a mutation_history entry is [step index, action], got {pair!r}"
            )
        history.append((pair[0], pair[1]))
    return tuple(history)


def _trajectory(raw: Any, variant_id: str, context: str) -> Trajectory:
    """Rebuild the typed trajectory a row claims, refusing anything else.

    Reconstruction is the point. A mapping that merely *looks* like a trajectory
    proves nothing; a value that survives being turned back into the same typed
    object the environment produces is the only thing the evaluator can be
    re-run against.
    """
    if not isinstance(raw, Mapping):
        raise LedgerError(f"{context}: 'trajectory' must be a JSON object")
    inner = f"{context}: trajectory"
    _exact_keys(raw, _TRAJECTORY_KEYS, inner)
    recorded_variant = _text(raw, "variant_id", inner)
    if recorded_variant != variant_id:
        raise LedgerError(
            f"{inner}: the trajectory belongs to variant {recorded_variant!r}, but "
            f"the row records episode variant {variant_id!r}"
        )
    steps = tuple(
        _step(item, f"{inner}: steps[{position}]")
        for position, item in enumerate(_list_of(raw["steps"], "steps", inner))
    )
    trajectory = Trajectory(
        variant_id=recorded_variant,
        steps=steps,
        terminal_decision=_terminal_decision(raw["terminal_decision"], inner),
        observed_ids=_strings(raw["observed_ids"], "observed_ids", inner),
        mutation_history=_mutation_history(raw["mutation_history"], inner),
    )
    if canonical_json(trajectory.as_dict()) != canonical_json(dict(raw)):
        raise LedgerError(
            f"{inner}: the stored trajectory is not the canonical encoding of the "
            "trajectory it reconstructs to; it has been edited since it was written"
        )
    return trajectory


def _verify_replay(variant: Variant, trajectory: Trajectory, context: str) -> None:
    """Prove a stored trajectory is one the real environment could have produced.

    Reconstructing the typed object and demanding it re-encode to what was
    stored only proves the row is internally consistent; it says nothing about
    whether the environment would ever have produced these steps in this
    order. An impossible step index, a reveal the environment never granted, a
    mutation it never recorded, an action taken after the case had already
    terminated, or a fact pulled through the wrong channel all reconstruct and
    even regrade correctly, because neither reconstruction nor grading looks at
    where a step actually came from. Replaying every recorded action through a
    fresh environment, with the same dispatch a live episode uses, and
    requiring the result to match exactly is the only check that does.

    This says nothing about whether the trajectory's evaluator verdict is
    correct — a genuinely environment-generated trajectory that answers wrong
    replays to itself perfectly and is graded failing, which is exactly what a
    ledger of real attempts should hold.
    """
    try:
        replayed = replay_trajectory(variant, trajectory)
    except EnvironmentError as exc:
        raise LedgerError(
            f"{context}: the stored trajectory could not be replayed through the "
            f"real environment: {exc}"
        ) from exc
    if canonical_json(replayed.as_dict()) != canonical_json(trajectory.as_dict()):
        raise LedgerError(
            f"{context}: the stored trajectory is not what the real environment "
            "produces when its recorded actions are replayed in order; it does not "
            "describe an episode that could actually have happened"
        )


def _attempted_call(raw: Any, context: str) -> dict[str, Any]:
    """The call a rejected-call failure refused, checked as strictly as a step.

    ``arguments`` is nullable and null means something specific: the adapter did
    not offer a JSON object at all. That is itself one of the faults recorded
    here, so it has to be representable — writing ``{}`` instead would record an
    empty call the adapter never made.
    """
    if not isinstance(raw, Mapping):
        raise LedgerError(f"{context}: 'attempted_call' must be a JSON object")
    inner = f"{context}: attempted_call"
    _exact_keys(raw, _ATTEMPTED_CALL_KEYS, inner)
    action = raw["action"]
    # Not `_text`: an adapter may name the empty action, and refusing to read
    # back the row that records it would lose the evidence of that very fault.
    if not isinstance(action, str):
        raise LedgerError(f"{inner}: 'action' must be a string, got {action!r}")
    arguments = raw["arguments"]
    if arguments is not None and not isinstance(arguments, Mapping):
        raise LedgerError(f"{inner}: 'arguments' must be a JSON object or null")
    return {
        "action": action,
        "arguments": None if arguments is None else dict(arguments),
    }


def _evidence(
    raw: Mapping[str, Any], key: str, required: bool, kind: str, context: str
) -> str | None:
    """One evidence field, present exactly when its kind is defined by it."""
    value = raw[key]
    if required:
        if not isinstance(value, str) or not value.strip():
            raise LedgerError(
                f"{context}: a {kind!r} failure must record {key!r} as a non-empty "
                f"string, got {value!r}"
            )
        return value
    if value is not None:
        raise LedgerError(
            f"{context}: a {kind!r} failure records no {key!r}, so it must be null, "
            f"got {value!r}"
        )
    return None


def _failure_event(
    raw: Any,
    outcome: str,
    error_class: str | None,
    error_detail: str | None,
    context: str,
) -> tuple[Mapping[str, Any] | None, FailureSpec | None]:
    """Rebuild the recorded cause, then derive the classification from it.

    The summary fields are not compared to the event so much as *replaced* by
    what the event implies: the phase and kind pick one row of the taxonomy
    table, and that row states the outcome, the error class and — through the
    detail the event carries — the error detail. Anything the row says instead
    is a classification that does not follow from its own cause, which is
    exactly the forgery this exists to refuse.
    """
    if outcome == OUTCOME_SUCCESS:
        if raw is not None:
            raise LedgerError(
                f"{context}: 'failure_event' must be null for a successful episode; "
                "an episode that reached a terminal decision recorded no fault"
            )
        return None, None
    if raw is None:
        raise LedgerError(
            f"{context}: outcome {outcome!r} must carry a 'failure_event'; a failure "
            "with no recorded cause cannot be checked against its own classification"
        )
    if not isinstance(raw, Mapping):
        raise LedgerError(f"{context}: 'failure_event' must be a JSON object or null")
    inner = f"{context}: failure_event"
    _exact_keys(raw, _FAILURE_EVENT_KEYS, inner)
    phase, kind = raw["phase"], raw["kind"]
    if not isinstance(phase, str):
        raise LedgerError(f"{inner}: 'phase' must be a string, got {phase!r}")
    if not isinstance(kind, str):
        raise LedgerError(f"{inner}: 'kind' must be a string, got {kind!r}")
    spec = _FAILURE_SPECS.get((phase, kind))
    if spec is None:
        raise LedgerError(
            f"{inner}: unknown failure phase/kind {phase}/{kind}; this build "
            f"classifies {list(FAILURE_CLASSIFICATIONS)}"
        )

    detail = _text(raw, "detail", inner)
    attempted = raw["attempted_call"]
    call: dict[str, Any] | None
    if not spec.attempted_call:
        if attempted is not None:
            raise LedgerError(
                f"{inner}: a {kind!r} failure refuses no call, so 'attempted_call' "
                f"must be null, got {attempted!r}"
            )
        call = None
    elif attempted is None:
        raise LedgerError(
            f"{inner}: a {kind!r} failure is the refusal of one call, so "
            "'attempted_call' must record the call it refused"
        )
    else:
        call = _attempted_call(attempted, inner)
    response_type = _evidence(raw, "response_type", spec.response_type, kind, inner)
    exception_type = _evidence(raw, "exception_type", spec.exception_type, kind, inner)

    if outcome != spec.outcome:
        raise LedgerError(
            f"{inner}: a {phase}/{kind} failure is outcome {spec.outcome!r}, but the "
            f"row records {outcome!r}; the classification does not follow from the "
            "cause the row itself states"
        )
    expected_class = (
        exception_type if spec.class_from_exception and exception_type else kind
    )
    if error_class != expected_class:
        raise LedgerError(
            f"{inner}: a {phase}/{kind} failure has error_class {expected_class!r}, "
            f"but the row records {error_class!r}"
        )
    if error_detail != detail:
        raise LedgerError(
            f"{context}: error_detail {error_detail!r} is not the detail its "
            f"failure_event records ({detail!r})"
        )
    return {
        "phase": phase,
        "kind": kind,
        "attempted_call": call,
        "response_type": response_type,
        "exception_type": exception_type,
        "detail": detail,
    }, spec


def _verify_terminality(
    outcome: str,
    spec: FailureSpec | None,
    trajectory: Trajectory,
    context: str,
) -> None:
    """Bind the outcome class to the shape of the episode it claims to describe.

    Internal consistency is necessary but insufficient: counters can agree with
    the taxonomy while the verdict regrades and the trajectory replays. This
    check also asks whether the runner could have produced
    classification from the recorded trajectory. It cannot reach most of them:
    the loop breaks the instant a terminal dispatch returns, so it never asks
    the adapter again, never validates another response and never dispatches
    another action. An ``adapter_failure`` on a closed case is therefore not a
    failure this build can produce, however consistent the rest of the row is —
    and rewriting a genuine success into one was the cheapest forgery available.

    The two faults that *do* follow a terminal decision are the post-dispatch
    timeout and the evaluator raising, and both are required to carry one.
    """
    terminated = trajectory.terminal_decision is not None
    if spec is None:
        if not terminated:
            raise LedgerError(
                f"{context}: outcome {OUTCOME_SUCCESS!r} but the trajectory records "
                "no terminal decision; a successful episode is one that reached one"
            )
        return
    expected = spec.terminal_trajectory
    if expected is None or expected == terminated:
        return
    if expected:
        raise LedgerError(
            f"{context}: outcome {outcome!r} is only reachable after the case "
            "reached its terminal decision, but the trajectory records none; the "
            "classification does not describe an episode this runner can produce"
        )
    raise LedgerError(
        f"{context}: outcome {outcome!r} records a terminal decision, but the "
        "runner stops the episode at the terminal dispatch and can never reach "
        "this failure afterwards; the classification does not describe an "
        "episode this runner can produce"
    )


def _verify_counters(
    outcome: str,
    spec: FailureSpec | None,
    turns_used: int,
    messages_used: int,
    step_count: int,
    context: str,
) -> None:
    """Bind the row's counters to the trajectory and the fault that explain them.

    Neither counter was checked against anything but its own type: a
    type-correct ``turns_used`` and ``messages_used`` proved nothing about the
    episode they claim to summarise. A turn is one attempted adapter call: it
    costs the request that starts it and the response that ends it, so
    ``messages_used`` is twice ``turns_used`` — except for the one fault that
    leaves a request unanswered, an exception raised out of the call itself,
    where exactly one message is unmatched. Which of the two a row may claim is
    not its own choice: it follows from the phase and kind it recorded.

    A turn is also not the same event as a step. A completed turn adds exactly
    one recorded step, and a turn that died before the environment saw a
    validated action adds none. The taxonomy fixes which of those happened, so
    the allowance is exact per fault rather than "at most one, somewhere".
    """
    unanswered = 1 if spec is not None and spec.unanswered_request else 0
    expected_messages = MESSAGES_PER_TURN * turns_used - unanswered
    if messages_used != expected_messages:
        raise LedgerError(
            f"{context}: messages_used {messages_used} is not {expected_messages} "
            f"for {turns_used} turn(s); every turn is one request and one response, "
            "and only an adapter call that raised leaves its request unanswered"
        )
    if spec is None:
        if turns_used != step_count:
            raise LedgerError(
                f"{context}: a successful episode reports turns_used {turns_used} "
                f"but its trajectory records {step_count} step(s); a completed "
                "episode spends exactly one turn per recorded step"
            )
        return
    if turns_used - step_count not in spec.turns_beyond_steps:
        allowed = " or ".join(str(value) for value in spec.turns_beyond_steps)
        raise LedgerError(
            f"{context}: outcome {outcome!r} reports turns_used {turns_used} against "
            f"{step_count} recorded step(s); the recorded failure spends exactly "
            f"{allowed} turn(s) more than it records steps"
        )


def _usage(raw: Mapping[str, Any], context: str) -> dict[str, Any]:
    inner = f"{context}: usage"
    _exact_keys(raw, _USAGE_KEYS, inner)
    checked: dict[str, Any] = {}
    for key in _TOKEN_KEYS:
        value = raw[key]
        if value is None:
            checked[key] = None
            continue
        # bool is an int subclass, so `true` must not be read as one token.
        if type(value) is not int or value < 0:
            raise LedgerError(
                f"{inner}: {key!r} must be a non-negative integer or null, got {value!r}"
            )
        checked[key] = value
    for key in _AMOUNT_KEYS:
        value = raw[key]
        if value is None:
            checked[key] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LedgerError(
                f"{inner}: {key!r} must be a non-negative finite number or null, got "
                f"{value!r}"
            )
        if not math.isfinite(float(value)) or float(value) < 0:
            raise LedgerError(
                f"{inner}: {key!r} must be a non-negative finite number or null, got "
                f"{value!r}"
            )
        checked[key] = value
    return checked


def _optional_amount(value: Any, key: str, context: str) -> float | None:
    """A non-negative finite number, or ``None`` meaning *not measured*."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LedgerError(
            f"{context}: {key!r} must be a non-negative finite number or null, got "
            f"{value!r}"
        )
    if not math.isfinite(float(value)) or float(value) < 0:
        raise LedgerError(
            f"{context}: {key!r} must be a non-negative finite number or null, got "
            f"{value!r}"
        )
    return float(value)


def _bool(raw: Mapping[str, Any], key: str, context: str) -> bool:
    value = raw[key]
    if not isinstance(value, bool):
        raise LedgerError(f"{context}: {key!r} must be a boolean, got {value!r}")
    return value


def _fault_counts(raw: Any, expected: Mapping[str, int], context: str) -> dict[str, int]:
    """The per-fault tally, re-derived from the attempts it claims to summarise."""
    if not isinstance(raw, Mapping):
        raise LedgerError(f"{context}: 'fault_counts' must be a JSON object")
    if dict(raw) != dict(expected):
        raise LedgerError(
            f"{context}: fault_counts {dict(raw)} is not the tally of the attempts "
            f"this row records ({dict(expected)}); the summary does not follow from "
            "its own evidence"
        )
    return dict(expected)


def _provider_attempt(
    raw: Any, position: int, context: str, expected: Decimal | None
) -> dict[str, Any]:
    """One recorded provider attempt, checked against the closed contract.

    Every field is a closed-set value, a small integer or a local measurement,
    so there is nowhere in a well-formed attempt for provider text to be. That
    is enforced here rather than assumed: a row is read by this build long after
    whatever wrote it, and "the writer would not have put a message there" is not
    a property a reader can rely on.
    """
    entry = f"{context}: attempts[{position}]"
    if not isinstance(raw, Mapping):
        raise LedgerError(f"{entry} must be a JSON object")
    _exact_keys(raw, _ATTEMPT_KEYS, entry)
    index = raw["index"]
    if type(index) is not int or index != position + 1:
        raise LedgerError(
            f"{entry}: 'index' must be {position + 1}; attempts are recorded in the "
            f"order they were made, got {index!r}"
        )
    outcome = raw["outcome"]
    if outcome not in ATTEMPT_OUTCOMES:
        raise LedgerError(
            f"{entry}: unknown attempt outcome {outcome!r}; allowed: "
            f"{list(ATTEMPT_OUTCOMES)}"
        )
    fault = raw["fault"]
    if fault is not None and fault not in PROVIDER_FAULTS_V7:
        raise LedgerError(
            f"{entry}: unknown provider fault {fault!r}; allowed: "
            f"{list(PROVIDER_FAULTS_V7)}"
        )
    if (outcome == ATTEMPT_OUTCOME_FAULT) != (fault is not None):
        raise LedgerError(
            f"{entry}: an attempt that ended in a fault names it and one that "
            f"returned a response names none; got outcome {outcome!r} with fault "
            f"{fault!r}"
        )
    status = raw["http_status"]
    # ``None`` is "there was no status", which is what a dropped connection has.
    # It is never 0, and the range is the one HTTP defines rather than any int.
    if status is not None and (type(status) is not int or not 100 <= status <= 599):
        raise LedgerError(
            f"{entry}: 'http_status' must be an HTTP status code or null, got {status!r}"
        )
    usage_reported = _bool(raw, "usage_reported", entry)
    response_received = _bool(raw, "response_received", entry)
    if usage_reported and not response_received:
        raise LedgerError(
            f"{entry}: an attempt cannot have reported usage without receiving a "
            "response; there would have been nothing to read it from"
        )
    reserved, settlement = _attempt_cost(raw, entry, expected)
    if settlement == ATTEMPT_SETTLEMENT_MEASURED and not response_received:
        raise LedgerError(
            f"{entry}: an attempt cannot have settled a measured cost without "
            "receiving a response; there would have been nothing to measure"
        )
    return {
        "index": index,
        "outcome": outcome,
        "fault": fault,
        "http_status": status,
        "latency_seconds": _optional_amount(
            raw["latency_seconds"], "latency_seconds", entry
        ),
        "response_received": response_received,
        "usage_reported": usage_reported,
        # The rederived amount, not the stored text it was proven equal to: what
        # the totals below are summed from is this build's own arithmetic.
        "cost_reservation_usd": None if reserved is None else usd_text(reserved),
        "cost_settlement": settlement,
    }


@dataclass(frozen=True)
class _RequestProjection:
    """Everything needed to rebuild the requests a capped run actually sent.

    A range check is not a derivation: an editor can lower each reservation to a
    plausible value and adjust the exposure total to match. The reservation is
    therefore rebuilt rather than bounded. A turn's reservation is
    its request's conservative input bound at this run's pinned output ceiling
    and pricing policy, and the request is a pure projection of four things the
    row does not get to state: the manifest (the model, the ceiling, the turn
    limit, the pricing block), the scaffold it pins by digest, the compiled
    variant the episode ran, and the trajectory — which is separately replayed
    through the real environment before this is asked, so it is not a free
    variable either. Editing any of them is refused elsewhere; editing the
    reservation alone now disagrees with all of them.

    ``None`` when the run declares no cap: then no request reserved anything,
    and every attempt says so with a null rather than a number.
    """

    scaffold: Scaffold
    model: str
    price: ModelPrice
    max_output_tokens: int
    max_turns: int
    #: The integration whose request mapping and input-token bound rebuild this
    #: run's requests. Resolved once, from the provider the manifest records.
    replay: _ProviderReplay


def derive_cost_exposure_usd(telemetry: Mapping[str, Any] | None) -> Decimal:
    """One episode's conservative exposure, summed from the attempts that hold it.

    Exposure is the reservations of the attempts that returned nothing this build
    could measure — no more and no less. Recomputing it here prevents an edited
    row total from authorising already exposed budget a second time.
    """
    total = Decimal(0)
    if telemetry is None:
        return total
    for turn in telemetry["turns"]:
        for attempt in turn["attempts"]:
            if attempt["cost_settlement"] == ATTEMPT_SETTLEMENT_FORFEITED:
                total += Decimal(attempt["cost_reservation_usd"])
    return total


def derive_authorised_measured_usd(telemetry: Mapping[str, Any] | None) -> Decimal:
    """The most this episode's measured attempts were authorised to cost.

    The sum of the reservations of the attempts that settled as measured. A row
    whose measured cost exceeds it did not have that spend authorised, and the
    only outcome allowed to say so is
    :data:`OUTCOME_COST_RESERVATION_BREACHED`.
    """
    total = Decimal(0)
    if telemetry is None:
        return total
    for turn in telemetry["turns"]:
        for attempt in turn["attempts"]:
            if attempt["cost_settlement"] == ATTEMPT_SETTLEMENT_MEASURED:
                total += Decimal(attempt["cost_reservation_usd"])
    return total


def derive_measured_cost_usd(
    controls: CostControls, usage: Mapping[str, Any]
) -> float | None:
    """What this episode's own token counts cost at the run's pinned price.

    ``None`` when either count is null, and then the row's cost must be null too:
    a turn that reported half a measurement was not half free. ``None`` also for
    a run with no cost cap, which pins no price and therefore records no cost.

    A stored non-negative finite value is insufficient: cost is a function of the
    counts beside it and the policy the manifest pins, and the value is checked by
    applying that function again.
    """
    if not controls.enforces_cost or controls.price is None:
        return None
    tokens = (usage["input_tokens"], usage["output_tokens"])
    if any(count is None for count in tokens):
        return None
    return float(controls.price.cost(input_tokens=tokens[0], output_tokens=tokens[1]))


def _check_row_cost(
    outcome: str,
    controls: CostControls,
    usage: Mapping[str, Any],
    telemetry: Mapping[str, Any] | None,
    cost_exposure_usd: float | None,
    context: str,
) -> None:
    """Re-derive both of a row's money fields and demand the row already says so."""
    capped = controls.enforces_cost and controls.price is not None
    exposure = float(derive_cost_exposure_usd(telemetry)) if capped else None
    if cost_exposure_usd != exposure:
        raise LedgerError(
            f"{context}: 'cost_exposure_usd' is {cost_exposure_usd!r}, and the "
            f"reservations this row's own attempts forfeited come to {exposure!r}. "
            "An episode's exposure is the sum of the attempts that hold it, so a "
            "total that does not follow from them is not evidence of what the run "
            "may have spent"
        )
    measured = derive_measured_cost_usd(controls, usage)
    if usage["cost_usd"] != measured:
        raise LedgerError(
            f"{context}: usage 'cost_usd' is {usage['cost_usd']!r}, and this row's "
            f"own token counts at this run's pinned pricing policy come to "
            f"{measured!r}. A measured cost is computed from the counts beside it "
            "and is never a number stated independently of them"
        )
    if measured is None:
        return
    authorised = derive_authorised_measured_usd(telemetry)
    breached = Decimal(str(measured)) > authorised
    if breached and outcome != OUTCOME_COST_RESERVATION_BREACHED:
        raise LedgerError(
            f"{context}: this row measures USD {measured} against reservations of "
            f"USD {usd_text(authorised)} on the attempts that produced it, which is "
            f"spend this run never authorised, and records outcome {outcome!r}. The "
            f"only outcome that may record it is "
            f"{OUTCOME_COST_RESERVATION_BREACHED!r}"
        )
    if not breached and outcome == OUTCOME_COST_RESERVATION_BREACHED:
        raise LedgerError(
            f"{context}: this row records outcome "
            f"{OUTCOME_COST_RESERVATION_BREACHED!r} and its measured USD {measured} "
            f"is within the USD {usd_text(authorised)} its attempts reserved. A "
            "breach that did not happen is not a failure this run met"
        )


@dataclass(frozen=True)
class _ProviderReplay:
    """How one integration's requests are rebuilt when its ledger is read.

    Every field is taken from that integration's own module rather than restated
    here. A generic body that rebuilt "a request" would be a second
    implementation of a request mapping that is already inside run identity, and
    the two would drift silently: the reader would derive reservations for
    bodies the adapter never sent, and refuse rows that are correct.

    So this is wiring, not logic. What each provider contributes is its own
    request mapping, its own pinned settings and its own input-token bound —
    the same functions its adapter calls before it authorises a request.
    """

    provider: str
    implementation: str
    max_output_tokens: int
    #: The adapter settings a run of this model must have recorded, as this
    #: build would send them.
    pins: Callable[[str], Mapping[str, Any]]
    #: This integration's exact request mapping, for one turn of one model.
    build_payload: Callable[[TurnRequest, str], Mapping[str, Any]]
    #: This integration's own conservative input bound over that body.
    token_bound: Callable[[Mapping[str, Any]], int]


def _anthropic_replay() -> _ProviderReplay:
    from boundarybench.providers import anthropic_messages as module

    return _ProviderReplay(
        provider=module.ANTHROPIC_PROVIDER,
        implementation=module.ANTHROPIC_IMPLEMENTATION,
        max_output_tokens=module.MAX_OUTPUT_TOKENS,
        pins=lambda model: {
            "request_mapping": module.REQUEST_MAPPING_VERSION,
            "max_output_tokens": module.MAX_OUTPUT_TOKENS,
            # A mapping on this API and a string on the other three, which is
            # why the pinned value is spelled per integration rather than
            # copied from one shape.
            "tool_choice": dict(module.TOOL_CHOICE),
            **module.request_profile_for(model).as_settings(),
        },
        build_payload=lambda request, model: module.build_messages_request(
            request, model=model
        ),
        token_bound=module.request_input_token_bound,
    )


def _openai_replay() -> _ProviderReplay:
    from boundarybench.providers import openai_responses as module

    return _ProviderReplay(
        provider=module.OPENAI_PROVIDER,
        implementation=module.OPENAI_IMPLEMENTATION,
        max_output_tokens=module.MAX_OUTPUT_TOKENS,
        pins=lambda model: {
            "request_mapping": module.REQUEST_MAPPING_VERSION,
            "max_output_tokens": module.MAX_OUTPUT_TOKENS,
            "tool_choice": module.TOOL_CHOICE,
            **module.request_profile_for(model).as_settings(),
        },
        build_payload=lambda request, model: module.build_responses_request(
            request, model=model
        ),
        token_bound=module.request_token_bound,
    )


def _xai_replay() -> _ProviderReplay:
    from boundarybench.providers import xai_openai_compat as module

    return _ProviderReplay(
        provider=module.XAI_PROVIDER,
        implementation=module.XAI_IMPLEMENTATION,
        max_output_tokens=module.MAX_OUTPUT_TOKENS,
        pins=lambda model: {
            "request_mapping": module.REQUEST_MAPPING_VERSION,
            "max_output_tokens": module.MAX_OUTPUT_TOKENS,
            "tool_choice": module.TOOL_CHOICE,
            **module.request_profile_for(model).as_settings(),
        },
        build_payload=lambda request, model: module.build_chat_request(
            request, model=model
        ),
        token_bound=module.request_token_bound,
    )


def _mistral_replay() -> _ProviderReplay:
    from boundarybench.providers import mistral_chat as module

    return _ProviderReplay(
        provider=module.MISTRAL_PROVIDER,
        implementation=module.MISTRAL_IMPLEMENTATION,
        max_output_tokens=module.MAX_OUTPUT_TOKENS,
        pins=lambda model: {
            "request_mapping": module.REQUEST_MAPPING_VERSION,
            "max_output_tokens": module.MAX_OUTPUT_TOKENS,
            "tool_choice": module.TOOL_CHOICE,
            **module.request_profile_for(model).as_settings(),
        },
        build_payload=lambda request, model: module.build_mistral_request(
            request, model=model
        ),
        token_bound=module.request_token_bound,
    )


#: Every provider whose capped ledger this build can read, and how.
#:
#: Keyed by the provider name an adapter identity records, and resolved lazily:
#: each loader imports one integration, and an integration imports a provider
#: SDK. Reading the ledger of a run this build never priced — an in-process
#: fake, which has no cap and therefore no reservations to rederive — must not
#: require any SDK to be installed, so nothing here is imported until a run that
#: declares a cap is read.
_PROVIDER_REPLAYS: Mapping[str, Any] = MappingProxyType(
    {
        "anthropic": _anthropic_replay,
        "mistral": _mistral_replay,
        "openai": _openai_replay,
        "xai": _xai_replay,
    }
)


def _provider_replay(provider: str) -> _ProviderReplay:
    """The integration a capped run's rows are checked by, or a refusal.

    Fail-closed on an unknown name, and matched whole: a provider this build
    does not implement is one whose requests it cannot rebuild, so its
    reservations could only be range-checked — which is the defect this
    replaced.
    """
    loader = _PROVIDER_REPLAYS.get(provider)
    if loader is None:
        raise LedgerError(
            f"this run declares a cost cap and names provider {provider!r}, and "
            f"this build implements exactly {sorted(_PROVIDER_REPLAYS)}. What each "
            "attempt reserved is a function of the request that was authorised, so "
            "a request this build cannot rebuild is a reservation it cannot check, "
            "and a ledger it can only range-check is not read"
        )
    replay: _ProviderReplay = loader()
    return replay


def _request_projection(
    manifest: RunManifest, scaffold: Scaffold | None
) -> _RequestProjection | None:
    """How this build would rebuild this run's requests, or why it cannot.

    Refusing is the whole point of the checks here. A capped run whose requests
    this build cannot reproduce byte-for-byte — a provider it does not implement,
    a request mapping it does not ship, an output ceiling or a sampling setting
    the manifest pins to something other than what this adapter sends — is a run
    whose reservations cannot be rederived, and a reservation that cannot be
    rederived can only be range-checked, which is the defect this replaced. So
    the ledger is refused rather than read under a weaker rule.
    """
    controls = manifest.cost_controls
    if not controls.enforces_cost or controls.price is None:
        return None
    identity = manifest.adapter
    # Dispatched on the provider the manifest records, exactly: this build ships
    # four integrations and each one has its own request mapping, so a reader
    # wired to a single provider could write any of their ledgers and read only
    # one of them back.
    replay = _provider_replay(identity.provider)
    if identity.implementation != replay.implementation:
        raise LedgerError(
            "this run declares a cost cap, and this build cannot rebuild the "
            f"requests its adapter ({identity.provider!r}, "
            f"{identity.implementation!r}) sent. What each attempt reserved is a "
            "function of the request that was authorised, so a reservation this "
            "build cannot reproduce cannot be checked, and a ledger it can only "
            "range-check is not read"
        )
    settings = manifest.adapter_settings
    # The request shape is a property of the run's *model* as well as its
    # provider, not of one module-level constant: since adapter 0.6.0 two models
    # are sent different bodies, so what this build would send is looked up for
    # the exact pair the manifest pins.
    pinned = {
        # What the request offered is as much a part of it as what it asked for:
        # a run whose bodies declared a different tool set reserved against
        # different bodies, and this build cannot rebuild those.
        "action_surface": ACTION_SURFACE_CONTRACT,
        # And what the offered tools *accepted*: the resolution contract fixes
        # the enum every acquisition tool carried, so a run made under a
        # different one sent different bodies of a different size.
        "query_resolution": QUERY_RESOLUTION_CONTRACT,
        **replay.pins(identity.model),
    }
    for key, expected in pinned.items():
        stored = to_json(settings.get(key))
        if type(stored) is not type(expected) or stored != expected:
            raise LedgerError(
                f"this run declares a cost cap and pins adapter setting {key!r} to "
                f"{stored!r}, and this build sends {expected!r}. The request that "
                "was authorised is what fixes what it reserved, so a request this "
                "build would not send is one whose reservations it cannot rederive"
            )
    return _RequestProjection(
        scaffold=_projection_scaffold(manifest, scaffold),
        model=identity.model,
        price=controls.price,
        max_output_tokens=replay.max_output_tokens,
        max_turns=manifest.limits.max_turns,
        replay=replay,
    )


def _projection_scaffold(manifest: RunManifest, scaffold: Scaffold | None) -> Scaffold:
    """The scaffold this run pinned, proven to be the one it pinned.

    A caller that already loaded the run's scaffold — the runner, on every
    resume — hands it in; a reader that has only the directory falls back to the
    scaffold this build ships. Either way the object used is required to hash to
    the digest the manifest records, so what the requests are rebuilt from is
    fixed by the manifest rather than by whoever called the reader.
    """
    if scaffold is None:
        try:
            scaffold = load_scaffold(STANDARD_SCAFFOLD)
        except ScaffoldError as exc:
            raise LedgerError(
                "this run declares a cost cap, and the scaffold its requests would "
                f"be rebuilt from cannot be loaded: {exc}"
            ) from exc
    if scaffold.content_digest != manifest.scaffold_content_digest:
        raise LedgerError(
            f"this run pins scaffold {manifest.scaffold_content_digest}, and the "
            f"scaffold available to rebuild its requests is "
            f"{scaffold.content_digest}. A capped run's reservations are rederived "
            "from the requests its own scaffold produces, so they cannot be checked "
            "against a different one"
        )
    return scaffold


def _turn_reservations(
    projection: _RequestProjection,
    variant: Variant,
    trajectory: Trajectory,
    turns_used: int,
    context: str,
) -> dict[int, Decimal]:
    """What each turn of this episode reserved, rebuilt from its own request.

    The trajectory has already been replayed through the real environment and
    the row's turn and step counters have already been bound to the fault it
    records, so the state turn ``t`` started from is the state after ``t - 1``
    steps. A row that claims more turns than that correspondence can place is
    refused rather than guessed at.
    """
    replay = projection.replay
    reservations: dict[int, Decimal] = {}
    states = replay_turn_states(variant, trajectory)
    for turn in range(1, turns_used + 1):
        try:
            environment = next(states)
        except StopIteration:
            raise LedgerError(
                f"{context}: the row records {turns_used} turn(s) against "
                f"{len(trajectory.steps)} recorded step(s), so the request turn "
                f"{turn} sent cannot be placed in this episode and what it reserved "
                "cannot be rederived"
            ) from None
        request = build_turn_request(
            scaffold=projection.scaffold,
            environment=environment,
            turns_remaining=projection.max_turns - (turn - 1),
        )
        try:
            payload = replay.build_payload(request, projection.model)
            bound = replay.token_bound(payload)
        except (AdapterError, JsonSafetyError, ValueError) as exc:
            raise LedgerError(
                f"{context}: the request turn {turn} sent cannot be rebuilt from "
                f"this run's own scaffold, variant and trajectory: {exc}"
            ) from exc
        reservations[turn] = projection.price.cost(
            input_tokens=bound, output_tokens=projection.max_output_tokens
        )
    return reservations


def _attempt_cost(
    raw: Mapping[str, Any], entry: str, expected: Decimal | None
) -> tuple[Decimal | None, str | None]:
    """One attempt's reservation and how it was resolved, checked as a pair.

    The pair is the point. A reservation with no settlement is money the row
    cannot account for; a settlement with no reservation resolves nothing; and
    either of them present on a run with no cap is a claim about an
    authorisation that never happened.

    ``expected`` is what this turn's own request reserved, rebuilt by
    :func:`_turn_reservations` from the manifest, the scaffold it pins, the
    compiled variant and the replayed trajectory. The stored amount is required
    to be exactly it, rendered exactly as :func:`~boundarybench.budget.usd_text`
    renders it, because that is the one rendering this build writes. A range
    would accept a plausible edited reservation with an exposure total adjusted
    to match; exact derivation binds both to the model-specific request profile.
    """
    reserved_text = raw["cost_reservation_usd"]
    settlement = raw["cost_settlement"]
    if (reserved_text is None) != (settlement is None):
        raise LedgerError(
            f"{entry}: 'cost_reservation_usd' and 'cost_settlement' are stated "
            f"together or not at all, got {reserved_text!r} and {settlement!r}"
        )
    if reserved_text is None:
        if expected is not None:
            raise LedgerError(
                f"{entry}: this run declares a cost cap, so every attempt it made "
                "was authorised against it and states the reservation that "
                "authorised it. A null is 'this run authorised no amount', which "
                "is not true of a capped run"
            )
        return None, None
    if expected is None:
        raise LedgerError(
            f"{entry}: this run declares no cost cap, so nothing authorised an "
            f"amount for this request, and the row states a reservation of "
            f"{reserved_text!r}"
        )
    if settlement not in ATTEMPT_SETTLEMENTS:
        raise LedgerError(
            f"{entry}: unknown cost settlement {settlement!r}; allowed: "
            f"{list(ATTEMPT_SETTLEMENTS)}"
        )
    if not isinstance(reserved_text, str):
        raise LedgerError(
            f"{entry}: 'cost_reservation_usd' must be a decimal *string* or null, "
            f"got {reserved_text!r}; a JSON number is a binary float to most "
            "readers and is not the amount that was reserved"
        )
    if reserved_text != usd_text(expected):
        raise LedgerError(
            f"{entry}: 'cost_reservation_usd' is {reserved_text!r}, and the request "
            f"this turn sent reserves USD {usd_text(expected)} at this run's pinned "
            "output ceiling and pricing policy. The amount is rebuilt from the "
            "manifest, the scaffold it pins, the compiled variant and the replayed "
            "trajectory — none of which this row states — so an attempt claiming a "
            "different amount is refused however plausible it is, and whatever the "
            "totals derived from it add up to"
        )
    return expected, settlement


def _retry_max_attempts(manifest: RunManifest, context: str) -> int:
    """How many attempts one turn of this run was permitted to make.

    Read from the retry policy the run's adapter settings pin, which is inside
    the configuration digest and therefore inside run identity: it is not a
    number the row states about itself, and it is not this build's default
    either — an operator may run a different bound, and the run that records it
    is the authority on what its own turns were allowed to do.

    A row that records provider attempts under a manifest that declares no such
    policy is refused rather than read under an assumed one. An attempt count
    nothing bounds cannot be checked, and "the writer surely stopped at three"
    is exactly the assumption that let a fourth attempt be appended.
    """
    settings = manifest.adapter_settings
    retry = to_json(settings.get("retry"))
    if not isinstance(retry, Mapping):
        raise LedgerError(
            f"{context}: this row records provider attempts, and the run's pinned "
            "adapter settings declare no 'retry' policy to bound them. How many "
            "attempts a turn was permitted is part of run identity, and attempts "
            "nothing bounds are not evidence of what the run was allowed to do"
        )
    permitted = retry.get("max_attempts")
    if type(permitted) is not int or permitted < 1:
        raise LedgerError(
            f"{context}: this run pins retry 'max_attempts' to {permitted!r}, which "
            "is not a number of attempts a turn could have been permitted"
        )
    return permitted


def _check_turn_sequence(
    attempts: Sequence[Mapping[str, Any]],
    reason: Any,
    max_attempts: int,
    entry: str,
) -> str | None:
    """Prove one turn's attempts are a sequence its retry loop could have made.

    Three things a per-attempt check cannot see, in the order a turn meets them:

    * **Only the last attempt may end the turn.** A response returns out of the
      retry loop and a fault that repeating cannot change raises out of it, so
      neither can be followed by another attempt. A genuine
      ``[fault, fault, response]`` turn reordered to
      ``[response, fault, fault]`` and renumbered keeps every count, tally and
      total it had, which is why arithmetic accepted it.
    * **A turn may not exceed the policy the run pinned.** The bound is the
      manifest's, not this build's default, and a fourth attempt under a
      three-attempt policy is a request this run's own adapter would not have
      made however plausible the attempt looks.
    * **A turn that stopped early has to say why.** Everything above places a
      turn that ran to a response, to a terminal fault or to the end of its
      policy. A turn that stopped after a retryable fault with attempts still
      permitted is the one state the attempts cannot place — it is what a
      spent wall clock, a cancelled episode, an unaffordable backoff and an
      exhausted cost cap all legitimately produce, and it is also what deleting
      a turn's later attempts produces. So the writer records which of those it
      was, from a closed set, and this requires it to be one of them. It is not
      inferred from any message: the reason is a typed field or it is nothing.

    Returns the reason, proven to be the one this turn's own evidence permits.
    """
    if reason is not None and reason not in TURN_TERMINAL_REASONS:
        raise LedgerError(
            f"{entry}: unknown terminal_reason {reason!r}; a turn ends for one of "
            f"the reasons this contract defines, or states none: "
            f"{list(TURN_TERMINAL_REASONS)}"
        )
    for position, attempt in enumerate(attempts[:-1]):
        if (
            attempt["outcome"] != ATTEMPT_OUTCOME_FAULT
            or attempt["fault"] not in RETRYABLE_PROVIDER_FAULTS_V7
        ):
            raise LedgerError(
                f"{entry}: attempt {position + 1} of {len(attempts)} ended in "
                f"{attempt['outcome']!r}/{attempt['fault']!r} and is followed by "
                "another attempt. A response ends the turn, and so does a fault "
                "repeating cannot change; only a retryable fault may be retried, "
                "so no attempt can follow one of the others"
            )
    if len(attempts) > max_attempts:
        raise LedgerError(
            f"{entry}: the turn records {len(attempts)} attempt(s), and this run "
            f"pins retry max_attempts to {max_attempts}. The bound is inside run "
            "identity, so an attempt beyond it is one this run's adapter could not "
            "have made"
        )
    permitted = _permitted_terminal_reasons(attempts, max_attempts)
    if reason not in permitted:
        raise LedgerError(
            f"{entry}: terminal_reason {reason!r} is not what this turn's own "
            f"{len(attempts)} recorded attempt(s) under a {max_attempts}-attempt "
            f"policy can have ended for ({list(permitted)}). A turn that stopped "
            "with retries still permitted stopped for a reason this run holds — a "
            "spent or cancelled deadline, a backoff that would not fit, an "
            "exhausted cost cap, or a refused final client check — and a turn that "
            "ran to a response, to a terminal fault or to the end of its policy "
            "ended for exactly that"
        )
    return None if reason is None else str(reason)


def _permitted_terminal_reasons(
    attempts: Sequence[Mapping[str, Any]], max_attempts: int
) -> tuple[Any, ...]:
    """Which endings this turn's recorded attempts allow, and no others.

    Every branch but the last is a single value re-derived from the attempts
    themselves, so the stored reason adds nothing there and is required to agree.
    The last — a retryable fault with attempts still permitted — is the state
    the attempts cannot distinguish, and is the only place the recorded reason
    is load-bearing. A turn with no attempts at all is the same case seen before
    its first request: the run's own clock, budget or final client admission
    stopped it, or it never reached the retry loop and reports no ending.
    """
    if not attempts:
        return (
            None,
            TURN_END_DEADLINE_EXCEEDED,
            TURN_END_COST_CAP_EXHAUSTED,
            TURN_END_PRE_DISPATCH_REFUSED,
        )
    last = attempts[-1]
    if last["outcome"] == ATTEMPT_OUTCOME_RESPONSE:
        return (TURN_END_RESPONSE,)
    if last["outcome"] == ATTEMPT_OUTCOME_UNCLASSIFIED:
        return (TURN_END_UNCLASSIFIED,)
    if last["fault"] not in RETRYABLE_PROVIDER_FAULTS_V7:
        return (TURN_END_FAULT_NOT_RETRYABLE,)
    if len(attempts) == max_attempts:
        return (TURN_END_RETRIES_EXHAUSTED,)
    return TURN_END_EARLY_REASONS


def _telemetry_turn(
    raw: Any,
    position: int,
    turns_used: int,
    context: str,
    reservations: Mapping[int, Decimal] | None,
    max_attempts: int,
) -> dict[str, Any]:
    entry = f"{context}: turns[{position}]"
    if not isinstance(raw, Mapping):
        raise LedgerError(f"{entry} must be a JSON object")
    _exact_keys(raw, _TELEMETRY_TURN_KEYS, entry)
    turn_index = raw["turn_index"]
    if type(turn_index) is not int or not 1 <= turn_index <= turns_used:
        raise LedgerError(
            f"{entry}: 'turn_index' must be a turn this episode spent (1..{turns_used}"
            f"), got {turn_index!r}"
        )
    # Every attempt of one turn is the same request, retried, so they are all
    # checked against that turn's own rederived reservation.
    expected = None if reservations is None else reservations[turn_index]
    attempts = [
        _provider_attempt(item, position_, entry, expected)
        for position_, item in enumerate(_list_of(raw["attempts"], "attempts", entry))
    ]
    count = raw["attempt_count"]
    if type(count) is not int or count != len(attempts):
        raise LedgerError(
            f"{entry}: attempt_count {count!r} is not the number of attempts this "
            f"turn records ({len(attempts)})"
        )
    # Before any summary is re-derived: a tally of attempts that could not have
    # happened in that order, or in that number, is arithmetic over evidence
    # rather than evidence.
    terminal_reason = _check_turn_sequence(
        attempts, raw["terminal_reason"], max_attempts, entry
    )
    expected_faults: dict[str, int] = {}
    for attempt in attempts:
        if attempt["fault"] is not None:
            expected_faults[attempt["fault"]] = (
                expected_faults.get(attempt["fault"], 0) + 1
            )
    return {
        "turn_index": turn_index,
        "attempts": attempts,
        "attempt_count": count,
        "fault_counts": _fault_counts(
            raw["fault_counts"], dict(sorted(expected_faults.items())), entry
        ),
        "response_received": _derived_flag(
            raw, "response_received", any(a["response_received"] for a in attempts), entry
        ),
        "usage_reported": _derived_flag(
            raw, "usage_reported", any(a["usage_reported"] for a in attempts), entry
        ),
        "turn_latency_seconds": _optional_amount(
            raw["turn_latency_seconds"], "turn_latency_seconds", entry
        ),
        "terminal_reason": terminal_reason,
    }


def _derived_flag(raw: Mapping[str, Any], key: str, expected: bool, context: str) -> bool:
    stored = _bool(raw, key, context)
    if stored != expected:
        raise LedgerError(
            f"{context}: {key} {stored!r} does not follow from the attempts this "
            f"turn records ({expected!r})"
        )
    return stored


def _provider_telemetry(
    raw: Any,
    turns_used: int,
    context: str,
    reservations: Mapping[int, Decimal] | None,
    manifest: RunManifest,
    every_turn_reached_the_provider: bool,
) -> Mapping[str, Any] | None:
    """Rebuild the turn-by-turn provider evidence, or prove there is none.

    ``null`` is a claim in its own right — no turn of this episode reached a
    provider — and is therefore allowed. An empty ``turns`` list is not: it says
    the same thing in a second way, and two spellings of one fact are two things
    a count can disagree about.

    ``every_turn_reached_the_provider`` is what makes the accounting *exact*
    rather than merely ordered. On a run this build can rebuild the requests of
    — a capped run through the provider adapter it implements — every turn the
    episode counted was one request to that provider, banked whatever happened
    next, so the recorded turns are ``1..turns_used`` and nothing else. Without
    it the reader could only require indices to be sorted and unique, and a
    complete turn deleted from a two-turn episode satisfied that.
    """
    if raw is None:
        if every_turn_reached_the_provider and turns_used > 0:
            raise LedgerError(
                f"{context}: the row records {turns_used} turn(s) of a run whose "
                "every turn is a request to the provider, and no provider telemetry "
                "at all. An episode that reached a provider has evidence of what it "
                "asked; 'null' is the claim that it reached none"
            )
        return None
    if not isinstance(raw, Mapping):
        raise LedgerError(
            f"{context}: 'provider_telemetry' must be a JSON object or null"
        )
    inner = f"{context}: provider_telemetry"
    _exact_keys(raw, _TELEMETRY_KEYS, inner)
    items = _list_of(raw["turns"], "turns", inner)
    if not items:
        raise LedgerError(
            f"{inner}: 'turns' must not be empty; an episode that reached no "
            "provider records 'provider_telemetry': null rather than an empty list"
        )
    max_attempts = _retry_max_attempts(manifest, inner)
    turns = [
        _telemetry_turn(item, position, turns_used, inner, reservations, max_attempts)
        for position, item in enumerate(items)
    ]
    indices = [turn["turn_index"] for turn in turns]
    if indices != sorted(set(indices)):
        raise LedgerError(
            f"{inner}: turns are recorded once each, in the order they ran; got "
            f"turn indices {indices}"
        )
    if every_turn_reached_the_provider and indices != list(range(1, turns_used + 1)):
        raise LedgerError(
            f"{inner}: the row spent {turns_used} turn(s) and records provider "
            f"evidence for turn(s) {indices}. Every turn of this run is one request "
            "to the provider, counted before it is sent and recorded whatever comes "
            "back, so the turns an episode spent and the turns it has evidence for "
            "are the same turns. A turn missing from the evidence is exposure "
            "missing from the accounting"
        )
    # A turn that did not return a response did not return to the runner either:
    # every other ending raises out of the adapter call and ends the episode with
    # it. So at most one turn — the last — can have ended any other way.
    for turn in turns[:-1]:
        if turn["terminal_reason"] != TURN_END_RESPONSE:
            raise LedgerError(
                f"{inner}: turn {turn['turn_index']} ended in "
                f"{turn['terminal_reason']!r} and is followed by another turn. Only "
                "a turn that returned a response is answered and continued; every "
                "other ending is raised out of the adapter call and ends the "
                "episode, so it is the last turn there is"
            )
    total = sum(turn["attempt_count"] for turn in turns)
    count = raw["attempt_count"]
    if type(count) is not int or count != total:
        raise LedgerError(
            f"{inner}: attempt_count {count!r} is not the number of attempts this "
            f"episode records ({total})"
        )
    expected_faults: dict[str, int] = {}
    for turn in turns:
        for fault, number in turn["fault_counts"].items():
            expected_faults[fault] = expected_faults.get(fault, 0) + number
    status = raw["measurement_status"]
    expected_status = measurement_status_for(
        turns_used, sum(1 for turn in turns if turn["turn_latency_seconds"] is not None)
    )
    if status != expected_status:
        raise LedgerError(
            f"{inner}: measurement_status {status!r} does not follow from the "
            f"{len(turns)} recorded turn(s) of {turns_used} spent ({expected_status!r})"
        )
    return {
        "turns": turns,
        "attempt_count": count,
        "fault_counts": _fault_counts(
            raw["fault_counts"], dict(sorted(expected_faults.items())), inner
        ),
        "measurement_status": status,
    }


def measurement_status_for(turns_used: int, turns_measured: int) -> str:
    """Full, partial or unmeasured — never a boolean.

    A single "measured" flag on an episode was the defect: an episode where one
    turn of four reported a measurement read exactly like one where all four
    did, and a report that counted measured episodes was counting two different
    things under one name.
    """
    if turns_measured == 0:
        return MEASUREMENT_UNMEASURED
    if turns_used > 0 and turns_measured == turns_used:
        return MEASUREMENT_FULL
    return MEASUREMENT_PARTIAL


def derive_measurement(
    telemetry: Mapping[str, Any] | None, turns_used: int
) -> dict[str, Any]:
    """How much of one episode was actually measured, over turns and attempts.

    ``attempts_total`` is ``None`` rather than ``0`` when no turn reported
    telemetry: nothing counted the attempts, which is not the same claim as
    counting none. A provider turn whose budget was spent before a request could
    be dispatched genuinely made zero attempts, and that is recorded as ``0``.
    """
    if telemetry is None:
        return {
            "turns_total": turns_used,
            "turns_measured": 0,
            "attempts_total": None,
            "attempts_measured": None,
            "status": MEASUREMENT_UNMEASURED,
        }
    turns = list(telemetry["turns"])
    attempts = [attempt for turn in turns for attempt in turn["attempts"]]
    turns_measured = sum(1 for turn in turns if turn["turn_latency_seconds"] is not None)
    return {
        "turns_total": turns_used,
        "turns_measured": turns_measured,
        "attempts_total": len(attempts),
        "attempts_measured": sum(
            1 for attempt in attempts if attempt["latency_seconds"] is not None
        ),
        "status": measurement_status_for(turns_used, turns_measured),
    }


def _measurement(
    raw: Any, telemetry: Mapping[str, Any] | None, turns_used: int, context: str
) -> Mapping[str, Any]:
    """Re-derive the coverage summary and demand the row already says so."""
    if not isinstance(raw, Mapping):
        raise LedgerError(f"{context}: 'measurement' must be a JSON object")
    inner = f"{context}: measurement"
    _exact_keys(raw, _MEASUREMENT_KEYS, inner)
    expected = derive_measurement(telemetry, turns_used)
    if dict(raw) != expected:
        raise LedgerError(
            f"{inner}: the recorded coverage {dict(raw)} is not what this row's own "
            f"turns and attempts produce ({expected}); a measurement summary that "
            "does not follow from its evidence is not a measurement"
        )
    return expected


def _check_evaluation_shape(raw: Mapping[str, Any], context: str) -> None:
    inner = f"{context}: evaluation"
    _exact_keys(raw, _EVALUATION_KEYS, inner)
    if not isinstance(raw["passed"], bool):
        raise LedgerError(f"{inner}: 'passed' must be a boolean, got {raw['passed']!r}")
    _strings(raw["failed_predicates"], "failed_predicates", inner)
    predicates = _list_of(raw["predicates"], "predicates", inner)
    for position, item in enumerate(predicates):
        entry = f"{inner}: predicates[{position}]"
        if not isinstance(item, Mapping):
            raise LedgerError(f"{entry} must be a JSON object")
        _exact_keys(item, _PREDICATE_KEYS, entry)
        _text(item, "name", entry)
        if not isinstance(item["passed"], bool):
            raise LedgerError(f"{entry}: 'passed' must be a boolean")
        if not isinstance(item["detail"], str):
            raise LedgerError(f"{entry}: 'detail' must be a string")


def _bind_evaluation(
    raw: Any,
    outcome: str,
    variant: Variant,
    trajectory: Trajectory,
    context: str,
) -> Mapping[str, Any] | None:
    """Regrade the reconstructed trajectory and demand the row already says so.

    A stored verdict is a claim about what the evaluator decided. The only way
    to check that claim is to run the evaluator again over the reconstructed
    trajectory and require exact canonical agreement — which is also what makes
    a forged predicate detail, a flipped ``passed`` flag or a success recorded
    without any verdict impossible to resume onto.
    """
    if outcome != OUTCOME_SUCCESS:
        if raw is not None:
            raise LedgerError(
                f"{context}: 'evaluation' must be null for outcome {outcome!r}; only "
                "an episode that reached a terminal decision is graded"
            )
        return None
    if raw is None:
        raise LedgerError(
            f"{context}: 'evaluation' must not be null for a successful episode; a "
            "success without a verdict is not evidence of anything"
        )
    if not isinstance(raw, Mapping):
        raise LedgerError(f"{context}: 'evaluation' must be a JSON object or null")
    # That the trajectory reached a terminal decision is settled before this is
    # called, by ``_verify_terminality``; a success that had not would never
    # arrive here.
    _check_evaluation_shape(raw, context)
    try:
        recomputed = evaluate(variant, trajectory)
    except Exception as exc:
        raise LedgerError(
            f"{context}: the recorded trajectory cannot be graded against variant "
            f"{variant.variant_id}: {type(exc).__name__}: {exc}"
        ) from exc
    if canonical_json(recomputed.as_dict()) != canonical_json(dict(raw)):
        raise LedgerError(
            f"{context}: the stored 'evaluation' is not what the evaluator gives for "
            "this trajectory; the recorded verdict does not follow from the recorded "
            "episode"
        )
    return dict(raw)


def _parse_row(
    raw: Any,
    manifest: RunManifest,
    variants: Mapping[str, Variant],
    projection: _RequestProjection | None,
    context: str,
) -> EpisodeRecord:
    if not isinstance(raw, Mapping):
        raise LedgerCorruptionError(f"{context}: each row must be a JSON object")
    # Before any field is read: a row is only evidence if it can be re-encoded
    # to the bytes it claims to be, and a JSON ``\ud800`` escape parses into a
    # string that cannot be. Checking the whole row once here means no later
    # comparison, digest or replay has to guard against it separately.
    try:
        ensure_json_safe(raw, context)
    except JsonSafetyError as exc:
        raise LedgerCorruptionError(f"{context}: {exc}") from exc
    _exact_keys(raw, _ROW_KEYS, context)

    version = raw["schema_version"]
    if type(version) is not int or version != LEDGER_SCHEMA_VERSION:
        raise LedgerError(
            f"{context}: unsupported schema_version {version!r}; this build reads "
            f"{LEDGER_SCHEMA_VERSION}"
        )

    configuration_id = _text(raw, "configuration_id", context)
    if configuration_id != manifest.configuration_id:
        raise LedgerError(
            f"{context}: row belongs to configuration {configuration_id}, not "
            f"{manifest.configuration_id}; a ledger holds exactly one run"
        )
    # The execution, as well as the configuration. Two independent runs of one
    # plan share a configuration id exactly, so checking only that would let a
    # row from another directory's execution be read as evidence about this one.
    execution_id = _text(raw, "execution_id", context)
    if execution_id != manifest.execution_id:
        raise LedgerError(
            f"{context}: row was written by execution {execution_id}, but this run "
            f"directory holds execution {manifest.execution_id}. A configuration can "
            "be executed any number of times; a ledger records exactly one of those "
            "executions."
        )

    episode_id = _text(raw, "episode_id", context)
    planned = {entry.episode_id: entry for entry in manifest.episode_plan}
    entry = planned.get(episode_id)
    if entry is None:
        raise LedgerError(
            f"{context}: episode {episode_id!r} is not in this run's episode plan"
        )
    variant_id = _text(raw, "variant_id", context)
    if variant_id != entry.variant_id:
        raise LedgerError(
            f"{context}: variant_id {variant_id!r} does not match the planned "
            f"variant {entry.variant_id!r} for this episode"
        )
    trial_index = _positive_int(raw, "trial_index", context)
    if trial_index != entry.trial_index:
        raise LedgerError(
            f"{context}: trial_index {trial_index} does not match the planned "
            f"trial {entry.trial_index} for this episode"
        )

    outcome = _text(raw, "outcome", context)
    if outcome not in OUTCOMES:
        raise LedgerError(
            f"{context}: unknown outcome {outcome!r}; allowed: {list(OUTCOMES)}"
        )
    # Re-derived from the outcome rather than compared field-by-field to a
    # stored opinion: eligibility is a closed function of what happened, so a
    # row claiming any other status is a row whose own counts contradict it.
    sample_status = _text(raw, "sample_status", context)
    if sample_status != sample_status_for(outcome):
        raise LedgerError(
            f"{context}: outcome {outcome!r} is a {sample_status_for(outcome)!r}, but "
            f"the row records {sample_status!r}. What a row is a sample of follows "
            "from what happened in it and is never stated independently."
        )
    error_class = _optional_text(raw, "error_class", context)
    error_detail = _optional_text(raw, "error_detail", context)
    if outcome == OUTCOME_SUCCESS:
        if error_class is not None or error_detail is not None:
            raise LedgerError(
                f"{context}: a successful episode carries no 'error_class' or "
                f"'error_detail', got {error_class!r} and {error_detail!r}"
            )
    elif not error_class or not error_detail:
        raise LedgerError(
            f"{context}: outcome {outcome!r} must name its 'error_class' and "
            "'error_detail'; a failure with no classification is not a record of "
            "anything"
        )

    variant = variants.get(entry.variant_id)
    if variant is None:
        raise LedgerError(
            f"{context}: variant {entry.variant_id!r} is not in the compiled suite "
            "this run is bound to, so the row cannot be checked against it"
        )
    variant_digest = _digest(
        raw, "variant_content_digest", variant.content_digest, context
    )
    trajectory = _trajectory(raw["trajectory"], entry.variant_id, context)
    _verify_replay(variant, trajectory, context)
    # Terminality before grading: the outcome class decides whether the episode
    # is even allowed to have reached a decision, and grading a trajectory whose
    # shape contradicts its own classification would answer the wrong question.
    failure_event, spec = _failure_event(
        raw["failure_event"], outcome, error_class, error_detail, context
    )
    _verify_terminality(outcome, spec, trajectory, context)
    evaluation = _bind_evaluation(
        raw["evaluation"], outcome, variant, trajectory, context
    )

    turns_used = _positive_int(raw, "turns_used", context)
    messages_used = _positive_int(raw, "messages_used", context)
    _verify_counters(
        outcome, spec, turns_used, messages_used, len(trajectory.steps), context
    )

    # The episode's own window. Refused rather than repaired when it does not
    # parse as this build's one UTC rendering, and refused when it runs
    # backwards: a row whose episode completed before it started is not a record
    # of an episode. The two timestamps are string-comparable because the
    # rendering is fixed-width UTC, which is why there is only one rendering.
    #
    # The writer cannot produce an inverted window: ``completed_at_utc`` is
    # derived as the start plus a non-negative monotonic duration rather than
    # read from the wall clock a second time (see
    # ``runner.EpisodeResult.completed_at_utc``). The check stays because this
    # reader's job is to accept only rows that are internally consistent,
    # whoever or whatever wrote them.
    try:
        started_at_utc = check_utc_timestamp(raw["started_at_utc"], "started_at_utc")
        completed_at_utc = check_utc_timestamp(
            raw["completed_at_utc"], "completed_at_utc"
        )
    except RunManifestError as exc:
        raise LedgerError(f"{context}: {exc}") from exc
    if completed_at_utc < started_at_utc:
        raise LedgerError(
            f"{context}: the episode is recorded as completing at {completed_at_utc} "
            f"and starting at {started_at_utc}; an episode does not finish before it "
            "begins"
        )
    elapsed_seconds = _optional_amount(raw["elapsed_seconds"], "elapsed_seconds", context)
    if elapsed_seconds is None:
        raise LedgerError(
            f"{context}: 'elapsed_seconds' must be a non-negative finite number; the "
            "runner measures every episode's duration locally, so there is no such "
            "thing as an episode whose duration went unmeasured"
        )

    usage = _usage(_mapping(raw, "usage", context), context)
    # Read with the same rule as every other amount: null is "not accounted for"
    # and zero is "accounted for, and none". A row whose exposure is null in a
    # capped run would let a resume recover less spend than the run had, which
    # is the one direction a cost cap cannot survive — so the runner writes the
    # zero and this reads it back as one.
    cost_exposure_usd = _optional_amount(
        raw["cost_exposure_usd"], "cost_exposure_usd", context
    )
    controls = manifest.cost_controls
    # Rebuilt from the run's own pinned artefacts and this row's replayed
    # trajectory, and only then compared with what the row says it reserved.
    reservations = (
        None
        if projection is None
        else _turn_reservations(projection, variant, trajectory, turns_used, context)
    )
    telemetry = _provider_telemetry(
        raw["provider_telemetry"],
        turns_used,
        context,
        reservations,
        manifest,
        # The same condition that makes a reservation rederivable makes the turn
        # accounting exact, and for the same reason: this is a run whose requests
        # this build can rebuild because it is the adapter that sent them, and
        # that adapter reports what every turn asked of the provider.
        projection is not None,
    )
    measurement = _measurement(raw["measurement"], telemetry, turns_used, context)
    # Both money fields, re-derived rather than read: see ``_check_row_cost``.
    _check_row_cost(outcome, controls, usage, telemetry, cost_exposure_usd, context)
    adapter = _mapping(raw, "adapter", context)
    _exact_keys(adapter, _ADAPTER_KEYS, f"{context}: adapter")
    if adapter != manifest.adapter.as_dict():
        raise LedgerError(
            f"{context}: adapter {dict(adapter)} does not match this run's adapter"
        )
    contracts = _mapping(raw, "contract_versions", context)
    _exact_keys(contracts, _CONTRACT_KEYS, f"{context}: contract_versions")
    if contracts != dict(manifest.contract_versions):
        raise LedgerError(
            f"{context}: contract_versions {dict(contracts)} do not match this run's "
            f"{dict(manifest.contract_versions)}; the row was produced by a "
            "different runner, evaluator or adapter contract and is not evidence "
            "about this run"
        )

    return EpisodeRecord(
        schema_version=version,
        configuration_id=configuration_id,
        execution_id=execution_id,
        episode_id=episode_id,
        variant_id=variant_id,
        trial_index=trial_index,
        outcome=outcome,
        sample_status=sample_status,
        error_class=error_class,
        error_detail=error_detail,
        failure_event=failure_event,
        turns_used=turns_used,
        messages_used=messages_used,
        started_at_utc=started_at_utc,
        completed_at_utc=completed_at_utc,
        elapsed_seconds=elapsed_seconds,
        trajectory=_mapping(raw, "trajectory", context),
        evaluation=evaluation,
        usage=usage,
        provider_telemetry=telemetry,
        measurement=measurement,
        cost_exposure_usd=cost_exposure_usd,
        suite_content_digest=_digest(
            raw, "suite_content_digest", manifest.suite_content_digest, context
        ),
        scaffold_content_digest=_digest(
            raw, "scaffold_content_digest", manifest.scaffold_content_digest, context
        ),
        variant_content_digest=variant_digest,
        adapter=adapter,
        contract_versions=contracts,
    )


def _read_bytes_nofollow(path: Path) -> bytes | None:
    """Read the ledger without following a link that was swapped in for it.

    ``None`` when there is no ledger yet. Opening by descriptor with
    ``O_NOFOLLOW`` closes the window in which a validated regular file is
    replaced by a symlink to something else between the check and the read.
    """
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise LedgerIOError(
            f"cannot read ledger {path}: {exc}. A ledger is a regular file written "
            "in place; it is never read through a link."
        ) from exc
    try:
        with os.fdopen(descriptor, "rb") as handle:
            return handle.read()
    except OSError as exc:  # pragma: no cover - unreadable after opening
        raise LedgerIOError(f"cannot read ledger {path}: {exc}") from exc


def read_ledger(
    path: str | Path,
    manifest: RunManifest,
    variants: Mapping[str, Variant],
    *,
    scaffold: Scaffold | None = None,
) -> tuple[EpisodeRecord, ...]:
    """Parse and validate every row against the run it claims to belong to.

    ``variants`` is the compiled suite this run is bound to. It is required, not
    optional: without it a row's ``variant_content_digest`` can only be checked
    for *shape*, and a well-formed digest of nothing in particular passes. With
    it, every row is tied to the exact compiled variant the plan named, its
    trajectory is rebuilt and its verdict is recomputed.

    ``scaffold`` is the scaffold this run pinned, and it only matters for a run
    with a cost cap: what each attempt reserved is rederived from the request
    its turn actually sent, and the scaffold is part of that request. A caller
    holding the run's scaffold passes it; one that does not falls back to the
    scaffold this build ships. Either object has to hash to the digest the
    manifest pins, so it is the manifest that decides what the requests are
    rebuilt from — see :func:`_projection_scaffold`.

    A missing ledger is an empty one — that is a run that has not written an
    episode yet, not a corrupt run. Every row is checked against the public JSON
    depth limit before parsing.
    """
    return _read_ledger(Path(path), manifest, variants, scaffold)


def _read_ledger(
    path: Path,
    manifest: RunManifest,
    variants: Mapping[str, Variant],
    scaffold: Scaffold | None,
) -> tuple[EpisodeRecord, ...]:
    raw = _read_bytes_nofollow(path)
    if raw is None or not raw:
        return ()
    if not raw.endswith(b"\n"):
        raise LedgerCorruptionError(
            f"{path}: the final row does not end with a newline, so it was only "
            "partly written. The run was interrupted mid-append; this ledger is "
            "not repaired automatically."
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LedgerCorruptionError(f"{path} is not valid UTF-8: {exc}") from exc

    # Once per file, not once per row: what a capped run's requests are rebuilt
    # from is a property of the run, and a ledger this build cannot rederive the
    # reservations of is refused before any row is read as evidence.
    projection = _request_projection(manifest, scaffold)

    records: list[EpisodeRecord] = []
    seen: set[str] = set()
    planned = manifest.planned_episode_ids
    for number, line in enumerate(text.splitlines(), start=1):
        context = f"{path}: line {number}"
        if not line.strip():
            raise LedgerCorruptionError(f"{context} is blank; every row is one episode")
        try:
            ensure_raw_json_depth(line, context)
            parsed = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_constant,
            )
        except NestingDepthError as exc:
            raise LedgerCorruptionError(str(exc)) from exc
        except json.JSONDecodeError as exc:
            raise LedgerCorruptionError(f"{context} is not valid JSON: {exc}") from exc
        except LedgerCorruptionError as exc:
            raise LedgerCorruptionError(f"{context}: {exc}") from exc
        record = _parse_row(parsed, manifest, variants, projection, context)
        if record.episode_id in seen:
            raise LedgerError(
                f"{context}: duplicate episode {record.episode_id!r}; an episode is "
                "recorded exactly once"
            )
        # The ledger is written in plan order and appended to, so what is on disk
        # must be an exact prefix of the plan. Accepting any subset would let a
        # resume fill a gap in the middle and produce a file whose order no
        # longer reproduces the run it describes.
        if number > len(planned) or record.episode_id != planned[number - 1]:
            expected = planned[number - 1] if number <= len(planned) else "no episode"
            raise LedgerError(
                f"{context}: episode {record.episode_id!r} is out of plan order; the "
                f"ledger must be an exact prefix of the episode plan, which has "
                f"{expected!r} in this position"
            )
        seen.add(record.episode_id)
        records.append(record)
    return tuple(records)


__all__ = [
    "FAILURE_CLASSIFICATIONS",
    "KIND_ACTION_REJECTED",
    "KIND_ADAPTER_EXCEPTION",
    "KIND_COST_CAP_EXHAUSTED",
    "KIND_COST_RESERVATION_BREACHED",
    "KIND_ENVIRONMENT_EXCEPTION",
    "KIND_EPISODE_TIMEOUT",
    "KIND_EVALUATOR_EXCEPTION",
    "KIND_INCOMPLETE_SCAFFOLD",
    "KIND_MALFORMED_ARGUMENTS",
    "KIND_MALFORMED_RESPONSE",
    "KIND_MALFORMED_USAGE",
    "KIND_MAX_MESSAGES",
    "KIND_MAX_TURNS",
    "KIND_OUTPUT_TOKEN_LIMIT",
    "KIND_PROTOCOL_ERROR",
    "KIND_PROVIDER_AUTHENTICATION",
    "KIND_PROVIDER_CONFIGURATION",
    "KIND_PROVIDER_NETWORK_ERROR",
    "KIND_PROVIDER_RATE_LIMITED",
    "KIND_PROVIDER_REQUEST_REJECTED",
    "KIND_PROVIDER_RESPONSE_INVALID",
    "KIND_PROVIDER_SERVER_ERROR",
    "KIND_PROVIDER_TIMEOUT",
    "KIND_RUNNER_EXCEPTION",
    "KIND_UNKNOWN_ACTION",
    "KIND_UNREPRESENTABLE_TEXT",
    "KIND_USAGE_EXCEPTION",
    "KIND_WRONG_CHANNEL",
    "LEDGER_INTEGRITY_NOTE",
    "LEDGER_SCHEMA_VERSION",
    "MEASUREMENT_FULL",
    "MEASUREMENT_PARTIAL",
    "MEASUREMENT_STATUSES",
    "MEASUREMENT_UNMEASURED",
    "MESSAGES_PER_TURN",
    "OUTCOMES",
    "OUTCOME_ADAPTER_FAILURE",
    "OUTCOME_COST_CAP_REACHED",
    "OUTCOME_COST_RESERVATION_BREACHED",
    "OUTCOME_ENVIRONMENT_FAILURE",
    "OUTCOME_EVALUATOR_FAILURE",
    "OUTCOME_LIMIT_EXHAUSTED",
    "OUTCOME_MODEL_ACTION_FAILURE",
    "OUTCOME_MODEL_OUTPUT_LIMIT",
    "OUTCOME_MODEL_PROTOCOL_FAILURE",
    "OUTCOME_PROVIDER_AUTH_FAILURE",
    "OUTCOME_PROVIDER_CONFIGURATION_FAILURE",
    "OUTCOME_PROVIDER_FAILURE",
    "OUTCOME_PROVIDER_RATE_LIMITED",
    "OUTCOME_PROVIDER_REQUEST_FAILURE",
    "OUTCOME_RUNNER_FAILURE",
    "OUTCOME_SCAFFOLD_FAILURE",
    "OUTCOME_SUCCESS",
    "OUTCOME_TIMEOUT",
    "PHASE_ADAPTER_CALL",
    "PHASE_DISPATCH",
    "PHASE_EVALUATION",
    "PHASE_RESPONSE_VALIDATION",
    "PHASE_RUNNER",
    "PHASE_SCAFFOLD_CONTRACT",
    "PHASE_TURN_START",
    "PHASE_USAGE",
    "RUN_TERMINAL_OUTCOMES",
    "SAMPLE_BEHAVIORAL_ATTEMPT",
    "SAMPLE_COMPLETED",
    "SAMPLE_ELIGIBILITY_NOTE",
    "SAMPLE_INFRASTRUCTURE_INELIGIBLE",
    "SAMPLE_STATUSES",
    "SAMPLE_STATUS_BY_OUTCOME",
    "EpisodeRecord",
    "FailureSpec",
    "LedgerCorruptionError",
    "LedgerError",
    "LedgerIOError",
    "append_episode",
    "classify_failure",
    "derive_authorised_measured_usd",
    "derive_cost_exposure_usd",
    "derive_measured_cost_usd",
    "derive_measurement",
    "measurement_status_for",
    "read_ledger",
    "sample_status_for",
    "unmatched_query_counts",
    "unmatched_query_events",
]
