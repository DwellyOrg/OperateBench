"""The tool loop: one episode, executed for real and graded deterministically.

Nothing here synthesises a trajectory. Every action an adapter names is passed
to the real :class:`~boundarybench.environment.Environment`, which is the only
thing that mints observation handles and records mutations, and the resulting
trajectory is graded by the real :func:`~boundarybench.evaluator.evaluate`. A
summary the adapter offers about what it did is never trusted, because a summary
is exactly the thing an agent can be wrong about.

Every way an episode can end is classified. The distinction that matters most is
between a *model/protocol* failure — a well-behaved adapter naming a call the
scaffold does not define — and an *adapter* failure, which is the integration or
the provider breaking. Conflating them would make a flaky network look like a
model that cannot follow a schema.

There are no retries. A failed episode is recorded as failed, once. Retrying
silently would turn one provider hiccup into a second sample of the same
episode, which is not a thing this design can measure.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import Any

from boundarybench.adapter import (
    ADAPTER_BOUNDARY_LAST_USAGE,
    ADAPTER_BOUNDARY_NEXT_CALL,
    AdapterCall,
    AdapterOutputLimitError,
    AdapterProtocolError,
    AdapterProviderError,
    AdapterUsage,
    ModelAdapter,
    ProviderTelemetry,
    TurnDeadline,
    build_turn_request,
    redacted_adapter_detail,
)
from boundarybench.budget import (
    COST_ACCOUNTING_NOTE,
    BudgetError,
    CostCapExceededError,
    CostReservationBreachedError,
    RunCostGuard,
    usd_text,
)
from boundarybench.compiler import Variant, variant_content_digest
from boundarybench.environment import (
    Environment,
    EnvironmentError,
    channel_for,
)
from boundarybench.evaluator import VariantEvaluation, evaluate
from boundarybench.freezing import to_json
from boundarybench.jsonsafe import JsonSafetyError, check_text, ensure_json_safe
from boundarybench.ledger import (
    KIND_ACTION_REJECTED,
    KIND_ADAPTER_EXCEPTION,
    KIND_COST_CAP_EXHAUSTED,
    KIND_COST_RESERVATION_BREACHED,
    KIND_ENVIRONMENT_EXCEPTION,
    KIND_EPISODE_TIMEOUT,
    KIND_EVALUATOR_EXCEPTION,
    KIND_INCOMPLETE_SCAFFOLD,
    KIND_MALFORMED_ARGUMENTS,
    KIND_MALFORMED_RESPONSE,
    KIND_MALFORMED_USAGE,
    KIND_MAX_MESSAGES,
    KIND_MAX_TURNS,
    KIND_OUTPUT_TOKEN_LIMIT,
    KIND_PROTOCOL_ERROR,
    KIND_RUNNER_EXCEPTION,
    KIND_UNKNOWN_ACTION,
    KIND_UNREPRESENTABLE_TEXT,
    KIND_USAGE_EXCEPTION,
    KIND_WRONG_CHANNEL,
    LEDGER_INTEGRITY_NOTE,
    MEASUREMENT_STATUSES,
    MESSAGES_PER_TURN,
    OUTCOME_SUCCESS,
    PHASE_ADAPTER_CALL,
    PHASE_DISPATCH,
    PHASE_EVALUATION,
    PHASE_RESPONSE_VALIDATION,
    PHASE_RUNNER,
    PHASE_SCAFFOLD_CONTRACT,
    PHASE_TURN_START,
    PHASE_USAGE,
    RUN_TERMINAL_OUTCOMES,
    SAMPLE_COMPLETED,
    SAMPLE_ELIGIBILITY_NOTE,
    SAMPLE_STATUSES,
    EpisodeRecord,
    append_episode,
    classify_failure,
    derive_measurement,
    measurement_status_for,
    read_ledger,
    sample_status_for,
)
from boundarybench.runmanifest import (
    BUILD_CONTRACT_VERSIONS,
    RETRY_POLICY,
    RUN_PLAN_NOTE,
    EpisodePlanEntry,
    RunLimits,
    RunManifest,
    RunManifestError,
    RunPaths,
    RunSession,
    WallClock,
    format_utc_timestamp,
    normalized_adapter_settings,
    run_track_for_provider,
    utc_now,
    verify_manifest_identity,
)
from boundarybench.scaffold import (
    WORKFLOW_ACTION_TEMPLATE,
    ActionSchema,
    Scaffold,
    scaffold_content_digest,
)
from boundarybench.suite import (
    SuiteError,
    SuiteReport,
    enforce_suite_constraints,
    suite_content_digest,
)
from boundarybench.trajectory import Trajectory

#: A monotonic clock in seconds. Injected so the timeout path is deterministic.
Clock = Callable[[], float]

READ_ACTION = "read_records"
TERMINAL_ACTION = "complete_case"
#: The template every case's own workflow actions are minted from. Never offered
#: under this name — see :func:`~boundarybench.scaffold.project_action_surface`.
WORKFLOW_ACTION = WORKFLOW_ACTION_TEMPLATE
ELICITATION_ACTIONS: tuple[str, ...] = ("ask_user", "call_tool")

#: The four measurements an adapter may report per turn, in report order.
USAGE_FIELDS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "latency_seconds",
)

#: Said in every report, because a missing measurement and a zero one are not
#: the same claim and a total silently conflating them is worse than no total.
USAGE_NOTE = (
    "Usage totals sum only the episodes that reported a measurement. A null "
    "total means no episode measured that field — a fake reports none of them — "
    "and is never rendered as zero. Where some episodes measured and others did "
    "not, the total is a real sum over the measured subset and the measured and "
    "unmeasured episode counts state which subset that was."
)

#: Said in every report, because "measured" was one word doing three jobs.
MEASUREMENT_NOTE = (
    "Measurement coverage is reported at three resolutions — episodes, turns and "
    "provider attempts — because a figure at one hides the others: twelve "
    "episodes each measuring one turn of four is 12/12 by episode and a quarter "
    "of the run by turn. An episode is 'full' when every turn it spent reported "
    "a local measurement, 'partial' when some did, and 'unmeasured' when none "
    "did. A null total means nothing measured that quantity and is never "
    "rendered as zero; a zero means something counted and found none. An "
    "in-process fake contacts no provider, so its attempt totals are null rather "
    "than 0. All latency here is this build's own local wall-clock, never the "
    "provider's server-side processing time, which nothing here can observe."
)

#: Said whenever a run stopped early, in the report of the run that stopped and
#: of every later read of it.
RUN_TERMINATED_NOTE = (
    "This run met a failure that condemns its whole configuration — a refused "
    "credential, or a provider that cannot find the model or resource this run "
    "pins — and stopped after recording the first one durably, rather than "
    "repeating an identical failure for every remaining planned episode. Only "
    "those two stop a run, because only they are proven to hold for every "
    "remaining episode: one credential and one pinned model are sent unchanged "
    "on every turn. A rejected request is not one of them and never stops the "
    "plan. The unexecuted episodes have no rows and are not samples of anything. "
    "Whatever fixes the failure is deliberately outside configuration identity, "
    "so this directory can never continue: use a new output directory, which "
    "mints a new execution and leaves this evidence untouched."
)

#: Said whenever an invocation stopped short of its plan because a hard limit
#: was reached rather than because anything failed.
LIMITS_STOP_NOTE = (
    "This invocation stopped before the end of its plan because it reached a "
    "hard limit this run was authorised under, not because anything failed. The "
    "unexecuted episodes have no rows and are not samples of anything. Raising "
    "the limit changes configuration identity, so the continuation is a new run "
    "in a new directory and this evidence is left exactly as it is."
)

#: Said in every report, whether or not an invocation was staged, because the
#: one thing a staged report must never be read as is a run configured to be
#: smaller than it is.
INVOCATION_STAGING_NOTE = (
    "An invocation limit bounds how many *new* episodes this one invocation "
    "executes, and nothing else. It is operational staging, not a run setting: "
    "it is not written to the manifest, it is not part of configuration or "
    "execution identity, and it changes neither the episode plan, the caps, the "
    "pricing nor any episode's identity. It can only stop an invocation earlier "
    "than the plan; it can never authorise an episode or a dollar this run's own "
    "durable limits refuse, and a cost, episode-ceiling or run-terminal stop "
    "takes precedence over it. The episodes it leaves unexecuted have no rows "
    "and are not samples of anything. Resuming the same manifest in the same "
    "directory continues from the next planned episode, so the completed run is "
    "the same run this one is a stage of."
)

#: Said in every report, because the two numbers are routinely confused.
COUNTS_NOTE = (
    "'succeeded' and 'failed' are execution outcomes: whether the loop reached a "
    "terminal decision through the real environment. 'verdict_passed' and "
    "'verdict_failed' are the deterministic evaluator's judgement of whether that "
    "decision was correct. An episode can complete successfully and still be "
    "graded wrong, so the two are never added together and neither is a score."
)

#: What each protocol action must offer for the loop to be able to dispatch it:
#: the argument name, the type it must be declared as, and the fact that it is
#: required. Names alone are not the contract — an argument the loop always
#: sends but the scaffold declares optional, or declares as an array when the
#: loop sends a string, is a scaffold that cannot express this protocol even
#: though every name matches.
#:
#: :data:`WORKFLOW_ACTION` is the one entry whose argument is not sent. It is a
#: template rather than an offer, and its ``action`` argument is what
#: :func:`~boundarybench.scaffold.project_action_surface` turns into the *name*
#: of each workflow tool a case offers. It is still required to be declared, and
#: declared as exactly one required string, because a template that named its
#: subject differently is one this projection cannot read.
_REQUIRED_PARAMETERS: Mapping[str, Mapping[str, str]] = {
    READ_ACTION: {},
    "ask_user": {"fact": "string"},
    "call_tool": {"fact": "string"},
    WORKFLOW_ACTION: {"action": "string"},
    TERMINAL_ACTION: {
        "disposition": "string",
        "primary_reason_code": "string",
        "secondary_reason_codes": "array_of_string",
        "evidence_refs": "array_of_string",
    },
}


class RunnerError(RuntimeError):
    """The runner cannot proceed: infrastructure or inputs disagree."""


class RunProvenanceError(RunnerError):
    """What is about to execute is not what the manifest says will execute.

    The manifest is a *claim* about a run until something checks it against the
    objects actually in hand. Without that check a caller can execute adapter B
    under configuration B while every ledger row records adapter A, and the
    resulting evidence is false in exactly the way an audit cannot detect.
    """


class ScaffoldContractError(RuntimeError):
    """The scaffold cannot express the protocol this loop has to speak."""


class RunTerminatedError(RunnerError):
    """This run already met a failure that condemns the whole configuration.

    A refused credential, or a provider that cannot find the model this run
    pins, is not one episode's bad luck: it is a property of what the run was
    configured to do, and it will be true of every remaining episode, which is
    exactly the proof this build demands before stopping a plan. The run
    therefore recorded the first one durably and stopped. A merely *rejected
    request* does not qualify — that is about the payload one episode sent — and
    is recorded as an ineligible episode while the plan continues.

    Continuing in the same directory afterwards is refused, and the reason is
    precisely that this build's configuration identity is *correct*. A
    credential is not a configuration input — it is deliberately excluded, so it
    can never reach a manifest — so fixing one leaves ``configuration_id``
    identical and the resume check has nothing to object to. The ledger would
    then hold a single execution whose first row says the credential was refused
    and whose remaining rows say it was not, and no reader could tell that those
    rows describe two different states of the world. A new output directory
    mints a new execution, and the old evidence stays exactly where it is.
    """


@dataclass(frozen=True)
class EpisodeResult:
    """How one episode ended, and everything it produced on the way."""

    outcome: str
    error_class: str | None
    error_detail: str | None
    #: The typed cause the other three fields are derived from. ``None`` on a
    #: success, and never ``None`` on anything else.
    failure_event: Mapping[str, Any] | None
    turns_used: int
    messages_used: int
    trajectory: Trajectory
    evaluation: VariantEvaluation | None
    usage: AdapterUsage
    #: When the episode began, from the injected wall clock: a single reading,
    #: taken before anything can fail. This is what makes a run that spans a
    #: provider date boundary readable afterwards.
    started_at_utc: str = ""
    #: When the episode ended — **derived**, not observed. It is
    #: :attr:`started_at_utc` plus :attr:`elapsed_seconds`, and no second
    #: wall-clock reading is taken. A wall clock can be stepped backwards under
    #: a running episode, and a second reading of it would then record a
    #: completion before the start, which is both false and a row the ledger
    #: refuses to read back. The estimate this substitutes is exact in the only
    #: sense available locally — the offset from the recorded start is the
    #: monotonic duration actually measured — and it hides nothing: any wall
    #: clock correction that happened mid-episode lands on the *next* episode's
    #: recorded start, where it is visible, rather than inside this one's span.
    completed_at_utc: str = ""
    #: How long the episode took, from the injected monotonic clock. This is a
    #: local measurement of this build's own loop — it includes provider
    #: latency because the loop waited for it, and it is not the provider's
    #: server-side processing time, which nothing here can observe.
    elapsed_seconds: float = 0.0
    #: Turn-by-turn provider attempt evidence, or ``None`` when no turn reached
    #: a provider at all.
    provider_telemetry: Mapping[str, Any] | None = None

    @property
    def succeeded(self) -> bool:
        return self.outcome == OUTCOME_SUCCESS


def check_scaffold_contract(scaffold: Scaffold) -> None:
    """Prove the scaffold declares every call the loop needs, before running."""
    declared = {action.name: action for action in scaffold.actions}
    for name, required in _REQUIRED_PARAMETERS.items():
        action = declared.get(name)
        if action is None:
            raise ScaffoldContractError(
                f"scaffold {scaffold.scaffold_id} does not declare the {name!r} "
                "action, so this loop cannot offer it to an adapter"
            )
        parameters = {parameter.name: parameter for parameter in action.parameters}
        missing = [name_ for name_ in required if name_ not in parameters]
        if missing:
            raise ScaffoldContractError(
                f"scaffold {scaffold.scaffold_id} declares {name!r} without "
                f"parameter(s) {missing}"
            )
        for parameter_name, parameter_type in required.items():
            parameter = parameters[parameter_name]
            if parameter.type != parameter_type:
                raise ScaffoldContractError(
                    f"scaffold {scaffold.scaffold_id} declares {name!r} argument "
                    f"{parameter_name!r} as {parameter.type!r}, but this loop "
                    f"sends it as {parameter_type!r}; it must be declared as "
                    f"{parameter_type!r}"
                )
            if not parameter.required:
                raise ScaffoldContractError(
                    f"scaffold {scaffold.scaffold_id} declares {name!r} argument "
                    f"{parameter_name!r} as optional, but this loop always sends "
                    "it, so it must be required"
                )
        unsupplied = sorted(
            parameter.name
            for parameter in action.parameters
            if parameter.required and parameter.name not in required
        )
        if unsupplied:
            raise ScaffoldContractError(
                f"scaffold {scaffold.scaffold_id} requires argument(s) {unsupplied} "
                f"for {name!r} that this loop never supplies, so every call to it "
                "would be rejected"
            )
    terminal = scaffold.terminal_action
    if terminal.name != TERMINAL_ACTION:
        raise ScaffoldContractError(
            f"scaffold {scaffold.scaffold_id} marks {terminal.name!r} as the "
            f"terminal action, but this loop terminates with {TERMINAL_ACTION!r}"
        )


def _argument_failure(action: ActionSchema, arguments: Mapping[str, Any]) -> str | None:
    """Check the adapter's arguments against the schema it was shown."""
    declared = {parameter.name: parameter for parameter in action.parameters}
    missing = [
        name
        for name, parameter in declared.items()
        if parameter.required and name not in arguments
    ]
    if missing:
        return f"call to {action.name!r} is missing required argument(s) {missing}"
    unknown = sorted(set(arguments) - set(declared))
    if unknown:
        return f"call to {action.name!r} carries unknown argument(s) {unknown}"
    for name, value in arguments.items():
        parameter = declared[name]
        if parameter.type == "string" and not isinstance(value, str):
            return (
                f"call to {action.name!r}: argument {name!r} must be a string, got "
                f"{type(value).__name__}"
            )
        # The earliest boundary that can decide this, and a generic one: the
        # closed set was in the schema the model was shown, so a value outside it
        # is an argument failure and not a choice the environment has to refuse
        # from information the model never had. That inversion is the defect
        # ``FACT_AFFORDANCE_CONTRACT`` exists to close.
        #
        # The keys are named because they are the case's own offered interface.
        # The value the model produced is not: it is model-authored text, this
        # detail becomes a durable row, and the refused call is already recorded
        # verbatim as ``attempted_call``.
        if parameter.enum is not None and value not in parameter.enum:
            return (
                f"call to {action.name!r}: argument {name!r} must be one of the "
                f"{len(parameter.enum)} value(s) this case offers, "
                f"{list(parameter.enum)}; the value the call carried is not quoted "
                "here, and is recorded as the attempted call"
            )
        if parameter.type == "array_of_string" and (
            isinstance(value, (str, bytes))
            or not isinstance(value, list)
            or not all(isinstance(item, str) for item in value)
        ):
            return (
                f"call to {action.name!r}: argument {name!r} must be an array of strings"
            )
    return None


def _is_json_safe(value: Any) -> bool:
    """Whether a value survives being written to a row and read back unchanged.

    The same question :mod:`boundarybench.jsonsafe` answers by raising; asked
    here as a predicate because the caller records ``arguments: null`` rather
    than failing when the answer is no.
    """
    try:
        ensure_json_safe(value, "attempted call arguments")
    except JsonSafetyError:
        return False
    return True


def _attempted_call(call: AdapterCall) -> dict[str, Any]:
    """The refused call, in a form the row can carry back verbatim.

    ``arguments`` is null when the adapter did not offer a JSON object — which
    is one of the faults recorded this way, so it has to be sayable. Projecting
    it as ``{}`` would record an empty call the adapter never made, and letting
    a non-JSON value through would write a row the ledger then refuses to read.
    """
    arguments = call.arguments
    safe = isinstance(arguments, Mapping) and _is_json_safe(arguments)
    return {"action": call.action, "arguments": to_json(arguments) if safe else None}


class _EpisodeFailure(Exception):
    """Internal control flow: end the episode with this classification.

    The classification is not passed in — the phase, the kind and the evidence
    are, and :func:`~boundarybench.ledger.classify_failure` derives the rest.
    Fifteen call sites each naming their own outcome and error class is exactly
    how the summary came adrift from the cause.
    """

    def __init__(
        self,
        phase: str,
        kind: str,
        detail: str = "",
        *,
        exception: BaseException | None = None,
        attempted_call: Mapping[str, Any] | None = None,
        response_type: str | None = None,
    ) -> None:
        outcome, error_class, error_detail, event = classify_failure(
            phase=phase,
            kind=kind,
            detail=detail,
            exception=exception,
            attempted_call=attempted_call,
            response_type=response_type,
        )
        super().__init__(error_detail)
        self.outcome = outcome
        self.error_class = error_class
        self.detail = error_detail
        self.failure_event = event


def _dispatch(environment: Environment, call: AdapterCall) -> bool:
    """Perform one validated call against the environment. True when terminal.

    The call has already been checked against the actions this variant *offered*
    — the Cube's own ``allowed_actions`` — so every name that reaches here is one
    the case admits, and anything that is not a protocol action is one of the
    case's own workflow actions, dispatched under the name it was offered as.
    """
    arguments = call.arguments
    if call.action == READ_ACTION:
        environment.read_records()
        return False
    if call.action == TERMINAL_ACTION:
        environment.complete_case(
            disposition=str(arguments["disposition"]),
            primary_reason_code=str(arguments["primary_reason_code"]),
            secondary_reason_codes=tuple(arguments["secondary_reason_codes"]),
            evidence_refs=tuple(arguments["evidence_refs"]),
        )
        return True
    if call.action in ELICITATION_ACTIONS:
        fact = str(arguments["fact"])
        # Checked before dispatching, not after: the environment delivers a fact
        # through whichever channel the card declares, so calling it first would
        # file a step under an action the adapter never asked for.
        expected = channel_for(environment.variant, fact)
        if expected is not None and expected != call.action:
            raise _EpisodeFailure(
                PHASE_DISPATCH,
                KIND_WRONG_CHANNEL,
                f"fact {fact!r} is not available through {call.action!r}; this "
                f"case delivers it through {expected!r}",
                attempted_call=_attempted_call(call),
            )
        environment.obtain(fact)
        return False
    environment.act(call.action)
    return False


def _read_usage(adapter: ModelAdapter) -> AdapterUsage:
    """Ask the adapter what the turn cost, inside the classification boundary.

    Called from within the episode's try block so a provider whose usage
    endpoint raises is an adapter failure with a recorded class, rather than an
    exception escaping the loop entirely. It is the same trust boundary
    ``next_call`` is, so it is redacted the same way: the class is recorded and
    the exception's own message is not.
    """
    try:
        usage = adapter.last_usage()
    except Exception as exc:
        raise _EpisodeFailure(
            PHASE_USAGE,
            KIND_USAGE_EXCEPTION,
            redacted_adapter_detail(exc, boundary=ADAPTER_BOUNDARY_LAST_USAGE),
            exception=exc,
        ) from exc
    return _check_usage(usage)


def _accumulate_usage(total: AdapterUsage, turn: AdapterUsage) -> AdapterUsage:
    """Add one turn's measurement to the episode's running total.

    ``None`` means *not measured*, which is not the same as zero: a fake reports
    nothing and must not accumulate into a fabricated zero-cost episode. So a
    field stays ``None`` until some turn measures it, and from then on it sums.

    ``cost_usd`` sums through ``Decimal(str(...))`` rather than as binary floats,
    the same way :func:`recovered_spend` reads amounts back. Each turn's cost is
    an exact decimal amount rendered as the shortest float that round-trips it,
    so summing the strings is exact while summing the floats drifts in the last
    place — and a drifted total is one a reader cannot re-derive from the token
    counts beside it, which is precisely what the ledger now requires of it.
    """
    values: dict[str, Any] = {}
    for name in ("input_tokens", "output_tokens", "cost_usd", "latency_seconds"):
        running, measured = getattr(total, name), getattr(turn, name)
        if measured is None:
            values[name] = running
        elif running is None:
            values[name] = measured
        elif name == "cost_usd":
            values[name] = float(Decimal(str(running)) + Decimal(str(measured)))
        else:
            values[name] = running + measured
    return AdapterUsage(**values)


def _usage_problem(usage: Any) -> str | None:
    """Why this is not a measurement, or ``None`` when it is one.

    Stated as a question rather than as a refusal because it has two callers
    that need different answers to it. On a healthy turn, a measurement that is
    not one *is* the episode's failure. On a turn that has already failed, it
    means the turn measured nothing — see :func:`_measured_on_failure`.
    """
    if not isinstance(usage, AdapterUsage):
        return f"last_usage() returned {type(usage).__name__}, not an AdapterUsage"
    for name in ("input_tokens", "output_tokens"):
        value = getattr(usage, name)
        if value is None:
            continue
        # bool is an int subclass, so `True` must not be recorded as one token.
        if type(value) is not int or value < 0:
            return (
                f"last_usage().{name} must be a non-negative integer or None, got "
                f"{value!r}"
            )
    for name in ("cost_usd", "latency_seconds"):
        value = getattr(usage, name)
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            return (
                f"last_usage().{name} must be a non-negative finite number or None, "
                f"got {value!r}"
            )
    return None


def _check_usage(usage: Any) -> AdapterUsage:
    """Refuse a measurement that is not one, rather than persisting it."""
    problem = _usage_problem(usage)
    if problem is not None:
        raise _EpisodeFailure(PHASE_USAGE, KIND_MALFORMED_USAGE, problem)
    assert isinstance(usage, AdapterUsage)
    return usage


def _measured_on_failure(adapter: ModelAdapter, total: AdapterUsage) -> AdapterUsage:
    """Bank what a *failed* turn consumed, and never turn that into the failure.

    A provider response that arrived was paid for whether or not this build
    could read it as one action. A text-only answer to a tool-choice request is
    a model protocol failure and it is also 91 input and 92 output tokens off
    somebody's account, so an episode that reported the failure and omitted the
    spend was under-reporting real consumption on every protocol-invalid turn.

    The precedence is fixed and one-directional: **the failure that ended the
    turn always wins.** This runs while an ``_EpisodeFailure`` is already being
    raised, so it cannot raise one of its own — an adapter whose usage endpoint
    breaks while reporting a protocol failure would otherwise have the protocol
    failure silently rewritten into an adapter failure, which is the wrong
    outcome, the wrong error class and the wrong story about what happened. A
    usage endpoint that raises, returns something that is not an
    :class:`~boundarybench.adapter.AdapterUsage`, or reports a count that cannot
    be one therefore leaves the turn measuring *nothing*, which is the honest
    answer: null is "not measured", and it is not zero.

    The healthy path is unchanged — see :func:`_read_usage`, where an
    unmeasurable turn is still the episode's failure, because there is no other
    fault for it to defer to.
    """
    try:
        usage = adapter.last_usage()
    except Exception:
        # Deliberately swallowed, and deliberately not recorded: the row already
        # carries the fault that ended the turn, and a second, softer failure
        # has nowhere in the taxonomy to go without displacing the first.
        return total
    if _usage_problem(usage) is not None:
        return total
    return _accumulate_usage(total, usage)


def _turn_telemetry(adapter: ModelAdapter, turn_index: int) -> dict[str, Any] | None:
    """Bank what the turn asked of the provider, and never let that be the failure.

    The precedence is the same one-directional rule
    :func:`_measured_on_failure` obeys, and for the same reason: this runs on
    every classified failure path, including while an ``_EpisodeFailure`` is
    already on its way up, so it may not raise one of its own. An adapter whose
    telemetry accessor is missing, raises, or answers with something that is not
    a :class:`~boundarybench.adapter.ProviderTelemetry` therefore contributes
    *no* evidence for that turn rather than replacing the fault that ended it.
    Losing the attempt evidence for one turn is a smaller error than filing a
    rate limit as an adapter failure.

    Deliberately swallowed and deliberately not recorded as a second fault: the
    row already carries the fault that ended the turn, and a softer failure has
    nowhere in the taxonomy to go without displacing the first.
    """
    try:
        telemetry = adapter.last_telemetry()
    except Exception:
        return None
    if not isinstance(telemetry, ProviderTelemetry):
        # ``None`` is the contract's own answer for "this adapter reaches no
        # provider", so it lands here alongside a broken implementation and
        # produces the same, honest result: nothing is claimed about this turn.
        return None
    return {"turn_index": turn_index, **telemetry.as_dict()}


def _episode_telemetry(
    turns: Sequence[Mapping[str, Any]], turns_used: int
) -> dict[str, Any] | None:
    """Fold the banked turns into the episode's own evidence, or say there is none.

    ``None`` rather than an empty structure. An episode with no provider turns
    made no provider requests, and recording zero attempts would claim it tried.
    """
    if not turns:
        return None
    faults: dict[str, int] = {}
    for turn in turns:
        for fault, count in turn["fault_counts"].items():
            faults[fault] = faults.get(fault, 0) + count
    measured = sum(1 for turn in turns if turn["turn_latency_seconds"] is not None)
    return {
        "turns": [dict(turn) for turn in turns],
        "attempt_count": sum(int(turn["attempt_count"]) for turn in turns),
        "fault_counts": dict(sorted(faults.items())),
        "measurement_status": measurement_status_for(turns_used, measured),
    }


def run_episode(
    *,
    variant: Variant,
    scaffold: Scaffold,
    adapter: ModelAdapter,
    limits: RunLimits,
    clock: Clock = time.monotonic,
    wall_clock: WallClock = utc_now,
) -> EpisodeResult:
    """Run one episode to a terminal decision, a limit or a classified failure.

    Two clocks, because they answer two questions and neither can answer the
    other's. ``wall_clock`` says *when* the episode ran, which is what a resume
    spanning a provider date boundary needs and what no monotonic reading can
    supply. ``clock`` is monotonic and says *how long* it took, which is the only
    duration a local process can measure honestly — a wall clock can be stepped
    backwards under a running episode. Both are injected so a test states the
    record rather than observing whatever the machine happened to do.

    The wall clock is read exactly once, at the start, and the completion
    timestamp is *derived* from that reading plus the monotonic elapsed time.
    Reading it a second time made the two recorded instants observations of a
    quantity that can move between them: an NTP correction mid-episode wrote
    ``completed_at_utc < started_at_utc``, and the ledger — correctly — refused
    to read a row saying an episode finished before it began. Deriving the end
    keeps every row readable under a clock correction without hiding the
    duration, which is still the measured one; see
    :attr:`EpisodeResult.completed_at_utc`.
    """
    environment = Environment(variant)
    turns_used = 0
    messages_used = 0
    usage = AdapterUsage()
    telemetry_turns: list[Mapping[str, Any]] = []

    def bank_telemetry() -> None:
        """Record this turn's provider evidence, whatever happens next."""
        banked = _turn_telemetry(adapter, turns_used)
        if banked is not None:
            telemetry_turns.append(banked)

    outcome = OUTCOME_SUCCESS
    error_class: str | None = None
    error_detail: str | None = None
    failure_event: Mapping[str, Any] | None = None
    # One reading of each clock, taken before anything can fail. The wall-clock
    # moment is kept, not just its rendering, because the completion timestamp
    # is derived from it below. Two monotonic readings would be two definitions
    # of when the episode started, and the timeout arithmetic and the recorded
    # duration would then describe slightly different episodes.
    started_moment = wall_clock()
    started_at_utc = format_utc_timestamp(started_moment)
    started = clock()

    try:
        check_scaffold_contract(scaffold)

        def elapsed() -> float:
            return clock() - started

        def remaining() -> float:
            return limits.episode_timeout_seconds - elapsed()

        while True:
            # ``>=`` throughout, matching TurnDeadline.cancelled(), which reports
            # cancellation at ``remaining() <= 0``. With ``>`` the two disagreed
            # exactly on the deadline: the adapter was told it had been cancelled
            # while the loop still counted the turn as in budget.
            if elapsed() >= limits.episode_timeout_seconds:
                raise _EpisodeFailure(
                    PHASE_TURN_START,
                    KIND_EPISODE_TIMEOUT,
                    f"episode exceeded its {limits.episode_timeout_seconds} second "
                    f"budget after {turns_used} turn(s)",
                )
            if turns_used >= limits.max_turns:
                raise _EpisodeFailure(
                    PHASE_TURN_START,
                    KIND_MAX_TURNS,
                    f"episode reached its limit of {limits.max_turns} turn(s) "
                    "without a terminal decision",
                )
            # One turn is one request out and one response back, so it costs two
            # messages. A turn that cannot be paid for in full is not started:
            # dispatching a request the budget cannot answer would leave the
            # episode holding a message it may never receive.
            if messages_used + MESSAGES_PER_TURN > limits.max_messages:
                raise _EpisodeFailure(
                    PHASE_TURN_START,
                    KIND_MAX_MESSAGES,
                    f"episode reached its limit of {limits.max_messages} message(s) "
                    f"after {messages_used} message(s) and {turns_used} turn(s) "
                    "without a terminal decision",
                )

            request = build_turn_request(
                scaffold=scaffold,
                environment=environment,
                turns_remaining=limits.max_turns - turns_used,
            )
            # What the *turn* offered, not what the scaffold could express. A
            # model that names an action this case withholds must not be looked
            # up and dispatched into the
            # environment, which then refused it for a reason the model had no
            # way to see. An action that was never offered is a protocol
            # failure, and is one before the environment is touched.
            declared = {action.name: action for action in request.actions}
            budget = remaining()
            deadline = TurnDeadline(
                remaining_seconds=budget,
                cancelled=lambda: remaining() <= 0.0,
            )
            # A turn is one *attempted* adapter call, so it is counted before the
            # call is made and the request that starts it is counted with it. A
            # call that raises leaves the request unanswered, yielding the one
            # valid shape with an odd message count. Counting after the call would
            # violate the ledger invariant that every request belongs to a turn.
            turns_used += 1
            messages_used += 1
            # Every classified failure path below banks what the turn consumed
            # before it raises. A response that arrived was paid for even when
            # this build refuses it, and the adapter contract requires the
            # measurement to have been cleared at the top of the call, so a
            # failure that happened before any response measures nothing rather
            # than inheriting the previous turn's numbers.
            # Every path out of the call below banks the turn's attempt evidence
            # as well as its consumption, on exactly the same rule: what the turn
            # *did* is an observation, and a turn that failed is the turn whose
            # attempts most need recording, because nothing else can be learned
            # from it.
            try:
                call = adapter.next_call(request, deadline)
            except CostCapExceededError as exc:
                # Checked before every other adapter failure, because it is not
                # one: no request was sent, nothing was asked of the provider,
                # and the detail is entirely this build's own arithmetic over
                # its own pinned settings. Telemetry is still banked — a turn
                # refused on its third retry made two real attempts, and those
                # are the evidence of what the budget was spent on. The usage
                # measurement is deliberately not re-read: the guard refused
                # before the call, so there is nothing new to measure and the
                # attempts that did happen were banked when they happened.
                bank_telemetry()
                raise _EpisodeFailure(
                    PHASE_ADAPTER_CALL, KIND_COST_CAP_EXHAUSTED, str(exc)
                ) from exc
            except CostReservationBreachedError as exc:
                # The opposite shape to the refusal above: here the request *was*
                # made and answered, and what it measured is beyond the bound
                # that authorised it. The measurement is banked first and by the
                # ordinary path — the tokens were consumed and the row has to say
                # so — and only then is the turn classified as a breach. Its own
                # kind rather than a cost-cap exhaustion, because the two are
                # different facts: one is a request this run refused to make, and
                # this is spend it did not authorise and cannot take back.
                usage = _measured_on_failure(adapter, usage)
                bank_telemetry()
                raise _EpisodeFailure(
                    PHASE_ADAPTER_CALL, KIND_COST_RESERVATION_BREACHED, str(exc)
                ) from exc
            except AdapterProviderError as exc:
                # The *service* failed, not this build. Classified by the fault
                # the integration named rather than by the exception's type: the
                # type is always this one class, while the fault says whether an
                # operator has a wrong credential, a wrong model id, a rate limit
                # or an outage. The detail is the integration's own redacted
                # sentence and is recorded verbatim — see AdapterProviderError.
                usage = _measured_on_failure(adapter, usage)
                bank_telemetry()
                raise _EpisodeFailure(PHASE_ADAPTER_CALL, exc.fault, str(exc)) from exc
            except AdapterOutputLimitError as exc:
                # Checked before the protocol error it is not a subclass of. The
                # answer was cut off by this run's own pinned output ceiling, so
                # it is an output-budget failure and not a model that cannot
                # follow a schema; the tokens that produced the fragment were
                # still spent, and are banked like any other arrived response's.
                usage = _measured_on_failure(adapter, usage)
                bank_telemetry()
                raise _EpisodeFailure(
                    PHASE_ADAPTER_CALL, KIND_OUTPUT_TOKEN_LIMIT, str(exc), exception=exc
                ) from exc
            except AdapterProtocolError as exc:
                usage = _measured_on_failure(adapter, usage)
                bank_telemetry()
                raise _EpisodeFailure(
                    PHASE_ADAPTER_CALL, KIND_PROTOCOL_ERROR, str(exc), exception=exc
                ) from exc
            except Exception as exc:
                # Anything this contract does not classify. Recorded by class,
                # never by message: an SDK can build an exception message from
                # arbitrary external text, which must not reach a durable row.
                # The redaction is the contract's, not this loop's, so both
                # boundary calls answer for it the same way.
                usage = _measured_on_failure(adapter, usage)
                bank_telemetry()
                raise _EpisodeFailure(
                    PHASE_ADAPTER_CALL,
                    KIND_ADAPTER_EXCEPTION,
                    redacted_adapter_detail(exc, boundary=ADAPTER_BOUNDARY_NEXT_CALL),
                    exception=exc,
                ) from exc
            bank_telemetry()
            messages_used += 1

            # Checked the instant the call returns, and before anything else the
            # adapter is asked. An adapter that blocked past the budget spent
            # time the episode did not have, so whatever it hands back — an
            # answer, or a failure from its usage endpoint — arrived too late to
            # classify the episode. Reading usage first turned an overrun into
            # whatever the *next* call to the provider happened to do.
            if elapsed() >= limits.episode_timeout_seconds:
                # The *answer* is not counted — no action is dispatched from it
                # and no verdict follows — but the call that produced it was
                # made and paid for, so what it cost is banked all the same.
                # Banked through the failure path rather than by reading usage
                # directly, so a broken usage endpoint cannot turn an overrun
                # into whatever that endpoint happened to do.
                usage = _measured_on_failure(adapter, usage)
                raise _EpisodeFailure(
                    PHASE_ADAPTER_CALL,
                    KIND_EPISODE_TIMEOUT,
                    f"the adapter call on turn {turns_used} returned after the "
                    f"episode's {limits.episode_timeout_seconds} second budget had "
                    "already elapsed; its answer is not counted",
                )

            usage = _accumulate_usage(usage, _read_usage(adapter))

            if not isinstance(call, AdapterCall) or not isinstance(call.action, str):
                raise _EpisodeFailure(
                    PHASE_RESPONSE_VALIDATION,
                    KIND_MALFORMED_RESPONSE,
                    f"adapter returned {type(call).__name__}, not one action call",
                    response_type=type(call).__name__,
                )
            # Before the call is looked up, let alone dispatched: an action name
            # or an argument holding text canonical UTF-8 JSON cannot carry is
            # not a call this build can record. It is classified as malformed
            # protocol data rather than sanitised, because the row would then
            # claim the model said something it did not.
            try:
                check_text(call.action, "adapter call action")
            except JsonSafetyError as exc:
                raise _EpisodeFailure(
                    PHASE_RESPONSE_VALIDATION,
                    KIND_UNREPRESENTABLE_TEXT,
                    f"the adapter's answer on turn {turns_used} carries text this "
                    f"build cannot record: {exc}",
                ) from exc
            action = declared.get(call.action)
            if action is None:
                raise _EpisodeFailure(
                    PHASE_RESPONSE_VALIDATION,
                    KIND_UNKNOWN_ACTION,
                    f"adapter called {call.action!r}, which this case does not "
                    f"offer; it offers {sorted(declared)}",
                    attempted_call=_attempted_call(call),
                )
            if not isinstance(call.arguments, Mapping) or not all(
                isinstance(key, str) for key in call.arguments
            ):
                raise _EpisodeFailure(
                    PHASE_RESPONSE_VALIDATION,
                    KIND_MALFORMED_ARGUMENTS,
                    f"call to {call.action!r} did not carry a string-keyed argument "
                    "mapping",
                    attempted_call=_attempted_call(call),
                )
            problem = _argument_failure(action, call.arguments)
            if problem is not None:
                raise _EpisodeFailure(
                    PHASE_RESPONSE_VALIDATION,
                    KIND_MALFORMED_ARGUMENTS,
                    problem,
                    attempted_call=_attempted_call(call),
                )
            # The arguments conform to the schema and are about to become a
            # recorded step, so they have to survive the row that will hold them.
            # Checked here rather than at append time: an episode that cannot be
            # written is a classified failure, not a crash twelve rows in.
            try:
                ensure_json_safe(call.arguments, "adapter call arguments")
            except JsonSafetyError as exc:
                raise _EpisodeFailure(
                    PHASE_RESPONSE_VALIDATION,
                    KIND_UNREPRESENTABLE_TEXT,
                    f"the adapter's arguments on turn {turns_used} carry text this "
                    f"build cannot record: {exc}",
                ) from exc

            try:
                terminated = _dispatch(environment, call)
            except _EpisodeFailure:
                # Already classified inside the dispatch — a wrong-channel
                # request is refused before the environment is touched.
                raise
            except EnvironmentError as exc:
                # The environment refusing an action is the environment working:
                # the call was schema-valid and the case does not admit it, which
                # is the model's choice and not an infrastructure fault.
                raise _EpisodeFailure(
                    PHASE_DISPATCH,
                    KIND_ACTION_REJECTED,
                    str(exc),
                    exception=exc,
                    attempted_call=_attempted_call(call),
                ) from exc
            except Exception as exc:
                # Anything else out of the environment is the environment's own
                # machinery breaking, which is the one thing an environment
                # failure is reserved for.
                raise _EpisodeFailure(
                    PHASE_DISPATCH,
                    KIND_ENVIRONMENT_EXCEPTION,
                    str(exc),
                    exception=exc,
                    attempted_call=_attempted_call(call),
                ) from exc
            if terminated:
                # The dispatch itself takes time too, so the budget is checked
                # once more before the episode is allowed to call itself done.
                if elapsed() >= limits.episode_timeout_seconds:
                    raise _EpisodeFailure(
                        PHASE_DISPATCH,
                        KIND_EPISODE_TIMEOUT,
                        f"the terminal action on turn {turns_used} completed after "
                        f"the episode's {limits.episode_timeout_seconds} second "
                        "budget had already elapsed",
                    )
                break
    except ScaffoldContractError as exc:
        outcome, error_class, error_detail, failure_event = classify_failure(
            phase=PHASE_SCAFFOLD_CONTRACT,
            kind=KIND_INCOMPLETE_SCAFFOLD,
            detail=str(exc),
            exception=exc,
        )
    except _EpisodeFailure as exc:
        outcome, error_class, error_detail = exc.outcome, exc.error_class, exc.detail
        failure_event = exc.failure_event
    except Exception as exc:
        # The loop's own machinery — the clock, the request projection, the
        # environment's bookkeeping. Nothing here is the model's doing, and
        # letting it escape would turn an infrastructure fault into a traceback
        # instead of a recorded, classified episode.
        outcome, error_class, error_detail, failure_event = classify_failure(
            phase=PHASE_RUNNER,
            kind=KIND_RUNNER_EXCEPTION,
            detail=str(exc),
            exception=exc,
        )

    trajectory = environment.trajectory()
    evaluation: VariantEvaluation | None = None
    if outcome == OUTCOME_SUCCESS:
        # Only a completed episode is graded. A partial trajectory has no verdict
        # to give: its failure is already recorded in the outcome class, and
        # grading it would manufacture a predicate result for an episode that
        # never reached a decision.
        try:
            evaluation = evaluate(variant, trajectory)
        except Exception as exc:
            outcome, error_class, error_detail, failure_event = classify_failure(
                phase=PHASE_EVALUATION,
                kind=KIND_EVALUATOR_EXCEPTION,
                detail=str(exc),
                exception=exc,
            )

    # Clamped at zero rather than recorded negative. A monotonic clock does not
    # go backwards, but ``clock`` is an injected parameter and a negative
    # duration is not a measurement of anything; the floor keeps a broken
    # injection from writing a row the ledger would refuse to read while saying
    # nothing about which episode was mismeasured.
    elapsed_seconds = max(0.0, clock() - started)
    return EpisodeResult(
        outcome=outcome,
        error_class=error_class,
        error_detail=error_detail,
        failure_event=failure_event,
        turns_used=turns_used,
        messages_used=messages_used,
        trajectory=trajectory,
        evaluation=evaluation,
        usage=usage,
        provider_telemetry=_episode_telemetry(telemetry_turns, turns_used),
        started_at_utc=started_at_utc,
        # Derived from the start and the measured duration, never observed. See
        # ``EpisodeResult.completed_at_utc``: the wall clock is read once.
        completed_at_utc=format_utc_timestamp(
            started_moment + timedelta(seconds=elapsed_seconds)
        ),
        elapsed_seconds=elapsed_seconds,
    )


# -- a whole run -------------------------------------------------------------


@dataclass(frozen=True)
class RunReport:
    """What a run invocation did, and what the ledger now holds."""

    manifest: RunManifest
    paths: RunPaths
    records: tuple[EpisodeRecord, ...]
    resumed_count: int
    executed_count: int
    #: Which hard limit stopped this invocation short of its plan, or ``None``
    #: when nothing did. A cost cap stops the run through a recorded row
    #: instead, because a refused request is an event and has to be evidence;
    #: an episode ceiling stops the loop before an episode begins, so there is
    #: nothing to record and the report says so here.
    limits_stop: str | None = None
    #: The invocation limit this invocation was given, or ``None`` for the whole
    #: remaining plan. Reported, never recorded: it belongs to the invocation and
    #: not to the run, which is why it is here and not in the manifest.
    stop_after: int | None = None
    #: Whether that limit is what ended this invocation. ``False`` when the plan
    #: ran out first, because a limit that was never reached withheld nothing.
    staging_stop: bool = False

    @property
    def configuration_id(self) -> str:
        return self.manifest.configuration_id

    @property
    def execution_id(self) -> str:
        return self.manifest.execution_id

    @property
    def run_id(self) -> str:
        """Compatibility alias for :attr:`configuration_id`. See ``RunManifest``."""
        return self.manifest.run_id

    @property
    def run_terminal_failure(self) -> EpisodeRecord | None:
        """The recorded failure that condemned the run, if there was one.

        Derived from the ledger rather than remembered from this invocation: a
        run that stopped is a fact about the evidence on disk, and a resume that
        never executed anything has to reach the same answer as the invocation
        that produced it.
        """
        for record in self.records:
            if record.outcome in RUN_TERMINAL_OUTCOMES:
                return record
        return None

    @property
    def sample_counts(self) -> dict[str, int]:
        """How many rows are each kind of sample. Every status, always present.

        Zero is a real answer here and is stated rather than omitted: "no
        completed samples" is the sentence a reader most needs to see, and a key
        that vanishes when its count is zero is a key nobody reads.
        """
        counts = dict.fromkeys(SAMPLE_STATUSES, 0)
        for record in self.records:
            counts[record.sample_status] += 1
        return counts

    @property
    def completed_sample_count(self) -> int:
        """The only count that could ever be a denominator."""
        return self.sample_counts[SAMPLE_COMPLETED]

    @property
    def provider_attempt_totals(self) -> dict[str, int | None]:
        """Provider requests and provider turns across the run, or ``None``.

        ``None`` when no episode reported any provider telemetry — a run of
        fakes contacts nothing, and reporting ``0 attempts`` would claim it
        tried and got nowhere. A provider run whose every turn was cut off before
        a request genuinely made zero attempts, and reports ``0``.
        """
        measured = [
            record.provider_telemetry
            for record in self.records
            if record.provider_telemetry is not None
        ]
        if not measured:
            return {"attempts": None, "turns": None, "episodes": 0}
        return {
            "attempts": sum(int(entry["attempt_count"]) for entry in measured),
            "turns": sum(len(entry["turns"]) for entry in measured),
            "episodes": len(measured),
        }

    @property
    def provider_fault_counts(self) -> dict[str, int]:
        """Every provider fault the run met, including the ones it recovered from.

        A turn that was rate limited twice and then answered contributes two
        rate limits here and no failed episode, which is the point: the run's
        outcome counts cannot show a fault the run survived.
        """
        counts: dict[str, int] = {}
        for record in self.records:
            telemetry = record.provider_telemetry
            if telemetry is None:
                continue
            for fault, number in telemetry["fault_counts"].items():
                counts[fault] = counts.get(fault, 0) + int(number)
        return dict(sorted(counts.items()))

    @property
    def measurement_coverage(self) -> dict[str, Any]:
        """How much of the run was measured, at three resolutions.

        Episodes, turns and attempts, because a coverage figure at one
        resolution hides the others: twelve episodes each reporting one measured
        turn out of four is "12/12 measured" by episode and a quarter of the run
        by turn. ``full``/``partial``/``unmeasured`` is a closed three-valued
        answer for the same reason a boolean was not enough.
        """
        episodes = dict.fromkeys(MEASUREMENT_STATUSES, 0)
        turns_total = turns_measured = 0
        attempts_total: int | None = None
        attempts_measured: int | None = None
        for record in self.records:
            measurement = record.measurement
            episodes[str(measurement["status"])] += 1
            turns_total += int(measurement["turns_total"])
            turns_measured += int(measurement["turns_measured"])
            if measurement["attempts_total"] is not None:
                attempts_total = (attempts_total or 0) + int(
                    measurement["attempts_total"]
                )
                attempts_measured = (attempts_measured or 0) + int(
                    measurement["attempts_measured"]
                )
        return {
            "episodes": episodes,
            "turns": {"total": turns_total, "measured": turns_measured},
            "attempts": {"total": attempts_total, "measured": attempts_measured},
        }

    @property
    def planned_count(self) -> int:
        return len(self.manifest.episode_plan)

    @property
    def completed_count(self) -> int:
        return len(self.records)

    @property
    def succeeded_count(self) -> int:
        return sum(1 for record in self.records if record.succeeded)

    @property
    def failed_count(self) -> int:
        return self.completed_count - self.succeeded_count

    @property
    def verdict_passed_count(self) -> int:
        """Episodes the evaluator judged correct. *Not* the same as succeeded."""
        return sum(
            1
            for record in self.records
            if record.evaluation is not None and record.evaluation.get("passed") is True
        )

    @property
    def verdict_failed_count(self) -> int:
        return sum(
            1
            for record in self.records
            if record.evaluation is not None and record.evaluation.get("passed") is False
        )

    @property
    def verdict_absent_count(self) -> int:
        """Episodes with no verdict at all: they never reached a decision."""
        return sum(1 for record in self.records if record.evaluation is None)

    @property
    def outcome_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.outcome] = counts.get(record.outcome, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def error_class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            if record.error_class is not None:
                counts[record.error_class] = counts.get(record.error_class, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def usage_totals(self) -> dict[str, int | float | None]:
        """What the run cost, summed across episodes — or ``None`` if unmeasured.

        Null and zero are different facts and the difference survives the sum.
        A fake reports no tokens, no cost and no latency, so a run of fakes
        totals ``None`` on every field: reporting ``0`` would be a fabricated
        measurement of something nothing measured. A field becomes a number as
        soon as *some* episode measures it, and then sums only the episodes that
        did — which is why the measured and unmeasured episode counts are
        reported beside it. A partial total is a real total over a stated
        subset; it is not an estimate of the rest.
        """
        totals: dict[str, int | float | None] = {}
        for name in USAGE_FIELDS:
            running: int | float | None = None
            for record in self.records:
                measured = record.usage.get(name)
                if measured is None:
                    continue
                running = measured if running is None else running + measured
            totals[name] = running
        return totals

    @property
    def usage_measured_episodes(self) -> dict[str, int]:
        """How many episodes reported each measurement at all."""
        return {
            name: sum(1 for record in self.records if record.usage.get(name) is not None)
            for name in USAGE_FIELDS
        }

    @property
    def usage_unmeasured_episodes(self) -> dict[str, int]:
        measured = self.usage_measured_episodes
        return {name: self.completed_count - measured[name] for name in USAGE_FIELDS}

    @property
    def cost_accounting(self) -> dict[str, Any] | None:
        """What this run has spent, from the ledger — or ``None`` if it has no cap.

        Derived from the rows rather than from the guard, so an invocation that
        executed nothing reports the same totals as the one that produced them,
        and every amount is an exact decimal string rather than a JSON float.
        ``None`` means this run pinned no price and did no cost accounting at
        all, which is a different claim from "it spent zero".
        """
        controls = self.manifest.cost_controls
        if not controls.enforces_cost or controls.max_cost_usd is None:
            return None
        measured, exposure = recovered_spend(self.records)
        authorised = measured + exposure
        remaining = controls.max_cost_usd - authorised
        return {
            "max_cost_usd": usd_text(controls.max_cost_usd),
            "measured_usd": usd_text(measured),
            "exposure_usd": usd_text(exposure),
            "authorised_exposure_usd": usd_text(authorised),
            "remaining_usd": usd_text(remaining if remaining > 0 else Decimal(0)),
            "pricing": controls.as_dict()["pricing"],
            "note": COST_ACCOUNTING_NOTE,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            # Two ids, never one. The configuration is what a resume checks and
            # what makes two runs comparable; the execution is which of those
            # runs this was, and when it began.
            "configuration_id": self.configuration_id,
            "execution_id": self.execution_id,
            "created_at_utc": self.manifest.created_at_utc,
            # From the manifest, not from a module constant. A report that
            # stated this build's *only* status would say the same thing about a
            # provider run as about a fake one, which is the defect that made
            # the manifest carry a track in the first place.
            "track": self.manifest.run_track,
            "status": self.manifest.run_status,
            "scope": self.manifest.run_scope,
            "run_directory": str(self.paths.root),
            "manifest_path": str(self.paths.manifest_path),
            "ledger_path": str(self.paths.ledger_path),
            "suite_id": self.manifest.suite_id,
            "suite_content_digest": self.manifest.suite_content_digest,
            "scaffold_id": self.manifest.scaffold_id,
            "scaffold_content_digest": self.manifest.scaffold_content_digest,
            "adapter": self.manifest.adapter.as_dict(),
            "trials": self.manifest.trials,
            "limits": self.manifest.limits.as_dict(),
            "cost_controls": self.manifest.cost_controls.as_dict(),
            "cost_accounting": self.cost_accounting,
            "limits_stop": (
                None
                if self.limits_stop is None
                else {"reason": self.limits_stop, "note": LIMITS_STOP_NOTE}
            ),
            # Always present, and deliberately outside "limits" and
            # "cost_controls": those two blocks are the run's durable settings,
            # and this is a property of one invocation of it.
            "invocation_staging": {
                "stop_after": self.stop_after,
                "stopped_early": self.staging_stop,
                "note": INVOCATION_STAGING_NOTE,
            },
            "retry_policy": dict(RETRY_POLICY),
            "counts": {
                "planned": self.planned_count,
                "completed": self.completed_count,
                "executed": self.executed_count,
                "resumed": self.resumed_count,
                "succeeded": self.succeeded_count,
                "failed": self.failed_count,
                "verdict_passed": self.verdict_passed_count,
                "verdict_failed": self.verdict_failed_count,
                "verdict_absent": self.verdict_absent_count,
            },
            "counts_note": COUNTS_NOTE,
            "plan_note": RUN_PLAN_NOTE,
            "outcome_counts": self.outcome_counts,
            "error_class_counts": self.error_class_counts,
            "usage_totals": self.usage_totals,
            "usage_measured_episodes": self.usage_measured_episodes,
            "usage_unmeasured_episodes": self.usage_unmeasured_episodes,
            "usage_note": USAGE_NOTE,
            "sample_counts": self.sample_counts,
            "sample_eligibility_note": SAMPLE_ELIGIBILITY_NOTE,
            "provider_attempts": self.provider_attempt_totals,
            "provider_fault_counts": self.provider_fault_counts,
            "measurement_coverage": self.measurement_coverage,
            "measurement_note": MEASUREMENT_NOTE,
            "run_terminated": (
                None
                if self.run_terminal_failure is None
                else {
                    "episode_id": self.run_terminal_failure.episode_id,
                    "outcome": self.run_terminal_failure.outcome,
                    "note": RUN_TERMINATED_NOTE,
                }
            ),
            "ledger_integrity_note": LEDGER_INTEGRITY_NOTE,
        }


def recovered_spend(
    records: Sequence[EpisodeRecord],
) -> tuple[Decimal, Decimal]:
    """What a run's ledger says it has already spent: measured, then exposure.

    Read back through ``Decimal(str(value))`` rather than by summing the JSON
    numbers as floats. Every amount this build writes is the shortest decimal
    string that round-trips its float, so the string is the amount that was
    intended and summing the strings is exact; summing the floats would drift,
    and a cost cap enforced against a drifting total is enforced against nothing.

    Null is skipped rather than read as zero, on both quantities and for the
    same reason it is preserved everywhere else: a row that measured nothing did
    not measure zero.
    """
    measured = Decimal(0)
    exposure = Decimal(0)
    for record in records:
        cost = record.usage.get("cost_usd")
        if cost is not None:
            measured += Decimal(str(cost))
        if record.cost_exposure_usd is not None:
            exposure += Decimal(str(record.cost_exposure_usd))
    return measured, exposure


def build_episode_record(
    manifest: RunManifest,
    entry: EpisodePlanEntry,
    variant: Variant,
    result: EpisodeResult,
    cost_exposure_usd: float | None = None,
) -> EpisodeRecord:
    """Project one finished episode into the row that records it.

    Public because a test has to be able to build a *genuine* row — one whose
    trajectory really reconstructs and whose verdict really regrades — before it
    can prove that a tampered one is refused.
    """
    return EpisodeRecord(
        configuration_id=manifest.configuration_id,
        execution_id=manifest.execution_id,
        episode_id=entry.episode_id,
        variant_id=entry.variant_id,
        trial_index=entry.trial_index,
        outcome=result.outcome,
        # Derived here, from the outcome, and re-derived when the row is read.
        # Never a parameter: a caller who could choose what a row is a sample of
        # could inflate an n by choosing well.
        sample_status=sample_status_for(result.outcome),
        error_class=result.error_class,
        error_detail=result.error_detail,
        failure_event=result.failure_event,
        turns_used=result.turns_used,
        messages_used=result.messages_used,
        started_at_utc=result.started_at_utc,
        completed_at_utc=result.completed_at_utc,
        elapsed_seconds=result.elapsed_seconds,
        trajectory=result.trajectory.as_dict(),
        evaluation=None if result.evaluation is None else result.evaluation.as_dict(),
        usage=result.usage.as_dict(),
        provider_telemetry=result.provider_telemetry,
        measurement=derive_measurement(result.provider_telemetry, result.turns_used),
        cost_exposure_usd=cost_exposure_usd,
        suite_content_digest=manifest.suite_content_digest,
        scaffold_content_digest=manifest.scaffold_content_digest,
        variant_content_digest=variant.content_digest,
        adapter=manifest.adapter.as_dict(),
        contract_versions=dict(manifest.contract_versions),
    )


def _disagreement(name: str, expected: Any, actual: Any) -> str:
    return (
        f"{name} does not match the run manifest: the manifest records "
        f"{expected!r}, but this invocation supplied {actual!r}. A run may only "
        "execute the artefacts its manifest pins, because every ledger row "
        "claims that provenance."
    )


def compiled_variants(suite: SuiteReport) -> dict[str, Variant]:
    """Every compiled variant in the suite, by id."""
    return {
        variant.variant_id: variant
        for cube in suite.cubes
        for variant in cube.cube.variants
    }


def _self_assertion(name: str, declared: str, recomputed: str) -> str:
    return (
        f"{name} carries content_digest {declared!r}, but the object in hand "
        f"hashes to {recomputed!r}. A digest that travels inside the artefact it "
        "describes is a claim about it, not a check on it; this run may only "
        "execute artefacts that still hash to the identity they state."
    )


def _verify_recomputed_identity(*, suite: SuiteReport, scaffold: Scaffold) -> None:
    """Re-derive every pinned identity from the objects actually being executed.

    Order matters: each variant is checked before the suite digest that quotes
    it, because the suite payload carries the variants' *stored* digests. A
    variant rebuilt with a new grading contract and its old pin leaves the suite
    digest untouched, so the collection-level check alone would pass it.
    """
    for cube in suite.cubes:
        for variant in cube.cube.variants:
            recomputed = variant_content_digest(variant)
            if recomputed != variant.content_digest:
                raise RunProvenanceError(
                    _self_assertion(
                        f"variant {variant.variant_id}",
                        variant.content_digest,
                        recomputed,
                    )
                )

    recomputed_suite = suite_content_digest(suite.manifest, suite.cubes)
    if recomputed_suite != suite.content_digest:
        raise RunProvenanceError(
            _self_assertion(
                f"suite {suite.manifest.suite_id}",
                suite.content_digest,
                recomputed_suite,
            )
        )
    try:
        enforce_suite_constraints(suite.manifest, suite)
    except SuiteError as exc:
        raise RunProvenanceError(
            f"the suite in hand no longer satisfies the constraints it declares: {exc}"
        ) from exc

    recomputed_scaffold = scaffold_content_digest(scaffold)
    if recomputed_scaffold != scaffold.content_digest:
        raise RunProvenanceError(
            _self_assertion(
                f"scaffold {scaffold.scaffold_id}",
                scaffold.content_digest,
                recomputed_scaffold,
            )
        )


def verify_execution_provenance(
    *,
    suite: SuiteReport,
    scaffold: Scaffold,
    adapter: ModelAdapter,
    manifest: RunManifest,
) -> dict[str, Variant]:
    """Prove the objects in hand *are* the run the manifest describes.

    Runs before the ledger is read, let alone appended, so a mismatched adapter,
    configuration, suite or scaffold fails without producing a single row. Every
    field compared here is one a ledger row goes on to assert as fact.

    Comparing stored digests is not enough on its own. A suite, a cube, a variant
    and a scaffold are all frozen dataclasses that carry their own
    ``content_digest``, so ``dataclasses.replace`` produces an object with new
    content and the *old* pin — and every comparison against the manifest then
    agrees. So each digest is recomputed from the object actually in hand before
    it is compared to anything, and the suite's declared constraints are
    re-derived from its cubes rather than read back.

    The manifest is the same kind of object and gets the same treatment: it is
    rehashed here as well as when the session was opened, because every check
    below compares the artefacts in hand against *it*, and a manifest that does
    not describe its own identity makes all of them agree about the wrong run.
    """
    verify_manifest_identity(manifest)
    _verify_recomputed_identity(suite=suite, scaffold=scaffold)

    if suite.manifest.suite_id != manifest.suite_id:
        raise RunProvenanceError(
            _disagreement("suite_id", manifest.suite_id, suite.manifest.suite_id)
        )
    if suite.manifest.benchmark_version != manifest.suite_benchmark_version:
        raise RunProvenanceError(
            _disagreement(
                "suite benchmark_version",
                manifest.suite_benchmark_version,
                suite.manifest.benchmark_version,
            )
        )
    if suite.content_digest != manifest.suite_content_digest:
        raise RunProvenanceError(
            _disagreement(
                "suite_content_digest",
                manifest.suite_content_digest,
                suite.content_digest,
            )
        )

    if scaffold.scaffold_id != manifest.scaffold_id:
        raise RunProvenanceError(
            _disagreement("scaffold_id", manifest.scaffold_id, scaffold.scaffold_id)
        )
    if scaffold.scaffold_version != manifest.scaffold_version:
        raise RunProvenanceError(
            _disagreement(
                "scaffold_version", manifest.scaffold_version, scaffold.scaffold_version
            )
        )
    if scaffold.content_digest != manifest.scaffold_content_digest:
        raise RunProvenanceError(
            _disagreement(
                "scaffold_content_digest",
                manifest.scaffold_content_digest,
                scaffold.content_digest,
            )
        )

    try:
        identity = adapter.identity.as_dict()
        settings = normalized_adapter_settings(adapter.settings)
    except RunManifestError:
        raise
    except Exception as exc:
        raise RunProvenanceError(
            f"the adapter could not state its own identity or settings: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    # The track, before the identity it is derived from. Both would fire on a
    # swapped adapter, but they are not the same finding and the more specific
    # one is worth stating: an adapter from the other track means every row
    # about to be appended would carry a scope describing a different kind of
    # run — a fake claiming provider execution, or a provider run claiming that
    # no model was executed. Derived from the *live* adapter's provider, because
    # the manifest is the artefact under suspicion.
    live_track = run_track_for_provider(str(identity["provider"]))
    if live_track.track != manifest.run_track:
        raise RunProvenanceError(
            f"the adapter in hand runs on the {live_track.track!r} track, but this "
            f"run's manifest is on the {manifest.run_track!r} track. Every ledger "
            "row asserts the manifest's track as its provenance, so executing this "
            "adapter under this manifest would record a scope describing a "
            "different kind of run from the one that happened."
        )
    if identity != manifest.adapter.as_dict():
        raise RunProvenanceError(
            _disagreement("adapter", manifest.adapter.as_dict(), identity)
        )
    # ``to_json``, not ``dict``: the manifest's settings are deep-frozen, so a
    # shallow copy would compare read-only proxies and tuples against the plain
    # dicts and lists a live adapter reports, and disagree on shape alone.
    pinned = to_json(manifest.adapter_settings)
    if settings != pinned:
        raise RunProvenanceError(_disagreement("adapter_settings", pinned, settings))

    if dict(manifest.contract_versions) != dict(BUILD_CONTRACT_VERSIONS):
        raise RunProvenanceError(
            _disagreement(
                "contract_versions",
                dict(manifest.contract_versions),
                dict(BUILD_CONTRACT_VERSIONS),
            )
        )
    # No retry-policy comparison here: ``RunManifest.retry_policy`` *is* this
    # build's constant rather than a stored field, so comparing it to that
    # constant could only ever agree. A stored policy that names something else
    # is refused where it is actually read — when the manifest is loaded — and a
    # check that cannot fail is worse than no check, because it reads like one.

    variants = compiled_variants(suite)
    expected_plan = tuple(
        EpisodePlanEntry(variant_id=variant.variant_id, trial_index=trial)
        for cube in suite.cubes
        for variant in cube.cube.variants
        for trial in range(1, manifest.trials + 1)
    )
    if manifest.episode_plan != expected_plan:
        raise RunProvenanceError(
            "the manifest's episode plan is not the plan this suite and trial "
            "count produce; the manifest and the suite on disk describe different "
            "runs"
        )
    return variants


def bind_cost_guard(
    *,
    manifest: RunManifest,
    adapter: ModelAdapter,
    cost_guard: RunCostGuard | None,
) -> None:
    """Prove the run's spending is enforced by the limits its manifest records.

    Three things have to agree before a request can be authorised:

    * whether there is a guard at all must match whether the manifest declares a
      cap. A capped manifest run without a guard has a limit nothing enforces,
      and an uncapped manifest run with one is bounded by an amount its own
      durable record does not state;
    * the guard's controls must be exactly the manifest's, so the enforced limit
      is the limit in durable run identity;
    * the guard the runner holds must be the *same object* the adapter asks. Two
      guards with identical controls are not one guard: the run's totals, its
      per-episode exposure and every row derived from them come from the runner's
      one, and the requests are authorised by the adapter's.

    This runs before the ledger is read and before any episode begins, so a
    mismatch costs an error and no provider request.
    """
    declared = manifest.cost_controls
    held = getattr(adapter, "cost_guard", None)
    if not declared.enforces_cost:
        if cost_guard is not None or held is not None:
            raise RunProvenanceError(
                "this run's manifest declares no cost cap, and a cost guard was "
                "supplied for it. A guard bounds spending by an amount the run's "
                "own durable record does not state, so the run would be limited "
                "by a number no reader of this directory could ever see."
            )
        return
    if cost_guard is None:
        raise RunProvenanceError(
            f"this run's manifest declares a cost cap of USD "
            f"{declared.max_cost_usd}, and no cost guard was supplied to enforce "
            "it. A cap nothing enforces is a claim, not a control, so the run "
            "does not start."
        )
    try:
        cost_guard.bind_to(declared)
    except BudgetError as exc:
        raise RunProvenanceError(str(exc)) from exc
    if held is not cost_guard:
        raise RunProvenanceError(
            "the adapter this run would execute does not ask the cost guard this "
            "run accounts with. Every request would be authorised against one "
            "budget and every row would record another, so neither number would "
            "describe the run."
        )


#: The only text an invocation limit may be written as. Deliberately narrower
#: than :func:`int`, which reads ``" 1 "``, ``"+1"``, ``"1_0"`` and non-ASCII
#: digits: an operator staging a run that costs money states a plain count, and
#: a build that quietly reinterpreted what they typed would execute a number of
#: paid episodes nobody wrote down.
_STOP_AFTER_TEXT = re.compile(r"[0-9]+")


def check_stop_after(value: Any) -> int | None:
    """Return a valid invocation limit, or refuse before anything is executed.

    ``None`` means unlimited — execute the whole remaining plan — and is the
    only non-integer this accepts.

    ``type(value) is not int`` rather than ``isinstance``, for the same reason
    :class:`~boundarybench.budget.CostControls` spells it that way: ``bool`` is a
    subclass of ``int``, so a limit wired up from a flag would arrive as ``True``,
    mean "execute exactly one episode" and read in every report as "no limit".
    A float is refused rather than truncated, because ``1.9`` names no number of
    episodes anything could execute.
    """
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise RunnerError(
            f"the invocation limit (--stop-after) must be a positive whole number "
            f"of episodes, and this is {value!r}. It bounds how many new episodes "
            "one invocation executes, so zero, a negative, a fraction or a "
            "non-number is not a smaller stage — it is an instruction with no "
            "meaning, and this build refuses it rather than guessing at one"
        )
    return value


def parse_stop_after(text: str | None) -> int | None:
    """Read an invocation limit as an operator typed it. Strictly."""
    if text is None:
        return None
    if not isinstance(text, str) or not _STOP_AFTER_TEXT.fullmatch(text):
        raise RunnerError(
            f"the invocation limit (--stop-after) must be written as plain digits, "
            f"and this is {text!r}. Surrounding space, a sign, a decimal point, a "
            "separator or an exponent are each refused rather than reinterpreted: "
            "this number decides how many paid episodes an invocation executes"
        )
    return check_stop_after(int(text))


def execute_run(
    *,
    suite: SuiteReport,
    scaffold: Scaffold,
    adapter: ModelAdapter,
    session: RunSession,
    cost_guard: RunCostGuard | None = None,
    stop_after: int | None = None,
    clock: Clock = time.monotonic,
    wall_clock: WallClock = utc_now,
) -> RunReport:
    """Execute every planned episode that is not already recorded.

    Resume is decided by the ledger, not by a status field: an episode with a
    row is done — successfully or terminally — and is never run again. That is
    what makes a completed run a byte-for-byte no-op to repeat.

    Everything happens under the session's exclusive lock, and nothing is read
    or written until the supplied suite, scaffold and adapter have been proven
    to be the ones the manifest pins.

    Both of an episode's clocks are threaded through rather than defaulted
    inside the loop, so a test can state what a whole run's rows say about when
    it ran and how long it took — including what a wall clock stepped backwards
    between two episodes does to the ledger.

    ``stop_after`` bounds how many *new* episodes this invocation executes, and
    is the whole of staged execution: a plan that has to be audited after its
    first episode and again after its first trial block is executed by calling
    this three times with the same manifest, not by killing a process mid-flight.
    It is not a setting of the run — see :data:`INVOCATION_STAGING_NOTE` — and it
    is checked before anything is read, locked or dispatched.
    """
    stop_after = check_stop_after(stop_after)
    session.assert_held()
    manifest = session.manifest
    paths = session.paths
    # Before the provenance checks and so before anything else: those prove the
    # artefacts are the run's, and this proves the money is.
    bind_cost_guard(manifest=manifest, adapter=adapter, cost_guard=cost_guard)
    variants = verify_execution_provenance(
        suite=suite, scaffold=scaffold, adapter=adapter, manifest=manifest
    )

    existing = read_ledger(paths.ledger_path, manifest, variants, scaffold=scaffold)
    resumed = len(existing)
    # Before a single request can be authorised: the guard adopts what this
    # run's own ledger says it has already spent. A resume that started from
    # zero would be authorised to spend the whole cap again, which is the exact
    # failure a durable cap exists to prevent.
    if cost_guard is not None:
        measured, exposure = recovered_spend(existing)
        cost_guard.seed(measured_usd=measured, exposure_usd=exposure)
    # Before a single episode is considered: a run that already stopped for a
    # run-wide configuration failure may not be continued here, whatever the
    # ledger's shape. See ``RunTerminatedError`` — the resume check cannot catch
    # this, because the thing an operator changes to fix it is by design not part
    # of configuration identity.
    for record in existing:
        if record.outcome in RUN_TERMINAL_OUTCOMES:
            raise RunTerminatedError(
                f"{paths.root} holds execution {manifest.execution_id}, which stopped "
                f"at episode {record.episode_id} with outcome {record.outcome!r}. "
                f"{RUN_TERMINATED_NOTE}"
            )
    # The ledger is an exact prefix of the plan, so the work left is the suffix.
    remaining = manifest.episode_plan[resumed:]

    # The hard ceiling on how many episodes this run may execute. The plan was
    # already refused if it exceeded this (see
    # :func:`~boundarybench.runmanifest.check_episode_limit`), so this is the
    # backstop that answers for a manifest assembled some other way: two checks,
    # because "how many were authorised" and "how many actually ran" are
    # different questions and only the second one is a fact about execution.
    episode_limit = manifest.cost_controls.max_episodes
    limits_stop: str | None = None
    staging_stop = False

    executed = 0
    for entry in remaining:
        if episode_limit is not None and resumed + executed >= episode_limit:
            limits_stop = "episode_limit"
            break
        # After the authorised ceiling, never before it. Both stop this
        # invocation at the same episode, but only one of them is a limit the
        # run was authorised under, and a report that named the staging limit
        # would say the operator chose to stop where in fact the run had run
        # out of the episodes it was allowed.
        if stop_after is not None and executed >= stop_after:
            staging_stop = True
            break
        session.assert_held()
        # Re-derived per episode, not once per run. Every row about to be
        # appended asserts this run_id as its provenance, so the identity has to
        # still describe the manifest at the moment the row is written rather
        # than at the moment the run started.
        verify_manifest_identity(manifest)
        variant = variants[entry.variant_id]
        if cost_guard is not None:
            cost_guard.begin_episode()
        result = run_episode(
            variant=variant,
            scaffold=scaffold,
            adapter=adapter,
            limits=manifest.limits,
            clock=clock,
            wall_clock=wall_clock,
        )
        append_episode(
            paths.ledger_path,
            build_episode_record(
                manifest,
                entry,
                variant,
                result,
                cost_exposure_usd=(
                    None if cost_guard is None else float(cost_guard.episode_exposure_usd)
                ),
            ),
        )
        executed += 1
        # After the append, never before it. The evidence that the run met this
        # failure is the whole value of having attempted it, and it has to
        # survive a crash between the decision to stop and the stopping.
        if result.outcome in RUN_TERMINAL_OUTCOMES:
            break

    session.assert_held()
    return RunReport(
        manifest=manifest,
        paths=paths,
        records=read_ledger(paths.ledger_path, manifest, variants, scaffold=scaffold),
        resumed_count=resumed,
        executed_count=executed,
        limits_stop=limits_stop,
        stop_after=stop_after,
        staging_stop=staging_stop,
    )


__all__ = [
    "COUNTS_NOTE",
    "ELICITATION_ACTIONS",
    "INVOCATION_STAGING_NOTE",
    "LIMITS_STOP_NOTE",
    "MEASUREMENT_NOTE",
    "MESSAGES_PER_TURN",
    "READ_ACTION",
    "RUN_TERMINATED_NOTE",
    "TERMINAL_ACTION",
    "WORKFLOW_ACTION",
    "Clock",
    "EpisodeResult",
    "RunProvenanceError",
    "RunReport",
    "RunTerminatedError",
    "RunnerError",
    "ScaffoldContractError",
    "bind_cost_guard",
    "build_episode_record",
    "check_scaffold_contract",
    "check_stop_after",
    "compiled_variants",
    "execute_run",
    "parse_stop_after",
    "recovered_spend",
    "run_episode",
    "verify_execution_provenance",
]
