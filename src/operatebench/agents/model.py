"""A provider-neutral agent over the five Core outcomes.

:class:`ModelAgent` is an :class:`~operatebench.core.protocol.OperationAgent`
whose decisions come from a transport instead of from code. It names no vendor:
what it holds is a model string, a pinned request identity and one
:class:`~operatebench.agents.transport.ModelTransport`, and every test in this
build injects a fake one.

**The parse is the boundary, and every way it fails has a name.** A response
that carries no structured outcome, more than one, an unknown one, one whose
arguments do not satisfy the outcome contract, or one the provider truncated is
not an error to raise and not a decision to guess at. Each becomes a distinct
:class:`MalformedModelOutcome` subclass, which is *not* an ``AgentOutcome``, so
the engine's public boundary refuses it by name as a non-mutating
``MALFORMED_AGENT_OUTCOME`` and the invocation carries on under the ordinary
turn limit. The class name is the classification, which is what puts a stable,
provider-free label into the trajectory the engine writes.

**No raw provider text reaches durable evidence.** Not the prose beside a tool
call, not an unknown tool's name, not an unexpected argument key. A detail this
module writes is built from *this build's* vocabulary plus counts and lengths.
Quoting the model would put unreviewed output into a canonical artefact and
would make two runs that decided the same thing compare as different runs.

**A provider fault is not a decision.** A transport that raised, a response from
the wrong model, and a budget that stopped the run are execution facts. They
raise :class:`~operatebench.agents.transport.ProviderFailure` carrying the
execution record, so the run is excluded with its attempts intact rather than
completed with a business outcome nobody produced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from operatebench.agents.outcome_contract import (
    AGENT_TOOL_NAMES,
    RETRIEVE_TOOL,
    TOOL_NAMES,
    optional_field_names,
    outcome_fields,
    required_field_names,
)
from operatebench.agents.transport import (
    FAULT_BUDGET,
    FAULT_PROTOCOL,
    FAULT_TRANSPORT,
    MAX_ATTEMPTS,
    RETRY_COUNT,
    SOURCE_MODEL,
    ExecutionRecord,
    ModelRequest,
    ModelResponse,
    ModelTransport,
    ProviderAttempt,
    ProviderCallForbiddenError,
    ProviderFailure,
    ToolCall,
    content_digest,
    observation_digest,
)
from operatebench.core.errors import OperateBenchError
from operatebench.core.outcomes import (
    Act,
    AgentOutcome,
    Ask,
    Complete,
    Escalate,
    Wait,
    outcome_contract_problem,
)
from operatebench.core.protocol import AgentObservation, model_projection
from operatebench.core.retrieval import RetrievalRequest, RetrieveBatch
from operatebench.providers.config import ProviderConfigurationError
from operatebench.providers.cost import CostReservationBreachedError

#: The model protocol this build speaks. Part of request identity: a response
#: parsed under one protocol version is not evidence about another.
MODEL_PROTOCOL_VERSION = "operatebench.model.v4"

#: Historical protocol spoken by engine 0.6.0 and artifact contract 7.
#: This is a separate literal so the old identity cannot follow the live name.
MODEL_PROTOCOL_VERSION_V3 = "operatebench.model.v3"

#: The protocol contract v4 artefacts were written under. Still *read* — a
#: record is what it is — and no longer reproducible: v2 offers a sixth tool and
#: publishes three more observation fields, so every ``request_digest_sha256``
#: taken under v1 is over a request this build does not produce.
MODEL_PROTOCOL_VERSION_V1 = "operatebench.model.v1"

#: The protocol contract v5 artefacts were written under. Also still read, and
#: also no longer reproducible: v3 withdraws two observation fields — the
#: normalized record and the bare trigger envelope — so every request built under
#: v2 carried a projection this build cannot reconstruct without putting the
#: answer set back in front of a provider.
MODEL_PROTOCOL_VERSION_V2 = "operatebench.model.v2"

#: The output ceiling this build pins. A response that overran it is classified,
#: not truncated silently into a decision.
MAX_OUTPUT_TOKENS = 4096

#: The exact arguments each of the five outcomes accepts, required and optional,
#: read off the one field contract in
#: :mod:`operatebench.agents.outcome_contract`. A response naming anything
#: outside this table, or carrying an argument the table does not list, is
#: classified rather than coerced.
#:
#: Derived rather than written out, and that is the point of the contract
#: existing: the summary a model is shown, the check its answer is held to and
#: the JSON Schema a provider is sent are three readings of one statement, and
#: the only way they cannot drift is for there to be nothing to drift *from*.
_REQUIRED: Mapping[str, tuple[str, ...]] = {
    name: required_field_names(name) for name in AGENT_TOOL_NAMES
}

_OPTIONAL: Mapping[str, tuple[str, ...]] = {
    name: optional_field_names(name) for name in AGENT_TOOL_NAMES
}

#: The stop reason a provider uses to say it ran out of room mid-answer.
TRUNCATED_STOP_REASON = "max_tokens"


class ModelBoundaryError(OperateBenchError):
    """The model boundary was configured with something it cannot honour."""


# ------------------------------------------------------- named classifications


class MalformedModelOutcome:
    """A model response that is not a decision, carrying why.

    Deliberately not an ``AgentOutcome`` and deliberately not an exception.
    Returning one hands the engine an object its public boundary refuses by
    name — a recorded, non-mutating ``MALFORMED_AGENT_OUTCOME`` the episode
    survives — and the refusal detail the engine writes is this class's name,
    so the classification survives into the trajectory without this module
    having to reach into the ledger.
    """

    #: The stable machine-readable code. Set by each subclass.
    code = "MODEL_OUTPUT_UNCLASSIFIED"

    def __init__(self, detail: str = "") -> None:
        self.detail = detail

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"{type(self).__name__}({self.detail!r})"

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.detail == getattr(other, "detail", None)

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.detail))


class ModelReturnedNoToolCall(MalformedModelOutcome):
    """The response carried prose and no structured outcome."""

    code = "MODEL_NO_TOOL_CALL"


class ModelReturnedMultipleToolCalls(MalformedModelOutcome):
    """The response carried more than one structured outcome.

    Not "take the first". An invocation is one decision, and choosing among
    several would make the benchmark, not the agent, the thing that decided.
    """

    code = "MODEL_MULTIPLE_TOOL_CALLS"


class ModelOutputTruncated(MalformedModelOutcome):
    """The provider stopped mid-answer, so what arrived is not a whole decision."""

    code = "MODEL_OUTPUT_TRUNCATED"


class ModelExceededOutputLimit(MalformedModelOutcome):
    """The response reported more output than the pinned ceiling allows."""

    code = "MODEL_OUTPUT_LIMIT_EXCEEDED"


class ModelNamedAnUnknownTool(MalformedModelOutcome):
    """The structured outcome is not one of the five this build offers."""

    code = "MODEL_UNKNOWN_TOOL"


class ModelToolArgumentsMalformed(MalformedModelOutcome):
    """The named outcome's arguments do not satisfy the outcome contract."""

    code = "MODEL_MALFORMED_ARGUMENTS"


#: Every classification, by code. Playback rebuilds the exact class from the
#: code alone, which is why a recorded malformed decision reproduces the same
#: engine refusal without the artefact having to carry any provider output.
CLASSIFICATIONS: Mapping[str, type[MalformedModelOutcome]] = {
    cls.code: cls
    for cls in (
        ModelReturnedNoToolCall,
        ModelReturnedMultipleToolCalls,
        ModelOutputTruncated,
        ModelExceededOutputLimit,
        ModelNamedAnUnknownTool,
        ModelToolArgumentsMalformed,
    )
}


#: The code an object that is neither a decision nor one of the named
#: misbehaviours above is recorded as. It has no parsed class of its own — it is
#: :class:`MalformedModelOutcome` itself — which is exactly why it is easy to
#: leave out of a reader's vocabulary and why it is named here rather than
#: written out twice.
UNCLASSIFIED_CODE: str = MalformedModelOutcome.code

#: Every classification code a *reader* may meet in a recorded decision. Derived
#: from the two things that can produce one — the parsed classifications above
#: and the base code :func:`~operatebench.agents.playback.decision_record` falls
#: back to — because a hand-written copy is where the writer and the reader
#: silently diverge, and they did: a tape carrying the base code was written by
#: this build and then refused by it.
#:
#: Closed, and made of this build's own words. A classification carries a code
#: and nothing else, so no provider text reaches durable evidence through it.
READABLE_CLASSIFICATIONS: frozenset[str] = frozenset(
    {*CLASSIFICATIONS, UNCLASSIFIED_CODE}
)


def classification(code: str) -> MalformedModelOutcome:
    """Rebuild a classification from its stable code."""
    try:
        return CLASSIFICATIONS[code]()
    except KeyError as exc:
        raise ModelBoundaryError(
            f"{code!r} is not a model output classification this build writes; "
            f"known codes are {sorted(CLASSIFICATIONS)}"
        ) from exc


# ------------------------------------------------------------------- the agent


class ModelAgent:
    """One transport, five outcomes, and a pinned request identity.

    ``single_use`` is the load-bearing attribute. A model agent cannot be run
    twice to check that it reproduces itself — the second run is a second
    provider run, and a stochastic one need not agree with the first. The runner
    reads this flag and satisfies ``deterministic_replay`` by replaying the
    *recorded outcomes* instead, which is a stronger check and costs nothing.
    """

    #: Executing this agent again is executing the provider again.
    single_use = True

    #: What produced its decisions, for the run's execution record.
    outcome_source = SOURCE_MODEL

    def __init__(
        self,
        transport: ModelTransport,
        *,
        model: str,
        agent_id: str = "model",
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        max_transport_calls: int | None = None,
    ) -> None:
        if not model:
            raise ModelBoundaryError(
                "a model agent must name the model it runs; an unnamed model cannot "
                "be recorded as the identity of anything it produced"
            )
        if max_output_tokens <= 0:
            raise ModelBoundaryError(
                "the output ceiling must be a positive number of tokens, got "
                f"{max_output_tokens}"
            )
        if max_transport_calls is not None and max_transport_calls <= 0:
            raise ModelBoundaryError(
                f"a call budget must allow at least one call, got {max_transport_calls}"
            )
        self.agent_id = agent_id
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.max_transport_calls = max_transport_calls
        self._transport = transport
        self._identity: dict[str, Any] = {}
        self._attempts: list[ProviderAttempt] = []
        self._calls = 0

    # ------------------------------------------------------------ the protocol

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        self._identity = dict(identity)
        self._attempts = []
        self._calls = 0

    def decide(
        self, observation: AgentObservation
    ) -> AgentOutcome | RetrieveBatch | MalformedModelOutcome:
        request = self.build_request(observation)
        response = self._send(request)
        self._check_response_identity(request, response)
        decision = self._parse(response)
        self._attempts.append(
            ProviderAttempt(
                invocation_index=request.invocation_index,
                turn_index=request.turn_index,
                request_digest_sha256=request.request_digest_sha256,
                outcome=(
                    "classified"
                    if isinstance(decision, MalformedModelOutcome)
                    else "decided"
                ),
            )
        )
        return decision

    # ------------------------------------------------------------- the request

    def build_request(self, observation: AgentObservation) -> ModelRequest:
        """The exact request this observation produces. Identity is pinned here.

        The prompt is built from
        :func:`~operatebench.core.protocol.model_projection` — the explicit
        allowlist of model-visible fields — and never from a generic
        serialisation of whatever the observation happens to hold. The two look
        the same today and fail differently tomorrow: a field added to the
        observation for the evaluator's benefit reaches a provider through
        ``as_dict()`` and is refused by the allowlist. Disclosure is a decision
        somebody makes by name, not a default.
        """
        public = model_projection(observation)
        prompt = {
            "protocol_version": MODEL_PROTOCOL_VERSION,
            "tools": tool_schema(),
            "observation": public,
        }
        return ModelRequest(
            model=self.model,
            protocol_version=MODEL_PROTOCOL_VERSION,
            max_output_tokens=self.max_output_tokens,
            tool_names=AGENT_TOOL_NAMES,
            observation_digest_sha256=observation_digest(observation),
            prompt_digest_sha256=content_digest(prompt),
            invocation_index=observation.invocation_index,
            turn_index=observation.turn_index,
            prompt=prompt,
        )

    def _send(self, request: ModelRequest) -> ModelResponse:
        budget = self.max_transport_calls
        if budget is not None and self._calls >= budget:
            # A bounded stop, not a lost run. Every attempt already made stays in
            # the record: "we stopped at the budget after four calls" and "the
            # provider never answered" are different facts about a run.
            raise ProviderFailure(
                self._excluded(FAULT_BUDGET),
                f"the call budget of {budget} provider call(s) is "
                f"spent; the {len(self._attempts)} attempt(s) already made are "
                "retained and the run is excluded rather than completed",
            )
        self._calls += 1
        try:
            return self._transport.send(request)
        except ProviderCallForbiddenError:
            # The proof object for a zero-call path. Swallowing it into an
            # exclusion would turn a violated guarantee into a soft finding.
            raise
        except ProviderFailure:
            raise
        except ProviderConfigurationError:
            # The transport boundary was entered, but its own final admission
            # refused before dispatch. It is neither a provider call nor an
            # attempt and must retain its local configuration taxonomy.
            self._calls -= 1
            raise
        except CostReservationBreachedError as exc:
            # Dispatch answered, but settlement violated our admitted cost bound.
            # The executor already recorded that attempt and its settlement: do
            # not invent a transport fault or a pre-dispatch refusal as witness.
            raise ProviderFailure(
                self._excluded(FAULT_BUDGET),
                "provider usage breached the admitted cost reservation; "
                "this run is excluded, not scored",
            ) from exc
        except Exception as exc:
            self._attempts.append(
                ProviderAttempt(
                    invocation_index=request.invocation_index,
                    turn_index=request.turn_index,
                    request_digest_sha256=request.request_digest_sha256,
                    outcome="failed",
                    fault=FAULT_TRANSPORT,
                )
            )
            raise ProviderFailure(
                self._excluded(FAULT_TRANSPORT),
                f"the provider transport failed on invocation "
                f"{request.invocation_index} turn {request.turn_index} after "
                f"{MAX_ATTEMPTS} attempt(s) ({type(exc).__name__}); this run is "
                "excluded, not scored",
            ) from exc

    def _check_response_identity(
        self, request: ModelRequest, response: ModelResponse
    ) -> None:
        """A response from a model nobody asked is a protocol failure, not a decision.

        Checked before the parse. An answer produced by a different model under
        this run's recorded identity would make the artefact assert provenance
        for a request that was never sent.
        """
        if response.model == request.model:
            return
        self._attempts.append(
            ProviderAttempt(
                invocation_index=request.invocation_index,
                turn_index=request.turn_index,
                request_digest_sha256=request.request_digest_sha256,
                outcome="failed",
                fault=FAULT_PROTOCOL,
            )
        )
        raise ProviderFailure(
            self._excluded(FAULT_PROTOCOL),
            f"the response identifies a model {len(response.model)} character(s) long "
            f"that is not the {request.model!r} this run requested; a run cannot "
            "record one model's identity over another model's answers",
        )

    # --------------------------------------------------------------- the parse

    def _parse(
        self, response: ModelResponse
    ) -> AgentOutcome | RetrieveBatch | MalformedModelOutcome:
        if response.stop_reason == TRUNCATED_STOP_REASON:
            return ModelOutputTruncated(
                "the provider stopped at the output ceiling, so what arrived is a "
                "fragment rather than a decision"
            )
        if response.output_tokens > self.max_output_tokens:
            return ModelExceededOutputLimit(
                f"the response reports {response.output_tokens} output token(s) "
                f"against a pinned ceiling of {self.max_output_tokens}"
            )
        calls = tuple(response.tool_calls)
        if not calls:
            return ModelReturnedNoToolCall(
                "the response carried no structured outcome; prose is not a decision "
                "this benchmark can execute"
            )
        if len(calls) > 1:
            return ModelReturnedMultipleToolCalls(
                f"the response carried {len(calls)} structured outcomes; an "
                "invocation is one decision, and choosing among several would make "
                "the benchmark the thing that decided"
            )
        return parse_tool_call(calls[0])

    # ----------------------------------------------------------- the execution

    def execution_record(self) -> ExecutionRecord:
        """How this agent's decisions were produced, as the artefact records it."""
        return ExecutionRecord(
            outcome_source=SOURCE_MODEL,
            model=self.model,
            protocol_version=MODEL_PROTOCOL_VERSION,
            max_output_tokens=self.max_output_tokens,
            transport_calls=self._calls,
            retry_count=RETRY_COUNT,
            attempts=tuple(self._attempts),
            excluded=False,
            exclusion_code=None,
            exclusion_detail="",
        )

    def _excluded(self, fault: str) -> ExecutionRecord:
        base = self.execution_record()
        return ExecutionRecord(
            outcome_source=base.outcome_source,
            model=base.model,
            protocol_version=base.protocol_version,
            max_output_tokens=base.max_output_tokens,
            transport_calls=base.transport_calls,
            retry_count=base.retry_count,
            attempts=base.attempts,
            excluded=True,
            exclusion_code=fault,
            exclusion_detail=f"execution stopped at the provider boundary ({fault})",
        )


# ------------------------------------------------------------ argument parsing


def tool_schema() -> dict[str, Any]:
    """The output schema offered to the model, as a plain mapping.

    Derived from the same field contract the parser holds to, so the schema a
    model is shown and the schema its answer is checked against cannot drift
    apart. It is a *summary* — names, plus guidance only where the prompt needs
    to disambiguate a field — and that is why it is not what a real provider is
    sent: a JSON Schema needs the types and descriptions too, and
    :func:`~operatebench.agents.lifecycle_contract.outcome_tools` derives those
    from the same contract rather than from a second list.
    """
    summary: dict[str, Any] = {}
    for name in AGENT_TOOL_NAMES:
        tool: dict[str, Any] = {
            "required": list(_REQUIRED[name]),
            "optional": list(_OPTIONAL[name]),
        }
        evidence_guidance = {
            field.name: field.description
            for field in outcome_fields(name)
            if field.name == "evidence_refs"
        }
        if evidence_guidance:
            tool["field_guidance"] = evidence_guidance
        summary[name] = tool
    return summary


def parse_tool_call(
    call: ToolCall,
) -> AgentOutcome | RetrieveBatch | MalformedModelOutcome:
    """One structured call into one Core object, or a named classification.

    Six tools now, not five. The sixth returns a
    :class:`~operatebench.core.retrieval.RetrieveBatch`, which is not an outcome
    and is deliberately not one: the engine branches on the type before the
    outcome contract is consulted, so this widening cannot widen what the
    operation will execute.
    """
    name = call.name
    if not isinstance(name, str) or name not in _REQUIRED:
        # The name itself is provider output and is never echoed. Its length is
        # enough to tell "empty" from "hallucinated" without quoting it.
        length = len(name) if isinstance(name, str) else -1
        return ModelNamedAnUnknownTool(
            f"the response named a structured outcome of {length} character(s) that "
            f"is not one of {list(AGENT_TOOL_NAMES)}"
        )
    arguments = call.arguments
    if not isinstance(arguments, Mapping):
        return ModelToolArgumentsMalformed(
            f"the {name!r} arguments are a {type(arguments).__name__}, not a mapping"
        )
    problem = _argument_names_problem(name, arguments)
    if problem is not None:
        return ModelToolArgumentsMalformed(problem)
    try:
        outcome = _build(name, arguments)
    except OperateBenchError as exc:
        # The outcome dataclasses refuse an empty action type and a wait that
        # can never end. That refusal is a classification here, not a crash.
        return ModelToolArgumentsMalformed(
            f"the {name!r} arguments do not satisfy the outcome contract "
            f"({type(exc).__name__})"
        )
    except (TypeError, ValueError) as exc:
        return ModelToolArgumentsMalformed(
            f"the {name!r} arguments could not be read as an outcome "
            f"({type(exc).__name__})"
        )
    if isinstance(outcome, (MalformedModelOutcome, RetrieveBatch)):
        # A retrieval is not an outcome, so the outcome contract has nothing to
        # say about it. Its own contract — the catalogue, the bounds and the
        # budget — is the environment's to apply, and it applies it against the
        # published affordances this parser cannot see.
        return outcome
    contract = outcome_contract_problem(outcome)
    if contract is not None:
        # Held to the same contract the engine holds it to, one frame earlier,
        # so the classification is the model's rather than a generic refusal.
        return ModelToolArgumentsMalformed(
            f"the {name!r} arguments build an outcome the contract refuses: {contract}"
        )
    return outcome


def _argument_names_problem(name: str, arguments: Mapping[str, Any]) -> str | None:
    """Missing and unknown argument names, without quoting a single one of them."""
    keys = set(arguments)
    missing = sorted(set(_REQUIRED[name]) - keys)
    if missing:
        return f"the {name!r} arguments are missing {missing}"
    known = set(_REQUIRED[name]) | set(_OPTIONAL[name])
    unknown = len(keys - known)
    if unknown:
        return (
            f"the {name!r} arguments carry {unknown} field(s) this build does not "
            f"accept; it accepts {sorted(known)}"
        )
    return None


def _build(
    name: str, arguments: Mapping[str, Any]
) -> AgentOutcome | RetrieveBatch | MalformedModelOutcome:
    if name == RETRIEVE_TOOL:
        return _retrieve(arguments)
    if name == "act":
        return Act(
            action_type=arguments["action_type"],
            payload=arguments.get("payload", {}),
            evidence_refs=_refs(arguments.get("evidence_refs", ())),
            rationale=arguments.get("rationale", ""),
        )
    if name == "wait":
        return _wait(arguments)
    if name == "ask":
        nested = arguments["wait"]
        if not isinstance(nested, Mapping):
            return ModelToolArgumentsMalformed(
                f"the ASK wait is a {type(nested).__name__}, not a mapping"
            )
        problem = _argument_names_problem("wait", nested)
        if problem is not None:
            return ModelToolArgumentsMalformed(problem)
        wait = _wait(nested)
        if isinstance(wait, MalformedModelOutcome):
            return wait
        return Ask(
            recipient_actor_id=arguments["recipient_actor_id"],
            message_fixture_id=arguments["message_fixture_id"],
            wait=wait,
            correlation_id=arguments.get("correlation_id"),
        )
    if name == "escalate":
        return Escalate(
            checkpoint_id=arguments["checkpoint_id"],
            exception_type=arguments["exception_type"],
            evidence_refs=_refs(arguments.get("evidence_refs", ())),
            deadline_after_minutes=arguments.get("deadline_after_minutes", 1440),
            rationale=arguments.get("rationale", ""),
        )
    return Complete(
        reason=arguments.get("reason", ""),
        evidence_refs=_refs(arguments.get("evidence_refs", ())),
    )


def _retrieve(arguments: Mapping[str, Any]) -> RetrieveBatch | MalformedModelOutcome:
    """One structured ``retrieve`` call into one batch, or a classification.

    The shape is checked here and the *contents* are not: whether a named tool
    exists, whether the batch is within its bounds and whether the budget is
    spent are all statements about the published affordances, and the environment
    is what holds those. Answering them here would put a second copy of the
    catalogue on the parser's side of the boundary.
    """
    requests = arguments["requests"]
    if not isinstance(requests, list):
        return ModelToolArgumentsMalformed(
            f"the retrieve requests are a {type(requests).__name__}, not an array"
        )
    built: list[RetrievalRequest] = []
    for position, entry in enumerate(requests):
        if not isinstance(entry, Mapping):
            return ModelToolArgumentsMalformed(
                f"retrieve request {position} is a {type(entry).__name__}, not an object"
            )
        unknown = len(set(entry) - {"tool", "arguments"})
        if unknown or "tool" not in entry:
            return ModelToolArgumentsMalformed(
                f"retrieve request {position} states {len(entry)} field(s); a request "
                "names a tool and, optionally, that tool's arguments"
            )
        tool = entry["tool"]
        if not isinstance(tool, str) or not tool:
            return ModelToolArgumentsMalformed(
                f"retrieve request {position} names a tool that is not a non-empty string"
            )
        given = entry.get("arguments")
        if given is None:
            given = {}
        if not isinstance(given, Mapping):
            return ModelToolArgumentsMalformed(
                f"retrieve request {position} carries arguments that are a "
                f"{type(given).__name__}, not an object"
            )
        built.append(RetrievalRequest(tool=tool, arguments=dict(given)))
    return RetrieveBatch(tuple(built))


def _wait(arguments: Mapping[str, Any]) -> Wait | MalformedModelOutcome:
    return Wait(
        reason=arguments["reason"],
        wake_on=_refs(arguments.get("wake_on", ())),
        fallback_after_minutes=arguments.get("fallback_after_minutes"),
    )


def _refs(value: Any) -> Sequence[str]:
    """A JSON array becomes a tuple; anything else is handed on to be refused.

    Deliberately not a coercion. A bare string satisfies ``Sequence[str]`` and
    iterates into characters, which is exactly the shape a model produces by
    answering ``evidence_refs: "evidence_1"`` — so it is passed through
    unchanged and the outcome contract names it.
    """
    if isinstance(value, list):
        return tuple(value)
    return value  # type: ignore[no-any-return]


__all__ = [
    "AGENT_TOOL_NAMES",
    "CLASSIFICATIONS",
    "MAX_OUTPUT_TOKENS",
    "MODEL_PROTOCOL_VERSION",
    "MODEL_PROTOCOL_VERSION_V1",
    "MODEL_PROTOCOL_VERSION_V2",
    "MODEL_PROTOCOL_VERSION_V3",
    "READABLE_CLASSIFICATIONS",
    "TOOL_NAMES",
    "TRUNCATED_STOP_REASON",
    "UNCLASSIFIED_CODE",
    "MalformedModelOutcome",
    "ModelAgent",
    "ModelBoundaryError",
    "ModelExceededOutputLimit",
    "ModelNamedAnUnknownTool",
    "ModelOutputTruncated",
    "ModelReturnedMultipleToolCalls",
    "ModelReturnedNoToolCall",
    "ModelToolArgumentsMalformed",
    "classification",
    "parse_tool_call",
    "tool_schema",
]
