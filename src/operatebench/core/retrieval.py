"""The sixth object an agent may return, and what the environment does with it.

A ``RETRIEVE`` is not a decision. It changes nothing, commits nothing and is
never graded as a business outcome — it is the agent saying *show me these
records*, and the environment is the only thing that may answer. That separation
is the whole design, and it is why :class:`RetrieveBatch` is deliberately **not**
a member of the :data:`~operatebench.core.outcomes.AgentOutcome` union: the
engine's public boundary keeps refusing anything it does not recognise, and a
retrieval is recognised one branch earlier, by type, rather than by widening the
contract five outcomes are held to.

Three properties earn their place here.

**A batch is finite and its canonical form is unique.** One to eight requests,
drawn from a published catalogue, canonicalised by tool name. Two agents that ask
for the same set in different orders, or that repeat a tool, produce the same
byte-identical tape row — which is what makes "this run read these records" a
comparison rather than a coincidence of typing.

**Every way a batch can be wrong has a name.** Empty, oversize, unknown tool,
malformed request, conflicting duplicate, exhausted budget: six codes, no silent
truncation and no raw exception out of the engine boundary. A retrieval refusal
is non-mutating by construction — nothing has been served when it is decided.

**A duplicate collapses; a conflict is refused.** Asking for ``list_quotes``
twelve times is one read. Asking for it twice with two different argument sets is
two different questions wearing one name, and resolving that quietly would make
the environment choose which one the agent meant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

#: What a ``RetrieveBatch`` calls itself. Outside :data:`OUTCOME_KINDS` on
#: purpose: the five business outcomes are what the operation records, and this
#: is not one of them.
RETRIEVE_KIND = "RETRIEVE"

#: A batch carries at least one request and at most eight distinct ones. The
#: upper bound is the size of the published vocabulary, so "everything at once"
#: is expressible and "more than everything" is not.
MIN_REQUESTS_PER_BATCH = 1
MAX_REQUESTS_PER_BATCH = 8

#: The authority classes a fact can carry, closed. An event says *someone
#: asserted this*; only a result whose authority is
#: :data:`AUTHORITY_AUTHORITATIVE_VERIFICATION` says *the record holds it*.
AUTHORITY_ACTOR_CLAIM = "actor_claim"
AUTHORITY_AUTHORITATIVE_VERIFICATION = "authoritative_verification"
AUTHORITY_COMMUNICATION = "communication"
AUTHORITY_SYSTEM_EVENT = "system_event"
AUTHORITY_SYSTEM_OF_RECORD = "system_of_record"

RETRIEVAL_AUTHORITIES: tuple[str, ...] = (
    AUTHORITY_ACTOR_CLAIM,
    AUTHORITY_AUTHORITATIVE_VERIFICATION,
    AUTHORITY_COMMUNICATION,
    AUTHORITY_SYSTEM_EVENT,
    AUTHORITY_SYSTEM_OF_RECORD,
)

#: Why a batch was refused. Every one of these is a statement about the *public*
#: contract — the catalogue, the bounds and the budget the observation carries —
#: so an agent could have avoided any of them from what it was told.
RETRIEVAL_BATCH_TOO_LARGE = "RETRIEVAL_BATCH_TOO_LARGE"
RETRIEVAL_BUDGET_EXHAUSTED = "RETRIEVAL_BUDGET_EXHAUSTED"
RETRIEVAL_CONFLICTING_ARGUMENTS = "CONFLICTING_RETRIEVAL_ARGUMENTS"
RETRIEVAL_EMPTY_BATCH = "EMPTY_RETRIEVAL_BATCH"
RETRIEVAL_MALFORMED_REQUEST = "MALFORMED_RETRIEVAL_REQUEST"
RETRIEVAL_UNKNOWN_TOOL = "UNKNOWN_RETRIEVAL_TOOL"

RETRIEVAL_REFUSAL_CODES: tuple[str, ...] = tuple(
    sorted(
        {
            RETRIEVAL_BATCH_TOO_LARGE,
            RETRIEVAL_BUDGET_EXHAUSTED,
            RETRIEVAL_CONFLICTING_ARGUMENTS,
            RETRIEVAL_EMPTY_BATCH,
            RETRIEVAL_MALFORMED_REQUEST,
            RETRIEVAL_UNKNOWN_TOOL,
        }
    )
)

#: Who asked for a read. Closed, and in this phase it holds exactly one member:
#: the environment never refreshes a read on the agent's behalf, so every served
#: result in this build is the answer to a decision the agent made. A later phase
#: that adds an environment refresh has to add a member here first, and grading
#: can then tell the two apart instead of assuming.
INITIATED_BY_AGENT = "agent"
RETRIEVAL_INITIATORS: tuple[str, ...] = (INITIATED_BY_AGENT,)

#: Why derived context stopped being usable. A wake and the agent's own accepted
#: effect are the same fact seen from two sides: the document it was looking at
#: has moved underneath it.
INVALIDATION_EFFECT_ACCEPTED = "effect_accepted"
INVALIDATION_WAKE = "wake"
INVALIDATION_CAUSES: tuple[str, ...] = tuple(
    sorted({INVALIDATION_EFFECT_ACCEPTED, INVALIDATION_WAKE})
)


@dataclass(frozen=True)
class RetrievalRequest:
    """One read: a tool from the published catalogue, and its exact arguments.

    ``arguments`` is held to the catalogue's per-tool schema, not to a generic
    key namespace. A vertical whose reads take no parameter publishes an empty
    schema and an invented key is refused — inventing one is interface guessing,
    which is the thing the published contract exists to make unnecessary.
    """

    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "arguments": dict(self.arguments)}


@dataclass(frozen=True)
class RetrieveBatch:
    """One to eight reads, served atomically at one instant by the environment.

    Not an :data:`~operatebench.core.outcomes.AgentOutcome`. The engine branches
    on the type before the outcome contract is consulted, so widening this never
    widens what an operation will execute.
    """

    requests: Sequence[RetrievalRequest] = ()
    kind: str = field(default=RETRIEVE_KIND, init=False)

    def tools(self) -> tuple[str, ...]:
        return tuple(request.tool for request in self.requests)


@dataclass(frozen=True)
class ToolResult:
    """One retrieved fact set, with the provenance that makes it citable.

    ``record_version`` is what a replay binds to: the same records read twice
    without an intervening change report the same version, and a change moves it.
    ``as_of`` is the instant the batch was served — one instant for the whole
    batch, so results inside it cannot disagree about when they were true.
    """

    tool: str
    source: str
    authority: str
    record_id: str
    record_version: str
    as_of: str
    schema_id: str
    records: Mapping[str, Any] = field(default_factory=dict)
    ok: bool = True
    error_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "source": self.source,
            "authority": self.authority,
            "record_id": self.record_id,
            "record_version": self.record_version,
            "as_of": self.as_of,
            "schema_id": self.schema_id,
            "records": dict(self.records),
            "ok": self.ok,
            "error_code": self.error_code,
        }

    def provenance(self) -> dict[str, Any]:
        """Everything about this result except what it says.

        What a replay compares. The records themselves are compared by the
        trajectory and the final state; what this states is *which record, at
        which version, from which source, under whose authority*.
        """
        body = self.as_dict()
        body.pop("records")
        return body


def tool_result_from_dict(body: Mapping[str, Any]) -> ToolResult:
    """Rebuild one result from the projection an observation carries."""
    return ToolResult(
        tool=str(body["tool"]),
        source=str(body["source"]),
        authority=str(body["authority"]),
        record_id=str(body["record_id"]),
        record_version=str(body["record_version"]),
        as_of=str(body["as_of"]),
        schema_id=str(body["schema_id"]),
        records=dict(body.get("records") or {}),
        ok=bool(body.get("ok", True)),
        error_code=body.get("error_code"),
    )


def _request_problem(request: Any) -> str | None:
    """Why this object cannot be read as a request, or ``None`` if it can."""
    if not isinstance(request, RetrievalRequest):
        return (
            "a retrieval request must be a RetrievalRequest, got "
            f"{type(request).__name__}"
        )
    if not isinstance(request.tool, str) or not request.tool:
        return "a retrieval request must name a non-empty tool"
    if not isinstance(request.arguments, Mapping):
        return (
            "a retrieval request's arguments must be a mapping, got "
            f"{type(request.arguments).__name__}"
        )
    for key in request.arguments:
        if not isinstance(key, str) or not key:
            return "a retrieval argument name must be a non-empty string"
    return None


def _argument_key(request: RetrievalRequest) -> tuple[tuple[str, str], ...]:
    """A hashable, order-independent identity for one request's arguments."""
    return tuple(sorted((name, repr(value)) for name, value in request.arguments.items()))


def canonical_requests(batch: RetrieveBatch) -> tuple[RetrievalRequest, ...]:
    """The batch's unique canonical form: sorted by tool, duplicates collapsed.

    Called after :func:`retrieval_batch_problem` has passed, so every request is
    well formed and no two requests name one tool with different arguments —
    which is what makes sorting by tool name a *total* order rather than one that
    has to break a tie by guessing.
    """
    seen: dict[str, RetrievalRequest] = {}
    for request in batch.requests:
        seen.setdefault(request.tool, request)
    return tuple(seen[tool] for tool in sorted(seen))


def retrieval_batch_problem(
    batch: RetrieveBatch,
    *,
    catalogue: Mapping[str, Mapping[str, Any]],
    batch_budget_remaining: int,
) -> tuple[str, str] | None:
    """Why this batch cannot be served, as ``(code, detail)``, or ``None``.

    The order of the checks is part of the contract. Structure first, because a
    request that is not a request cannot be counted or compared; then the empty
    batch, which is empty however the bounds are set; then size, because nine
    requests are refused for being nine whatever they name — an oversize batch
    cannot be re-labelled as an unknown-tool or conflicting-argument refusal by
    including one; then membership semantics — conflicting duplicates, then
    names outside the catalogue; then the per-tool argument schema; and the
    budget last, because spending it on a batch that was never legal would be a
    charge for nothing.

    No tool name a caller invented is echoed into ``detail``. A refusal reaches
    the observation and, through it, a durable record; provider-controlled text
    does not belong in either.
    """
    requests = batch.requests
    if isinstance(requests, (str, bytes, Mapping)) or not isinstance(requests, Sequence):
        return (
            RETRIEVAL_MALFORMED_REQUEST,
            "a retrieval batch carries a sequence of requests, got "
            f"{type(requests).__name__}",
        )
    for position, request in enumerate(requests):
        problem = _request_problem(request)
        if problem is not None:
            return (RETRIEVAL_MALFORMED_REQUEST, f"request {position}: {problem}")

    if len(requests) < MIN_REQUESTS_PER_BATCH:
        return (
            RETRIEVAL_EMPTY_BATCH,
            "a retrieval batch carries at least one request; an empty one asks "
            "the environment for nothing and would be a turn spent on silence",
        )

    # Size, before anything that reads the catalogue. Nine distinct requests are
    # refused for being nine whatever they name and however they disagree with
    # each other, which is what stops an oversize batch being re-labelled — as
    # an unknown-tool refusal by including one unknown name, or as a conflict by
    # including one conflicting pair. A batch answered for its conflict would
    # send an agent to drop one argument set and retry a batch that is still
    # oversize; a refusal that teaches the wrong repair is worse than a blunt
    # one. ``distinct`` counts *questions*, so a conflicting pair counts twice —
    # they are two questions wearing one name, which is exactly why the pair is
    # refused once the batch is small enough for the question to arise.
    distinct = {(request.tool, _argument_key(request)) for request in requests}
    if len(distinct) > MAX_REQUESTS_PER_BATCH:
        return (
            RETRIEVAL_BATCH_TOO_LARGE,
            f"a retrieval batch carries at most {MAX_REQUESTS_PER_BATCH} distinct "
            f"requests and this one carries {len(distinct)}",
        )

    conflicting = sorted(
        {
            request.tool
            for request in requests
            for other in requests
            if other.tool == request.tool
            and _argument_key(other) != _argument_key(request)
            and request.tool in catalogue
        }
    )
    if conflicting:
        return (
            RETRIEVAL_CONFLICTING_ARGUMENTS,
            f"the batch names {conflicting} more than once with different arguments; "
            "a duplicate collapses, but two argument sets under one tool name are "
            "two questions and the environment does not choose between them",
        )

    unknown = len({request.tool for request in requests} - set(catalogue))
    if unknown:
        return (
            RETRIEVAL_UNKNOWN_TOOL,
            f"the batch names {unknown} tool(s) outside the published catalogue; "
            f"it publishes {sorted(catalogue)}",
        )

    for request in requests:
        declared = catalogue[request.tool].get("arguments") or {}
        invented = sorted(set(request.arguments) - set(declared))
        if invented:
            return (
                RETRIEVAL_MALFORMED_REQUEST,
                f"the {request.tool!r} request carries argument name(s) {invented} "
                f"the catalogue does not publish; it publishes {sorted(declared)}",
            )

    if batch_budget_remaining <= 0:
        return (
            RETRIEVAL_BUDGET_EXHAUSTED,
            "the retrieval batch budget for this invocation is spent; the bound is "
            "published on every observation and this is a refusal rather than a "
            "silently truncated read",
        )
    return None


class RetrievalPlanner(Protocol):
    """What an agent that plans its own reads implements.

    Shared deliberately. The reference, a negative and a model-backed agent all
    reach the environment through the same object, so "the agent decided to read"
    is one mechanism rather than a privileged path per agent kind.
    """

    def plan_retrieval(self, observation: Any) -> RetrieveBatch | None:
        """The batch this observation calls for, or ``None`` to decide instead."""
        ...


__all__ = [
    "AUTHORITY_ACTOR_CLAIM",
    "AUTHORITY_AUTHORITATIVE_VERIFICATION",
    "AUTHORITY_COMMUNICATION",
    "AUTHORITY_SYSTEM_EVENT",
    "AUTHORITY_SYSTEM_OF_RECORD",
    "INITIATED_BY_AGENT",
    "INVALIDATION_CAUSES",
    "INVALIDATION_EFFECT_ACCEPTED",
    "INVALIDATION_WAKE",
    "MAX_REQUESTS_PER_BATCH",
    "MIN_REQUESTS_PER_BATCH",
    "RETRIEVAL_AUTHORITIES",
    "RETRIEVAL_BATCH_TOO_LARGE",
    "RETRIEVAL_BUDGET_EXHAUSTED",
    "RETRIEVAL_CONFLICTING_ARGUMENTS",
    "RETRIEVAL_EMPTY_BATCH",
    "RETRIEVAL_INITIATORS",
    "RETRIEVAL_MALFORMED_REQUEST",
    "RETRIEVAL_REFUSAL_CODES",
    "RETRIEVAL_UNKNOWN_TOOL",
    "RETRIEVE_KIND",
    "RetrievalPlanner",
    "RetrievalRequest",
    "RetrieveBatch",
    "ToolResult",
    "canonical_requests",
    "retrieval_batch_problem",
    "tool_result_from_dict",
]
