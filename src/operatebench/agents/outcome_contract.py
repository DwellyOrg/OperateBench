"""The five Core outcomes as a field contract, stated once.

Three things in this build have to agree about what an ``ACT`` or an ``ASK``
consists of: the summary a model is shown in its prompt, the argument-name check
:func:`~operatebench.agents.model.parse_tool_call` applies to what comes back,
and — since the provider bridge exists — the JSON Schema a real provider is sent.
The first two were already derived from one table. The third could not be:
"required and optional field *names*" says nothing about whether
``fallback_after_minutes`` is a positive integer or whether ``evidence_refs`` is
an array of non-empty strings, so a hand-written schema beside the table would be
a second statement of the contract, free to drift from the one the parser
enforces — and the drift would be silent, because a schema that permits what the
parser refuses simply produces refusals nobody can explain.

So the contract is here, once, with a **kind** on every field, and everything
else is derived from it. :data:`OUTCOME_TOOLS` is the whole of it.

The kinds are named after what
:func:`~operatebench.core.outcomes.outcome_contract_problem` actually enforces,
not after what would read well:

* :data:`FIELD_TEXT` — any string, empty included. ``rationale`` and
  ``COMPLETE``'s ``reason`` are this: the contract asks only that they be
  strings.
* :data:`FIELD_NON_EMPTY_TEXT` — a string with at least one character. Every
  identifier is this, because an empty one names nothing.
* :data:`FIELD_REFERENCE_LIST` — an array of non-empty strings. Deliberately not
  "a sequence of strings": a bare string satisfies that and iterates into
  characters, which is the exact shape a model produces by answering
  ``evidence_refs: "evidence_1"``.
* :data:`FIELD_PAYLOAD_OBJECT` — a JSON object whose keys are non-empty strings.
* :data:`FIELD_POSITIVE_MINUTES` — a whole number of minutes, at least one. A
  deadline of zero or fewer is not a deadline.
* :data:`FIELD_WAIT` — a nested ``WAIT``, which is what an ``ASK`` carries so
  that asking cannot become the unbounded wait ``WAIT`` refuses.

``required`` on a field means *the parser requires the argument to be present*.
It is not the same question as JSON Schema's ``required``, and the two have
disagreed: under OpenAI's strict mode every declared property has to be listed
required whatever a contract says, so the projection stated one thing and this
table another. The projection is non-strict now and states this table's answer
directly — see :mod:`operatebench.agents.openai_responses`.

This module names no vendor and imports nothing but the standard library. It is
the contract, not a projection of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

#: Any string, empty included.
FIELD_TEXT = "text"
#: A string of at least one character.
FIELD_NON_EMPTY_TEXT = "non_empty_text"
#: An array of non-empty strings.
FIELD_REFERENCE_LIST = "reference_list"
#: A JSON object whose keys are non-empty strings.
FIELD_PAYLOAD_OBJECT = "payload_object"
#: A whole number of minutes, at least one.
FIELD_POSITIVE_MINUTES = "positive_minutes"
#: A nested ``WAIT``, stated by :data:`WAIT_FIELDS`.
FIELD_WAIT = "wait"
#: An array of retrieval requests, each naming one tool and its arguments.
#: Bounded by :data:`MIN_RETRIEVAL_REQUESTS` and :data:`MAX_RETRIEVAL_REQUESTS`.
FIELD_RETRIEVAL_BATCH = "retrieval_batch"

#: Every kind this contract uses. Closed: a field carrying a kind outside it
#: would be one no projection knows how to state and no parser knows how to
#: check.
FIELD_KINDS: tuple[str, ...] = (
    FIELD_TEXT,
    FIELD_NON_EMPTY_TEXT,
    FIELD_REFERENCE_LIST,
    FIELD_PAYLOAD_OBJECT,
    FIELD_POSITIVE_MINUTES,
    FIELD_WAIT,
    FIELD_RETRIEVAL_BATCH,
)

#: How many requests one retrieval batch may carry. Restated from
#: :mod:`operatebench.core.retrieval` deliberately *not* by import: this module
#: imports nothing but the standard library, and the two are held together by the
#: suite rather than by a dependency that would put Core's outcome vocabulary on
#: the provider projection's import path.
MIN_RETRIEVAL_REQUESTS = 1
MAX_RETRIEVAL_REQUESTS = 8

#: The one model-visible meaning of ``evidence_refs`` wherever an outcome offers
#: it. Generic to the operation and retrieval protocol: no action, scenario,
#: expected outcome or reference agent contributes words to this description.
EVIDENCE_REFS_DESCRIPTION = (
    "Optional. Cite only identifiers or keys of citable operation-held records "
    "learned from the records body of successful retrieval results. The selected "
    "action schema's evidence_refs.required descriptors state any exact citations "
    "that action requires. A retrieval result's top-level record_id, event IDs, "
    "tool names, source/service names, and read handles are not valid evidence "
    "references. Omit evidence_refs or use [] only when the selected action schema "
    "has no required citations."
)


@dataclass(frozen=True)
class OutcomeField:
    """One argument of one outcome tool: what it is called, is, and must be.

    ``required`` is the parser's question — "must this argument be present?" —
    and it is what :func:`required_field_names` and
    :func:`optional_field_names` partition on. ``description`` is shown to the
    model and is this build's own words, never a model's.
    """

    name: str
    kind: str
    required: bool
    description: str

    def __post_init__(self) -> None:
        if self.kind not in FIELD_KINDS:
            raise ValueError(
                f"{self.kind!r} is not an outcome field kind this contract defines; "
                f"allowed: {list(FIELD_KINDS)}"
            )


#: ``WAIT``'s own fields, named separately because an ``ASK`` carries one.
#: Stated once so the nested wait and the standalone one cannot disagree.
WAIT_FIELDS: tuple[OutcomeField, ...] = (
    OutcomeField(
        name="reason",
        kind=FIELD_TEXT,
        required=True,
        description="Why this invocation is waiting rather than acting.",
    ),
    OutcomeField(
        name="wake_on",
        kind=FIELD_REFERENCE_LIST,
        required=False,
        description=(
            "Event types that would end this wait. Either this or a fallback "
            "deadline must be stated; a wait with neither can never end."
        ),
    ),
    OutcomeField(
        name="fallback_after_minutes",
        kind=FIELD_POSITIVE_MINUTES,
        required=False,
        description=(
            "Simulated minutes after which this wait ends even if no awaited "
            "event arrives."
        ),
    ),
)


#: The whole contract: each outcome tool, and its fields in the order this build
#: states them — required first, then optional.
OUTCOME_TOOLS: Mapping[str, tuple[OutcomeField, ...]] = MappingProxyType(
    {
        "act": (
            OutcomeField(
                name="action_type",
                kind=FIELD_NON_EMPTY_TEXT,
                required=True,
                description="The authorised business action being proposed.",
            ),
            OutcomeField(
                name="payload",
                kind=FIELD_PAYLOAD_OBJECT,
                required=False,
                description="The action's arguments, as a JSON object.",
            ),
            OutcomeField(
                name="evidence_refs",
                kind=FIELD_REFERENCE_LIST,
                required=False,
                description=EVIDENCE_REFS_DESCRIPTION,
            ),
            OutcomeField(
                name="rationale",
                kind=FIELD_TEXT,
                required=False,
                description="Why this action is the right one now.",
            ),
        ),
        "wait": WAIT_FIELDS,
        "ask": (
            OutcomeField(
                name="recipient_actor_id",
                kind=FIELD_NON_EMPTY_TEXT,
                required=True,
                description="The actor being asked.",
            ),
            OutcomeField(
                name="message_fixture_id",
                kind=FIELD_NON_EMPTY_TEXT,
                required=True,
                description="The message fixture to send.",
            ),
            OutcomeField(
                name="wait",
                kind=FIELD_WAIT,
                required=True,
                description=(
                    "The wait this ask opens. Asking without declaring what the "
                    "answer looks like is an unbounded wait."
                ),
            ),
            OutcomeField(
                name="correlation_id",
                kind=FIELD_NON_EMPTY_TEXT,
                required=False,
                description="An identifier tying the answer back to this ask.",
            ),
        ),
        "escalate": (
            OutcomeField(
                name="checkpoint_id",
                kind=FIELD_NON_EMPTY_TEXT,
                required=True,
                description="The checkpoint this escalation opens.",
            ),
            OutcomeField(
                name="exception_type",
                kind=FIELD_NON_EMPTY_TEXT,
                required=True,
                description=(
                    "The typed exception being claimed. An untyped escalation "
                    "cannot be judged necessary or unnecessary."
                ),
            ),
            OutcomeField(
                name="evidence_refs",
                kind=FIELD_REFERENCE_LIST,
                required=False,
                description=EVIDENCE_REFS_DESCRIPTION,
            ),
            OutcomeField(
                name="deadline_after_minutes",
                kind=FIELD_POSITIVE_MINUTES,
                required=False,
                description=(
                    "Simulated minutes the checkpoint may stay open before it is "
                    "overdue. Defaults to 1440 when not stated."
                ),
            ),
            OutcomeField(
                name="rationale",
                kind=FIELD_TEXT,
                required=False,
                description="Why this escalation is necessary.",
            ),
        ),
        "complete": (
            OutcomeField(
                name="reason",
                kind=FIELD_TEXT,
                required=False,
                description="Why the operation is finished.",
            ),
            OutcomeField(
                name="evidence_refs",
                kind=FIELD_REFERENCE_LIST,
                required=False,
                description=EVIDENCE_REFS_DESCRIPTION,
            ),
        ),
    }
)

#: The tool names the five Core outcomes are offered under, in the order the
#: outcome contract states them.
TOOL_NAMES: tuple[str, ...] = ("act", "wait", "ask", "escalate", "complete")

#: The name the sixth object is offered under. Not an outcome: it is offered
#: beside the five because a model has to be able to *ask* for records, and it is
#: kept out of :data:`TOOL_NAMES` and :data:`OUTCOME_TOOLS` because everything
#: keyed by those is about what the operation records as business.
RETRIEVE_TOOL = "retrieve"

#: What ``retrieve`` takes: one array of requests and nothing else.
#:
#: ``tool`` is declared as a plain non-empty string rather than as an enumeration
#: of the vocabulary, and that is deliberate. The finite catalogue is *published
#: on every observation*, where the model reads it beside the read requirements
#: of each action; enumerating it here would make this module — which names no
#: vendor and knows no domain — carry one domain's tool names, and would put a
#: second statement of the vocabulary a step away from the one the environment
#: enforces. A name outside the catalogue is refused by the environment, by name,
#: and the refusal reaches the next observation.
RETRIEVE_FIELDS: tuple[OutcomeField, ...] = (
    OutcomeField(
        name="requests",
        kind=FIELD_RETRIEVAL_BATCH,
        required=True,
        description=(
            "The records to read, as an array of "
            f"{MIN_RETRIEVAL_REQUESTS} to {MAX_RETRIEVAL_REQUESTS} requests, each "
            "naming one tool from the published retrieval catalogue and that "
            "tool's arguments. The environment performs the reads and their "
            "results appear on the next observation of this invocation."
        ),
    ),
)

#: Every tool an agent may be offered: the five outcomes, then the read.
AGENT_TOOLS: Mapping[str, tuple[OutcomeField, ...]] = MappingProxyType(
    {**OUTCOME_TOOLS, RETRIEVE_TOOL: RETRIEVE_FIELDS}
)

AGENT_TOOL_NAMES: tuple[str, ...] = (*TOOL_NAMES, RETRIEVE_TOOL)

#: What each tool is, in one sentence, for the surface that shows the model a
#: description. This build's own words.
TOOL_DESCRIPTIONS: Mapping[str, str] = MappingProxyType(
    {
        "act": "Perform an authorised business action. A proposal, not a commit.",
        "wait": "Wait for declared events or a declared deadline.",
        "ask": "Request missing information from a named actor, and wait for it.",
        "escalate": "Open a justified human or exception checkpoint.",
        "complete": "Propose a guarded terminal transition.",
        RETRIEVE_TOOL: (
            "Read records from the published catalogue. Not a decision: it "
            "commits nothing and its results arrive on the next observation."
        ),
    }
)


def outcome_fields(tool: str) -> tuple[OutcomeField, ...]:
    """Every field one agent tool declares, in the order it declares them.

    Reads the whole offered surface, not only the five outcomes: the parser, the
    prompt summary and the provider schema are all derived from this, and a
    lookup that knew about one tool fewer than the surface offers is exactly the
    drift this module exists to make impossible.
    """
    try:
        return AGENT_TOOLS[tool]
    except KeyError as exc:
        raise KeyError(
            f"{tool!r} is not a tool this build offers; it offers "
            f"{list(AGENT_TOOL_NAMES)}"
        ) from exc


def required_field_names(tool: str) -> tuple[str, ...]:
    """The arguments the parser requires this outcome's call to carry."""
    return tuple(field.name for field in outcome_fields(tool) if field.required)


def optional_field_names(tool: str) -> tuple[str, ...]:
    """The arguments this outcome accepts and can do without."""
    return tuple(field.name for field in outcome_fields(tool) if not field.required)


def field_kinds(tool: str) -> Mapping[str, str]:
    """Each of this outcome's fields, by name, and what it must be."""
    return MappingProxyType({field.name: field.kind for field in outcome_fields(tool)})


__all__ = [
    "AGENT_TOOLS",
    "AGENT_TOOL_NAMES",
    "EVIDENCE_REFS_DESCRIPTION",
    "FIELD_KINDS",
    "FIELD_NON_EMPTY_TEXT",
    "FIELD_PAYLOAD_OBJECT",
    "FIELD_POSITIVE_MINUTES",
    "FIELD_REFERENCE_LIST",
    "FIELD_RETRIEVAL_BATCH",
    "FIELD_TEXT",
    "FIELD_WAIT",
    "MAX_RETRIEVAL_REQUESTS",
    "MIN_RETRIEVAL_REQUESTS",
    "OUTCOME_TOOLS",
    "RETRIEVE_FIELDS",
    "RETRIEVE_TOOL",
    "TOOL_DESCRIPTIONS",
    "TOOL_NAMES",
    "WAIT_FIELDS",
    "OutcomeField",
    "field_kinds",
    "optional_field_names",
    "outcome_fields",
    "required_field_names",
]
