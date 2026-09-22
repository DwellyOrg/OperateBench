"""The execution ledger: every provider attempt a run made, as it made them.

A run that reaches a provider produces two kinds of fact. The first is what it
*decided*, which the episode artefact already carries. The second is what it
*did* — how many requests left, which of them faulted and with what status, what
each one was authorised to cost and what it actually cost — and this build has
had nowhere to put it. A run excluded at the provider boundary has no episode
artefact by contract, so its attempts, its fault and its spend had no durable
home at all.

This module is that home, and it is a **journal** rather than a document.

**Append, flush, fsync, then call.** The header row is written before the first
request leaves; each call row is written as soon as the turn that produced it
ends; the terminal row is written last. A run that dies mid-flight therefore
leaves a *prefix* of verified rows rather than nothing — and, more importantly,
a prefix can never be read as a cheap successful run: :func:`read_execution_ledger`
reports ``"incomplete"`` for anything that does not end in a terminal row whose
own digest and totals verify.

A row is written *whole* or not at all. Short writes are looped over,
interruptions are retried, and a write or ``fsync`` that fails truncates the
partial row away and poisons the writer permanently — no later row, terminal
row or abandonment is appended through it. The journal is created against a
descriptor for the directory that was validated, opened component by component
from the root refusing symbolic links, so nothing that renames a directory
between the check and the write can choose where the evidence lands.

**Only states a run could have been in.** The rows are checked against each
other as well as against the chain: attempts are bounded by the run's own retry
policy and end at the answer, ``(invocation_index, turn_index)`` strictly
increases, a decision exists exactly where an answer this build read exists, and
a ``scored`` terminal is refused over any call that faulted or produced no
decision. The writer refuses such a row before appending it and the reader
re-derives the same judgements independently.

**Hash-chained.** Every row states its index, the digest of the row before it
and its own digest over its canonical form. Editing a row breaks its digest;
re-digesting the edited row breaks the next row's back-link; deleting, dropping
in or reordering rows breaks the index sequence. A truncated final line is kept
out of the chain entirely rather than half-read.

**No business result, no credential, no provider prose.** There is no field in
any row for a terminal outcome, an evaluation or a final state: an excluded run
must not become readable as a scored one by way of its evidence. Every string a
row carries is a digest, an instant, an exact decimal amount, or a value from a
closed vocabulary this build defines — and a recursive scanner refuses
credential-shaped material and unbounded text on the way in *and* on the way
out, so the schema and the scan have to agree before anything is written.

**Amounts are three separate totals and never one.** ``measured_cost_usd`` is
what the provider's own reported counts came to at the run's pinned rates;
``forfeited_reservation_usd`` is budget consumed for attempts whose actual cost
is unknown; ``reserved_usd`` is what was authorised before dispatch. Summing a
forfeited reservation into spend would report money nobody can show was spent,
so the three are carried apart and are never added together here.

**And every amount is derived rather than read.** A reservation is recomputed
from the call's own input bound, the run's output ceiling and the run's pinned
rates; a measured cost from the counts the provider reported at those same
rates; a forfeited reservation is the whole reservation. An edit to an amount,
a token count, a bound or a rate — re-chained so no digest can see it — does not
survive the arithmetic.

Nothing in this module opens a socket, reads a credential or imports a provider
SDK.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import secrets
import stat
from collections.abc import Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from types import TracebackType
from typing import Any

import operatebench.providers.faults as live_provider_faults
from operatebench.core.clock import parse_timestamp
from operatebench.core.errors import MalformedTimestampError, OperateBenchError
from operatebench.jsonsafe import JsonSafetyError, canonical_json_text
from operatebench.providers.cost import usd_text

#: Historical identity whose closed fault vocabulary predates Anthropic's three
#: durable HTTP 400 classifications. An independent literal, never an alias to
#: the live version, because historical meaning must not move with the writer.
EXECUTION_LEDGER_VERSION_V2 = 2

#: What a row of this journal states. Moved by any change to what a row says,
#: independently of the engine, the model protocol and the artefact. Version 3
#: adds the three closed Anthropic HTTP 400 fault classifications and gives
#: ``response_received`` one provider-neutral meaning: every typed HTTP-status
#: exception records that the SDK received a response.
EXECUTION_LEDGER_VERSION = 3
SUPPORTED_EXECUTION_LEDGER_VERSIONS: tuple[int, ...] = (
    EXECUTION_LEDGER_VERSION_V2,
    EXECUTION_LEDGER_VERSION,
)

# Provider-attempt outcomes, settlements, and turn endings are durable row
# semantics. Both supported versions own complete literals even where their
# values happen to agree today.
ATTEMPT_OUTCOME_FAULT = "fault"
ATTEMPT_OUTCOME_RESPONSE = "response"
ATTEMPT_OUTCOME_UNCLASSIFIED = "unclassified_error"
ATTEMPT_OUTCOMES_V2: tuple[str, ...] = (
    "fault",
    "response",
    "unclassified_error",
)
ATTEMPT_OUTCOMES_V3: tuple[str, ...] = (
    "fault",
    "response",
    "unclassified_error",
    *(),
)
ATTEMPT_OUTCOMES_BY_EXECUTION_LEDGER_VERSION: Mapping[int, tuple[str, ...]] = {
    EXECUTION_LEDGER_VERSION_V2: ATTEMPT_OUTCOMES_V2,
    EXECUTION_LEDGER_VERSION: ATTEMPT_OUTCOMES_V3,
}
ATTEMPT_SETTLEMENT_FORFEITED = "forfeited"
ATTEMPT_SETTLEMENT_MEASURED = "measured"
ATTEMPT_SETTLEMENTS_V2: tuple[str, ...] = (
    "forfeited",
    "measured",
)
ATTEMPT_SETTLEMENTS_V3: tuple[str, ...] = (
    "forfeited",
    "measured",
    *(),
)
ATTEMPT_SETTLEMENTS_BY_EXECUTION_LEDGER_VERSION: Mapping[int, tuple[str, ...]] = {
    EXECUTION_LEDGER_VERSION_V2: ATTEMPT_SETTLEMENTS_V2,
    EXECUTION_LEDGER_VERSION: ATTEMPT_SETTLEMENTS_V3,
}
TURN_END_BACKOFF_UNAFFORDABLE = "backoff_unaffordable"
TURN_END_COST_CAP_EXHAUSTED = "cost_cap_exhausted"
TURN_END_DEADLINE_EXCEEDED = "deadline_exceeded"
TURN_END_FAULT_NOT_RETRYABLE = "fault_not_retryable"
TURN_END_PRE_DISPATCH_REFUSED = "pre_dispatch_refused"
TURN_END_RESPONSE = "response"
TURN_END_RETRIES_EXHAUSTED = "retries_exhausted"
TURN_END_UNCLASSIFIED = "unclassified_error"
TURN_TERMINAL_REASONS_V2: tuple[str, ...] = (
    "backoff_unaffordable",
    "cost_cap_exhausted",
    "deadline_exceeded",
    "fault_not_retryable",
    "pre_dispatch_refused",
    "response",
    "retries_exhausted",
    "unclassified_error",
)
TURN_TERMINAL_REASONS_V3: tuple[str, ...] = (
    "backoff_unaffordable",
    "cost_cap_exhausted",
    "deadline_exceeded",
    "fault_not_retryable",
    "pre_dispatch_refused",
    "response",
    "retries_exhausted",
    "unclassified_error",
    *(),
)
TURN_TERMINAL_REASONS_BY_EXECUTION_LEDGER_VERSION: Mapping[int, tuple[str, ...]] = {
    EXECUTION_LEDGER_VERSION_V2: TURN_TERMINAL_REASONS_V2,
    EXECUTION_LEDGER_VERSION: TURN_TERMINAL_REASONS_V3,
}
TURN_END_EARLY_REASONS: tuple[str, ...] = (
    "backoff_unaffordable",
    "cost_cap_exhausted",
    "deadline_exceeded",
    "pre_dispatch_refused",
)

#: The exact historical v2 fault vocabulary. These are literals rather than
#: imports or aliases to the live provider taxonomy so later provider additions
#: cannot silently rewrite what a v2 row was allowed to state.
PROVIDER_FAULTS_V2: tuple[str, ...] = (
    "provider_authentication",
    "provider_configuration",
    "provider_network_error",
    "provider_rate_limited",
    "provider_request_rejected",
    "provider_response_invalid",
    "provider_server_error",
    "provider_timeout",
)
RETRYABLE_PROVIDER_FAULTS_V2: tuple[str, ...] = (
    "provider_network_error",
    "provider_rate_limited",
    "provider_server_error",
    "provider_timeout",
)
PROVIDER_FAULTS_V3: tuple[str, ...] = (
    "provider_authentication",
    "provider_bad_request_unclassified",
    "provider_configuration",
    "provider_invalid_request_or_spend_limit",
    "provider_network_error",
    "provider_rate_limited",
    "provider_request_rejected",
    "provider_response_invalid",
    "provider_server_error",
    "provider_spend_limit",
    "provider_timeout",
)
RETRYABLE_PROVIDER_FAULTS_V3: tuple[str, ...] = (
    "provider_network_error",
    "provider_rate_limited",
    "provider_server_error",
    "provider_timeout",
    *(),
)
PROVIDER_FAULTS_BY_EXECUTION_LEDGER_VERSION: Mapping[int, tuple[str, ...]] = {
    EXECUTION_LEDGER_VERSION_V2: PROVIDER_FAULTS_V2,
    EXECUTION_LEDGER_VERSION: PROVIDER_FAULTS_V3,
}
RETRYABLE_PROVIDER_FAULTS_BY_EXECUTION_LEDGER_VERSION: Mapping[int, tuple[str, ...]] = {
    EXECUTION_LEDGER_VERSION_V2: RETRYABLE_PROVIDER_FAULTS_V2,
    EXECUTION_LEDGER_VERSION: RETRYABLE_PROVIDER_FAULTS_V3,
}


def assert_current_provider_ledger_compatibility() -> None:
    """Fail closed when the live provider taxonomy outgrows ledger version 3."""
    if (
        live_provider_faults.PROVIDER_FAULTS != PROVIDER_FAULTS_V3
        or live_provider_faults.RETRYABLE_PROVIDER_FAULTS != RETRYABLE_PROVIDER_FAULTS_V3
    ):
        raise LedgerSchemaError(
            "the live provider taxonomy is not the taxonomy owned by execution "
            "ledger version 3; choose and implement a ledger-version decision "
            "before writing provider evidence"
        )


#: The artefact version a ledger written by this build is evidence *for*. The
#: binding has shipped since artifact 7 and is current in artifact 8; the ledger
#: schema is independently versioned, and this build's intended artifact 8 is
#: stated independently.
INTENDED_ARTIFACT_VERSION = 8

#: The back-link of the first row. A literal rather than ``None`` so every row
#: has the same shape and the chain check has no special case to forget.
GENESIS_DIGEST = "0" * 64

#: What every execution run identity starts with, on the same terms as
#: :data:`~operatebench.core.instance.OPERATION_INSTANCE_ID_PREFIX`: opaque,
#: random, and labelled so it cannot be read as a digest.
EXECUTION_RUN_ID_PREFIX = "exec_"
EXECUTION_RUN_ID_ENTROPY_BYTES = 16
EXECUTION_RUN_ID_BODY_LENGTH = EXECUTION_RUN_ID_ENTROPY_BYTES * 2

ROW_HEADER = "header"
ROW_CALL = "call"
ROW_TERMINAL = "terminal"
ROW_KINDS: tuple[str, ...] = (ROW_CALL, ROW_HEADER, ROW_TERMINAL)

#: The run reached a business terminal and its episode was evaluated.
TERMINAL_SCORED = "scored"
#: The run stopped at the provider boundary. It has no episode artefact, and
#: this ledger is the whole of its durable evidence.
TERMINAL_EXCLUDED = "excluded"
#: The run stopped for a reason that is neither of those — an exception this
#: build does not classify, or an operator stop — and was finalised while the
#: process was still alive.
TERMINAL_ABORTED = "aborted"
TERMINAL_KINDS: tuple[str, ...] = (TERMINAL_ABORTED, TERMINAL_EXCLUDED, TERMINAL_SCORED)

#: What a reader reports for a journal that has no verified terminal row.
STATUS_INCOMPLETE = "incomplete"

#: The exclusion code an aborted run carries. Its own value rather than a
#: provider fault: nothing about the provider is being asserted.
ABORTED_CODE = "execution_aborted"

#: The exclusion codes a terminal row may carry, and this build's own fixed
#: detail for each. Closed, and made of this build's words only, so no provider
#: text can reach a durable row through the one field that reads like prose.
TERMINAL_DETAILS: Mapping[str, str] = {
    TERMINAL_SCORED: "",
    TERMINAL_EXCLUDED: "execution stopped at the provider boundary",
    TERMINAL_ABORTED: "execution did not reach a terminal state and was abandoned",
}

#: Every code a terminal row may be excluded under. The four the Lifecycle model
#: boundary raises, plus this module's own abort code.
#:
#: Written out here rather than imported, because this module is composed by the
#: track and does not import it — and the suite asserts the two lists are the
#: same list, so a fault added there and not here is a test failure rather than a
#: value a ledger silently could not express.
#: The one exclusion whose cause is this build's own bound rather than anything
#: the provider did: the run was authorised for a number of calls and reached
#: it. Named here because it is the one code a terminal row may carry over calls
#: that all answered — there is no fault to point at, and the bound itself is
#: the record.
EXCLUSION_BUDGET = "provider_budget"

#: The run's clock ran out. Like the budget, a bound of this build's own rather
#: than anything the provider did.
EXCLUSION_DEADLINE = "provider_deadline"

EXCLUSION_CODES: tuple[str, ...] = (
    ABORTED_CODE,
    EXCLUSION_BUDGET,
    EXCLUSION_DEADLINE,
    "provider_protocol",
    "provider_transport",
)

#: The exclusions that name a control this build enforces rather than an outage
#: it observed. A run stopped by one of these need not carry a fault: the stop
#: happened *instead of* a request, so there is no attempt for it to appear on.
CONTROL_EXCLUSIONS: tuple[str, ...] = (EXCLUSION_BUDGET, EXCLUSION_DEADLINE)

#: How a cost cap is enforced. One value today, named rather than implied.
COST_CAP_POLICY_REFUSE = "refuse_before_dispatch"
COST_CAP_POLICIES: tuple[str, ...] = (COST_CAP_POLICY_REFUSE,)

#: How a response ended, in this build's vocabulary. Deliberately *not* the
#: provider's own status string: a value from a closed set is comparable across
#: runs and cannot carry text nobody reviewed.
STOP_COMPLETED = "completed"
STOP_TRUNCATED = "max_tokens"
STOP_CLASSIFICATIONS: tuple[str, ...] = (STOP_COMPLETED, STOP_TRUNCATED)

#: The longest string any ledger field may hold. Every field is a digest, an
#: instant, an amount or a closed-vocabulary value, so this is a second net
#: under the schema rather than the first: prose does not fit through it.
MAX_LEDGER_TEXT_CHARACTERS = 256

#: The one shape a bounded identifier may take. No whitespace, so a sentence
#: cannot be one however short it is.
_TOKEN = re.compile(rf"\A[A-Za-z0-9._:@/+-]{{1,{MAX_LEDGER_TEXT_CHARACTERS}}}\Z")
_HEX_DIGEST = re.compile(r"\A[0-9a-f]{64}\Z")
_HEX_BODY = re.compile(r"\A[0-9a-f]+\Z")
_URL = re.compile(r"\Ahttps://[A-Za-z0-9.-]+(?::\d+)?(?:/[A-Za-z0-9._~/-]*)?\Z")

#: Credential-shaped material, refused wherever it appears in a row — in a key
#: or in a value, at any depth. The scan is independent of the schema on
#: purpose: two nets, and a field added without a validator still meets one.
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("an Authorization header", re.compile(r"(?i)authoriz")),
    ("a bearer credential", re.compile(r"(?i)\bbearer\b")),
    ("an API key", re.compile(r"(?i)api[\s_-]?key")),
    ("a secret or token field", re.compile(r"(?i)\b(?:secret|passwd|password)\b")),
    ("an OpenAI-style key literal", re.compile(r"sk-[A-Za-z0-9_-]{8,}")),
)


class ExecutionLedgerError(OperateBenchError):
    """The execution ledger refused something. Every refusal below is one."""


class LedgerSchemaError(ExecutionLedgerError):
    """A value is not one this build's ledger schema accepts."""


class LedgerChainError(ExecutionLedgerError):
    """The journal's rows do not form the chain this build wrote."""


class LedgerSecretError(ExecutionLedgerError):
    """Something credential-shaped, or unbounded, reached a ledger row."""


class LedgerPathError(ExecutionLedgerError):
    """The journal's location is not one this build will write to."""


class LedgerStateError(ExecutionLedgerError):
    """The journal was asked for something its own state does not allow."""


class LedgerWriteError(ExecutionLedgerError):
    """A row could not be made durable, and the journal is finished.

    Raised for a short write, a write that accepted nothing, a write that
    failed outright and an ``fsync`` that failed after one. In every case the
    partial bytes are truncated away and the writer is poisoned permanently, so
    what is on disk is the verified prefix that preceded the failure and
    nothing is ever appended past it.

    :attr:`rolled_back` says whether the truncation itself succeeded. When it
    did not, the file's tail is bytes this build could neither complete nor
    remove, and saying so is the point: a rollback reported as done when the
    device refused it would be the same class of lie as an ``fsync`` reported
    as done when it raised.
    """

    def __init__(self, detail: str, *, rolled_back: bool) -> None:
        super().__init__(detail)
        self.rolled_back = rolled_back


# ------------------------------------------------------------------- primitives


def ledger_digest(payload: Any) -> str:
    """SHA-256 over canonical UTF-8 JSON. The one digest this module takes."""
    try:
        text = canonical_json_text(payload, "execution ledger row")
    except JsonSafetyError as exc:
        raise LedgerSchemaError(
            f"an execution ledger row could not be canonically encoded ({exc})"
        ) from exc
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def row_digest(row: Mapping[str, Any]) -> str:
    """The digest of one row, over everything in it except the digest itself."""
    return ledger_digest({k: v for k, v in row.items() if k != "row_digest_sha256"})


def new_execution_run_id() -> str:
    """Mint one execution run identity. Random, opaque, never derived."""
    body = secrets.token_hex(EXECUTION_RUN_ID_ENTROPY_BYTES)
    return f"{EXECUTION_RUN_ID_PREFIX}{body}"


def execution_run_id_problem(value: object) -> str | None:
    """Why this is not an execution run identity, or ``None`` if it is one.

    The non-raising form, for readers whose refusals are their own — the
    artefact reader states its refusals as artefact errors, and re-deriving the
    rule there would be the place the two silently diverge.
    """
    if not isinstance(value, str) or not value.startswith(EXECUTION_RUN_ID_PREFIX):
        return (
            f"an execution run identity is labelled {EXECUTION_RUN_ID_PREFIX!r} so "
            "it cannot be read as a digest"
        )
    body = value[len(EXECUTION_RUN_ID_PREFIX) :]
    if len(body) != EXECUTION_RUN_ID_BODY_LENGTH or not _HEX_BODY.match(body):
        return (
            f"an execution run identity carries a {EXECUTION_RUN_ID_BODY_LENGTH}"
            "-character lowercase hexadecimal body"
        )
    return None


def check_execution_run_id(value: object, context: str) -> str:
    problem = execution_run_id_problem(value)
    if problem is not None:
        raise LedgerSchemaError(f"{context}: {problem}")
    assert isinstance(value, str)
    return value


def assert_no_secret_material(payload: Any, context: str) -> None:
    """Refuse credential-shaped material and unbounded text, at any depth.

    Applied to every row before it is written and again when one is read. It
    knows nothing about the schema, which is the point: a field added without a
    validator is still met by this, and a validator loosened by accident is
    still met by this.
    """
    for where, text in _strings(payload, context):
        if len(text) > MAX_LEDGER_TEXT_CHARACTERS:
            raise LedgerSecretError(
                f"{where}: a ledger field holds {len(text)} characters against a "
                f"bound of {MAX_LEDGER_TEXT_CHARACTERS}. Every field here is a "
                "digest, an instant, an exact amount or a closed-vocabulary "
                "value; text this long is not one of them and is refused rather "
                "than stored"
            )
        for label, pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                raise LedgerSecretError(
                    f"{where}: this looks like {label}. No credential material "
                    "reaches an execution ledger, by schema and by this scan; "
                    "the offending value is not quoted here"
                )


def _strings(payload: Any, context: str, depth: int = 0) -> Iterator[tuple[str, str]]:
    if depth > 32:
        raise LedgerSchemaError(f"{context}: a ledger payload is nested too deeply")
    if isinstance(payload, str):
        yield context, payload
    elif isinstance(payload, Mapping):
        for key, value in payload.items():
            if not isinstance(key, str):
                raise LedgerSchemaError(f"{context}: a ledger field name is not text")
            yield f"{context}.{key}", key
            yield from _strings(value, f"{context}.{key}", depth + 1)
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            yield from _strings(value, f"{context}[{index}]", depth + 1)


# ------------------------------------------------------------------- validators


def _token(value: object, context: str) -> str:
    if not isinstance(value, str) or not _TOKEN.match(value):
        raise LedgerSchemaError(
            f"{context}: expected a bounded identifier with no whitespace, got "
            f"{type(value).__name__}"
        )
    return value


def _optional_token(value: object, context: str) -> str | None:
    return None if value is None else _token(value, context)


def _digest(value: object, context: str) -> str:
    if not isinstance(value, str) or not _HEX_DIGEST.match(value):
        raise LedgerSchemaError(
            f"{context}: expected a 64-character lowercase SHA-256 digest"
        )
    return value


def _optional_digest(value: object, context: str) -> str | None:
    return None if value is None else _digest(value, context)


def _count(value: object, context: str, *, minimum: int = 0) -> int:
    # ``bool`` is an ``int`` subclass, so the type test is exact.
    if type(value) is not int or value < minimum:
        raise LedgerSchemaError(
            f"{context}: expected an integer of at least {minimum}, got {value!r}"
        )
    return value


def _optional_count(value: object, context: str, *, minimum: int = 0) -> int | None:
    return None if value is None else _count(value, context, minimum=minimum)


def _flag(value: object, context: str) -> bool:
    if type(value) is not bool:
        raise LedgerSchemaError(f"{context}: expected true or false, got {value!r}")
    return value


def _seconds(value: object, context: str) -> float | None:
    if value is None:
        return None
    if type(value) is not float and type(value) is not int:
        raise LedgerSchemaError(f"{context}: expected a number of seconds or null")
    if value < 0:
        raise LedgerSchemaError(f"{context}: a duration is never negative")
    return float(value)


def _amount(value: object, context: str) -> str:
    """One exact USD amount, as the only rendering this build reports.

    A string, never a JSON number: a binary float is not the amount that was
    authorised. Canonical, so two reports of the same amount compare equal as
    strings rather than as parsed numbers.
    """
    if not isinstance(value, str):
        raise LedgerSchemaError(
            f"{context}: a USD amount is an exact decimal *string*, not a "
            f"{type(value).__name__}; a JSON number is a binary float to most "
            "readers and is not the amount that was authorised"
        )
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise LedgerSchemaError(f"{context}: {value!r} is not a decimal amount") from exc
    if not parsed.is_finite() or parsed < 0:
        raise LedgerSchemaError(f"{context}: a USD amount is finite and non-negative")
    if usd_text(parsed) != value:
        raise LedgerSchemaError(
            f"{context}: {value!r} is not the canonical rendering of the amount it "
            f"names, which is {usd_text(parsed)!r}"
        )
    return value


def _optional_amount(value: object, context: str) -> str | None:
    return None if value is None else _amount(value, context)


def _canonical_amount(value: object, context: str) -> str:
    """One operator-stated amount, re-rendered into this build's one rendering.

    Every amount this build *computes* is already canonical, so :func:`_amount`
    holds those to it exactly. A cap is typed by a person, and ``"1.00"`` and
    ``"1"`` are the same authorisation written two ways; refusing one of them
    would be pedantry rather than a control, and storing both would leave two
    renderings of one number in durable evidence.
    """
    if not isinstance(value, str):
        raise LedgerSchemaError(
            f"{context}: a USD amount is an exact decimal *string*, not a "
            f"{type(value).__name__}"
        )
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise LedgerSchemaError(f"{context}: {value!r} is not a decimal amount") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise LedgerSchemaError(
            f"{context}: a cost cap is a finite, positive amount; a cap of zero "
            "authorises nothing and is refused rather than read as 'no limit'"
        )
    return usd_text(parsed)


def _instant(value: object, context: str) -> str:
    try:
        parse_timestamp(value, context)
    except MalformedTimestampError as exc:
        raise LedgerSchemaError(str(exc)) from exc
    return str(value)


def _member(value: object, allowed: Sequence[str], context: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise LedgerSchemaError(
            f"{context}: {value!r} is not a value this build writes; allowed: "
            f"{list(allowed)}"
        )
    return value


def _optional_member(value: object, allowed: Sequence[str], context: str) -> str | None:
    return None if value is None else _member(value, allowed, context)


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LedgerSchemaError(f"{context}: expected a JSON object")
    plain = {str(key): _plain(item, f"{context}.{key}") for key, item in value.items()}
    assert_no_secret_material(plain, context)
    return plain


def _plain(value: object, context: str) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _plain(v, f"{context}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item, f"{context}[{index}]") for index, item in enumerate(value)]
    raise LedgerSchemaError(
        f"{context}: a ledger field holds a {type(value).__name__}, which is not a "
        "JSON value this build stores"
    )


def _exact_fields(
    payload: Mapping[str, Any],
    names: Sequence[str],
    context: str,
    *,
    ledger_version: int = EXECUTION_LEDGER_VERSION,
) -> None:
    got = set(map(str, payload))
    missing = sorted(set(names) - got)
    unknown = sorted(got - set(names))
    if missing:
        raise LedgerSchemaError(f"{context}: missing required field(s) {missing}")
    if unknown:
        raise LedgerSchemaError(
            f"{context}: field(s) {unknown} are not part of execution ledger version "
            f"{ledger_version}; an unknown field is refused rather than "
            "carried, because a reader cannot say what a field it does not know means"
        )


def _sum(amounts: Sequence[str]) -> str:
    """The exact total of a list of exact amounts, rendered canonically."""
    with localcontext() as context:
        context.prec = 60
        total = Decimal(0)
        for amount in amounts:
            total += Decimal(amount)
    return usd_text(total)


# ------------------------------------------------------------------ the records


ATTEMPT_FIELDS: tuple[str, ...] = (
    "attempt_index",
    "outcome",
    "fault",
    "http_status",
    "latency_seconds",
    "response_received",
    "usage_reported",
    "cost_reservation_usd",
    "cost_settlement",
    "cost_measured_usd",
    "cost_forfeited_usd",
    "input_tokens",
    "output_tokens",
    "response_model",
    "response_id_digest_sha256",
    "response_stop_classification",
    "response_normalized_digest_sha256",
)


@dataclass(frozen=True)
class LedgerAttempt:
    """One request to the provider: what it met, what it reported, what it cost.

    The kernel's own measurement, carried verbatim. ``fault`` is the *kernel*
    fault — ``provider_rate_limited``, ``provider_server_error``,
    ``provider_response_invalid`` — and not the single Lifecycle value those
    collapse into on the way to an execution record. That collapse is why an
    outage's cause was previously unrecoverable from durable evidence, so it is
    exactly what this row refuses to repeat.

    ``cost_measured_usd`` and ``cost_forfeited_usd`` are two fields rather than
    one signed one. A measured cost is what the provider's reported counts came
    to; a forfeited reservation is budget consumed for an attempt whose actual
    cost is unknown. They are never added together.
    """

    attempt_index: int
    outcome: str
    fault: str | None = None
    http_status: int | None = None
    latency_seconds: float | None = None
    response_received: bool = False
    usage_reported: bool = False
    cost_reservation_usd: str | None = None
    cost_settlement: str | None = None
    cost_measured_usd: str | None = None
    cost_forfeited_usd: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_model: str | None = None
    response_id_digest_sha256: str | None = None
    response_stop_classification: str | None = None
    response_normalized_digest_sha256: str | None = None

    def __post_init__(self) -> None:
        where = f"attempt {self.attempt_index}"
        _count(self.attempt_index, f"{where}.attempt_index", minimum=1)
        _token(self.outcome, f"{where}.outcome")
        _optional_token(self.fault, f"{where}.fault")
        _optional_count(self.http_status, f"{where}.http_status", minimum=100)
        _seconds(self.latency_seconds, f"{where}.latency_seconds")
        _flag(self.response_received, f"{where}.response_received")
        _flag(self.usage_reported, f"{where}.usage_reported")
        _optional_amount(self.cost_reservation_usd, f"{where}.cost_reservation_usd")
        _optional_token(self.cost_settlement, f"{where}.cost_settlement")
        _optional_amount(self.cost_measured_usd, f"{where}.cost_measured_usd")
        _optional_amount(self.cost_forfeited_usd, f"{where}.cost_forfeited_usd")
        _optional_count(self.input_tokens, f"{where}.input_tokens")
        _optional_count(self.output_tokens, f"{where}.output_tokens")
        _optional_token(self.response_model, f"{where}.response_model")
        _optional_digest(
            self.response_id_digest_sha256, f"{where}.response_id_digest_sha256"
        )
        _optional_member(
            self.response_stop_classification,
            STOP_CLASSIFICATIONS,
            f"{where}.response_stop_classification",
        )
        _optional_digest(
            self.response_normalized_digest_sha256,
            f"{where}.response_normalized_digest_sha256",
        )
        if (self.fault is not None) != (self.outcome == ATTEMPT_OUTCOME_FAULT):
            raise LedgerSchemaError(
                f"{where}: an attempt names a fault exactly when its outcome is "
                f"{ATTEMPT_OUTCOME_FAULT!r}; got outcome {self.outcome!r} and fault "
                f"{self.fault!r}"
            )
        if self.outcome == ATTEMPT_OUTCOME_RESPONSE and not self.response_received:
            raise LedgerSchemaError(
                f"{where}: an attempt whose outcome is a response received one"
            )
        if self.usage_reported and not self.response_received:
            raise LedgerSchemaError(
                f"{where}: counts were reported for an attempt that received no body. "
                "Usage is read out of a response, so an attempt with no response has "
                "none to report and the counts are refused rather than banked"
            )
        if (self.cost_reservation_usd is None) != (self.cost_settlement is None):
            raise LedgerSchemaError(
                f"{where}: a reservation and how it was resolved are stated together "
                "or not at all; a reservation with no settlement is money this run "
                "cannot account for"
            )
        measured = self.cost_settlement == ATTEMPT_SETTLEMENT_MEASURED
        forfeited = self.cost_settlement == ATTEMPT_SETTLEMENT_FORFEITED
        if measured != (self.cost_measured_usd is not None):
            raise LedgerSchemaError(
                f"{where}: a settlement of {ATTEMPT_SETTLEMENT_MEASURED!r} states the "
                "measured amount, and nothing else may"
            )
        if forfeited != (self.cost_forfeited_usd is not None):
            raise LedgerSchemaError(
                f"{where}: a settlement of {ATTEMPT_SETTLEMENT_FORFEITED!r} states the "
                "forfeited reservation, and nothing else may"
            )
        if measured and not self.usage_reported:
            raise LedgerSchemaError(
                f"{where}: a settlement of {ATTEMPT_SETTLEMENT_MEASURED!r} is a cost "
                "computed from the counts the provider reported, so an attempt that "
                "reported none cannot have measured one. An attempt whose counts this "
                f"build could not read settles as {ATTEMPT_SETTLEMENT_FORFEITED!r}"
            )
        counted = self.input_tokens is not None and self.output_tokens is not None
        if self.usage_reported != counted:
            raise LedgerSchemaError(
                f"{where}: usage_reported is true exactly when both token counts are "
                f"stated; got {self.usage_reported!r} with {self.input_tokens!r} in "
                f"and {self.output_tokens!r} out"
            )
        if self.outcome != ATTEMPT_OUTCOME_RESPONSE and any(
            value is not None
            for value in (
                self.response_model,
                self.response_id_digest_sha256,
                self.response_stop_classification,
                self.response_normalized_digest_sha256,
            )
        ):
            raise LedgerSchemaError(
                f"{where}: an attempt whose outcome is not {ATTEMPT_OUTCOME_RESPONSE!r} "
                "may not carry evidence read out of a response. Bytes arriving is not "
                "an answer: a body this build refused to read is recorded as the fault "
                "it was, and nothing is quoted out of it"
            )

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in ATTEMPT_FIELDS}

    def validate_for_ledger_version(self, ledger_version: int, context: str) -> None:
        """Validate this provisional DTO against the semantics its row owns."""
        try:
            allowed = PROVIDER_FAULTS_BY_EXECUTION_LEDGER_VERSION[ledger_version]
            outcomes = ATTEMPT_OUTCOMES_BY_EXECUTION_LEDGER_VERSION[ledger_version]
            settlements = ATTEMPT_SETTLEMENTS_BY_EXECUTION_LEDGER_VERSION[ledger_version]
        except KeyError as exc:
            raise LedgerSchemaError(
                f"{context}: execution ledger version {ledger_version!r} has no "
                "attempt semantics in this build"
            ) from exc
        _member(self.outcome, outcomes, f"{context}.outcome")
        _optional_member(self.cost_settlement, settlements, f"{context}.cost_settlement")
        if self.fault is not None and self.fault not in allowed:
            raise LedgerSchemaError(
                f"{context}.fault: {self.fault!r} is not owned by execution ledger "
                f"version {ledger_version}; allowed: {list(allowed)}"
            )
        if (
            ledger_version == EXECUTION_LEDGER_VERSION
            and self.http_status is not None
            and not self.response_received
        ):
            raise LedgerSchemaError(
                f"{context}: an HTTP status is typed response evidence in execution "
                "ledger version 3 and therefore requires response_received=true"
            )

    @classmethod
    def from_row(
        cls,
        payload: Mapping[str, Any],
        context: str,
        *,
        ledger_version: int = EXECUTION_LEDGER_VERSION,
    ) -> LedgerAttempt:
        _exact_fields(payload, ATTEMPT_FIELDS, context, ledger_version=ledger_version)
        attempt = cls(**{name: payload[name] for name in ATTEMPT_FIELDS})
        attempt.validate_for_ledger_version(ledger_version, context)
        return attempt


DECISION_FIELDS: tuple[str, ...] = (
    "kind",
    "decision_digest_sha256",
    "classification_code",
)


@dataclass(frozen=True)
class LedgerDecision:
    """What the run took from this call, as the decision tape records it.

    Digest only. The decision itself lives in the tape, and repeating it here
    would make the ledger a second, drifting source of truth for something the
    artefact already states exactly.
    """

    kind: str
    decision_digest_sha256: str
    classification_code: str | None = None

    def __post_init__(self) -> None:
        _token(self.kind, "decision.kind")
        _digest(self.decision_digest_sha256, "decision.decision_digest_sha256")
        _optional_token(self.classification_code, "decision.classification_code")

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in DECISION_FIELDS}

    @classmethod
    def from_row(cls, payload: Mapping[str, Any], context: str) -> LedgerDecision:
        _exact_fields(payload, DECISION_FIELDS, context)
        return cls(**{name: payload[name] for name in DECISION_FIELDS})


CALL_FIELDS: tuple[str, ...] = (
    "call_index",
    "invocation_index",
    "turn_index",
    "request_digest_sha256",
    "prompt_digest_sha256",
    "observation_digest_sha256",
    "payload_digest_sha256",
    "wire_request_digest_sha256",
    "wire_request_bytes",
    "wire_request_method",
    "wire_request_url",
    "input_token_upper_bound",
    "started_at",
    "ended_at",
    "turn_latency_seconds",
    "terminal_reason",
    "attempts",
    "decision",
)


@dataclass(frozen=True)
class LedgerCall:
    """One model call: the request that identified it, the bytes that left, the
    attempts it made and the decision it produced.

    ``request_digest_sha256`` and ``wire_request_digest_sha256`` on one row are
    the load-bearing pair. The first is over what this build *pinned* about the
    request; the second is over the exact body the SDK serialised. Carrying both
    is what turns "the prompt that was hashed is the prompt that was sent" from
    an assertion into something a reader can check.

    ``decision`` is ``None`` for a call that produced no decision, which is what
    a faulted call is. The reverse is refused: a decision recorded against a
    call whose attempts never received a response would be a decision with no
    provider answer behind it.
    """

    call_index: int
    invocation_index: int
    turn_index: int
    request_digest_sha256: str
    prompt_digest_sha256: str
    observation_digest_sha256: str
    payload_digest_sha256: str
    input_token_upper_bound: int
    started_at: str
    ended_at: str
    attempts: tuple[LedgerAttempt, ...]
    wire_request_digest_sha256: str | None = None
    wire_request_bytes: int | None = None
    wire_request_method: str | None = None
    wire_request_url: str | None = None
    turn_latency_seconds: float | None = None
    terminal_reason: str | None = None
    decision: LedgerDecision | None = None

    def __post_init__(self) -> None:
        where = f"call {self.call_index}"
        _count(self.call_index, f"{where}.call_index")
        _count(self.invocation_index, f"{where}.invocation_index")
        _count(self.turn_index, f"{where}.turn_index")
        for name in (
            "request_digest_sha256",
            "prompt_digest_sha256",
            "observation_digest_sha256",
            "payload_digest_sha256",
        ):
            _digest(getattr(self, name), f"{where}.{name}")
        _optional_digest(
            self.wire_request_digest_sha256, f"{where}.wire_request_digest_sha256"
        )
        _optional_count(self.wire_request_bytes, f"{where}.wire_request_bytes")
        _optional_member(
            self.wire_request_method, ("POST",), f"{where}.wire_request_method"
        )
        if self.wire_request_url is not None and not _URL.match(self.wire_request_url):
            raise LedgerSchemaError(
                f"{where}.wire_request_url: a recorded endpoint is a bare https "
                "scheme, host and path; no query and no credential material"
            )
        _count(self.input_token_upper_bound, f"{where}.input_token_upper_bound")
        _instant(self.started_at, f"{where}.started_at")
        _instant(self.ended_at, f"{where}.ended_at")
        _seconds(self.turn_latency_seconds, f"{where}.turn_latency_seconds")
        _optional_token(self.terminal_reason, f"{where}.terminal_reason")
        if parse_timestamp(self.ended_at, where) < parse_timestamp(
            self.started_at, where
        ):
            raise LedgerSchemaError(f"{where}: a call ended before it started")
        if not isinstance(self.attempts, tuple) or not self.attempts:
            raise LedgerSchemaError(
                f"{where}: a recorded call made at least one attempt; a call row with "
                "no attempts describes a request that was never dispatched"
            )
        for position, attempt in enumerate(self.attempts, start=1):
            if not isinstance(attempt, LedgerAttempt):
                raise LedgerSchemaError(f"{where}: attempt {position} is not an attempt")
            if attempt.attempt_index != position:
                raise LedgerSchemaError(
                    f"{where}: attempt indexes are contiguous from 1; attempt "
                    f"{position} states {attempt.attempt_index}"
                )
        wire = (
            self.wire_request_digest_sha256,
            self.wire_request_bytes,
            self.wire_request_method,
            self.wire_request_url,
        )
        if any(value is None for value in wire) and any(
            value is not None for value in wire
        ):
            raise LedgerSchemaError(
                f"{where}: the exact wire request is stated in full — digest, byte "
                "count, method and endpoint — or not at all"
            )
        if self.decision is not None:
            if not isinstance(self.decision, LedgerDecision):
                raise LedgerSchemaError(f"{where}.decision: not a ledger decision")
            self._check_decision_binding(where)

    def _check_attempt_sequence(
        self, where: str, retryable_faults: tuple[str, ...]
    ) -> None:
        """Every attempt but the last is one another attempt could follow.

        A response ends the turn: there is nothing left for a retry to change,
        so an attempt after one that answered describes a turn that cannot have
        happened. So does an attempt after a fault that repeating cannot answer
        differently — that set is stated in the kernel's own contract — and so
        does an attempt after an error this build never classified, which ends
        the turn where it happened.
        """
        for attempt in self.attempts[:-1]:
            if attempt.outcome == ATTEMPT_OUTCOME_RESPONSE:
                raise LedgerSchemaError(
                    f"{where}: attempt {attempt.attempt_index} received a response and "
                    "a later attempt is recorded after it. A turn ends at its answer, "
                    "so this is not a sequence of requests this build can have made"
                )
            if attempt.outcome == ATTEMPT_OUTCOME_UNCLASSIFIED:
                raise LedgerSchemaError(
                    f"{where}: attempt {attempt.attempt_index} ended in an error this "
                    "build could not classify, which ends the turn, and a later "
                    "attempt is recorded after it"
                )
            if attempt.fault is not None and attempt.fault not in retryable_faults:
                raise LedgerSchemaError(
                    f"{where}: attempt {attempt.attempt_index} met {attempt.fault!r}, "
                    "which repeating cannot answer differently, and a later attempt is "
                    "recorded after it"
                )

    def validate_for_ledger_version(self, ledger_version: int, context: str) -> None:
        """Apply only the attempt and retry semantics owned by this row version."""
        try:
            retryable = RETRYABLE_PROVIDER_FAULTS_BY_EXECUTION_LEDGER_VERSION[
                ledger_version
            ]
            terminal_reasons = TURN_TERMINAL_REASONS_BY_EXECUTION_LEDGER_VERSION[
                ledger_version
            ]
        except KeyError as exc:
            raise LedgerSchemaError(
                f"{context}: execution ledger version {ledger_version!r} has no "
                "call semantics in this build"
            ) from exc
        for index, attempt in enumerate(self.attempts):
            attempt.validate_for_ledger_version(
                ledger_version, f"{context}.attempts[{index}]"
            )
        if (
            self.terminal_reason is not None
            and self.terminal_reason not in terminal_reasons
        ):
            raise LedgerSchemaError(
                f"{context}.terminal_reason: {self.terminal_reason!r} is not owned by "
                f"execution ledger version {ledger_version}; allowed: "
                f"{list(terminal_reasons)}"
            )
        self._check_attempt_sequence(context, retryable)
        self._check_terminal_reason(context)

    def _check_terminal_reason(self, where: str) -> None:
        """The recorded exit has to be the exit the last attempt produced."""
        if self.terminal_reason is None:
            return
        last = self.attempts[-1]
        answered = last.outcome == ATTEMPT_OUTCOME_RESPONSE
        unclassified = last.outcome == ATTEMPT_OUTCOME_UNCLASSIFIED
        if answered != (self.terminal_reason == TURN_END_RESPONSE):
            raise LedgerSchemaError(
                f"{where}: the turn states it ended on {self.terminal_reason!r} and its "
                f"last attempt ended in {last.outcome!r}. A turn's exit is the exit its "
                "last attempt produced, and a row stating otherwise describes two turns"
            )
        if unclassified != (self.terminal_reason == TURN_END_UNCLASSIFIED):
            raise LedgerSchemaError(
                f"{where}: the turn states it ended on {self.terminal_reason!r} and its "
                f"last attempt ended in {last.outcome!r}"
            )

    def _check_decision_binding(self, where: str) -> None:
        """A decision is what the run read out of an answer it actually got.

        The last attempt is the one whose answer a decision can have come from,
        and it has to be an answer: a body that arrived and was refused —
        ``provider_response_invalid`` — produced no decision, and neither did a
        transport fault. The normalized digest is required with it, because
        without it nothing on the row ties the decision to the answer beside it.
        """
        last = self.attempts[-1]
        if last.outcome != ATTEMPT_OUTCOME_RESPONSE:
            raise LedgerSchemaError(
                f"{where}: a decision is recorded against a call whose last attempt "
                f"ended in {last.outcome!r}"
                + (f" ({last.fault})" if last.fault is not None else "")
                + ". A decision with no provider answer behind it is not evidence of "
                "anything this run did"
            )
        if last.response_normalized_digest_sha256 is None:
            raise LedgerSchemaError(
                f"{where}: a decision is recorded against an answer this row states no "
                "normalized digest for, so nothing binds the decision to the answer it "
                "is claimed to have come from"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            **{
                name: getattr(self, name)
                for name in CALL_FIELDS
                if name not in ("attempts", "decision")
            },
            "attempts": [attempt.as_dict() for attempt in self.attempts],
            "decision": None if self.decision is None else self.decision.as_dict(),
        }

    @classmethod
    def from_row(
        cls,
        payload: Mapping[str, Any],
        context: str,
        *,
        ledger_version: int = EXECUTION_LEDGER_VERSION,
    ) -> LedgerCall:
        _exact_fields(payload, CALL_FIELDS, context, ledger_version=ledger_version)
        raw_attempts = payload["attempts"]
        if not isinstance(raw_attempts, list):
            raise LedgerSchemaError(f"{context}.attempts: expected an array")
        attempts: list[LedgerAttempt] = []
        for index, entry in enumerate(raw_attempts):
            where = f"{context}.attempts[{index}]"
            if not isinstance(entry, Mapping):
                raise LedgerSchemaError(f"{where}: expected a JSON object")
            _exact_fields(entry, ATTEMPT_FIELDS, where, ledger_version=ledger_version)
            attempts.append(
                LedgerAttempt(**{name: entry[name] for name in ATTEMPT_FIELDS})
            )
        raw_decision = payload["decision"]
        decision: LedgerDecision | None = None
        if raw_decision is not None:
            if not isinstance(raw_decision, Mapping):
                raise LedgerSchemaError(f"{context}.decision: expected a JSON object")
            decision = LedgerDecision.from_row(raw_decision, f"{context}.decision")
        fields = {
            name: payload[name]
            for name in CALL_FIELDS
            if name not in ("attempts", "decision")
        }
        call = cls(**fields, attempts=tuple(attempts), decision=decision)
        call.validate_for_ledger_version(ledger_version, context)
        return call


PROVIDER_FIELDS: tuple[str, ...] = (
    "provider",
    "api",
    "model",
    "base_url",
    "implementation",
    "request_mapping",
    "protocol_version",
    "sdk",
    "sdk_version",
    "settings",
    "settings_digest_sha256",
)


@dataclass(frozen=True)
class LedgerProviderIdentity:
    """Who was asked, through what, under exactly which settings.

    ``settings`` is the transport's own mapping, carried whole rather than
    summarised, and ``settings_digest_sha256`` is derived from it here rather
    than supplied: a digest a caller could state independently of the mapping it
    covers is a digest that can be made to agree with a mapping it does not
    describe.
    """

    provider: str
    api: str
    model: str
    base_url: str
    implementation: str
    request_mapping: str
    protocol_version: str
    sdk: str
    sdk_version: str
    settings: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "provider",
            "api",
            "model",
            "implementation",
            "request_mapping",
            "protocol_version",
            "sdk",
            "sdk_version",
        ):
            _token(getattr(self, name), f"provider.{name}")
        if not _URL.match(self.base_url):
            raise LedgerSchemaError(
                "provider.base_url: expected a bare https scheme, host and path"
            )
        object.__setattr__(self, "settings", _mapping(self.settings, "provider.settings"))

    @property
    def settings_digest_sha256(self) -> str:
        """The digest of the exact settings mapping this run asked under."""
        return ledger_digest(dict(self.settings))

    def as_dict(self) -> dict[str, Any]:
        return {
            **{
                name: getattr(self, name)
                for name in PROVIDER_FIELDS
                if name not in ("settings", "settings_digest_sha256")
            },
            "settings": dict(self.settings),
            "settings_digest_sha256": self.settings_digest_sha256,
        }

    @classmethod
    def from_row(cls, payload: Mapping[str, Any], context: str) -> LedgerProviderIdentity:
        _exact_fields(payload, PROVIDER_FIELDS, context)
        identity = cls(
            **{
                name: payload[name]
                for name in PROVIDER_FIELDS
                if name != "settings_digest_sha256"
            }
        )
        stated = _digest(payload["settings_digest_sha256"], f"{context}.settings_digest")
        if stated != identity.settings_digest_sha256:
            raise LedgerChainError(
                f"{context}: the stated settings digest is not the digest of the "
                "settings mapping beside it. A run's settings are what its answers "
                "were produced under, so a digest that does not cover them is refused"
            )
        return identity


#: What a run's pinned rate table has to state for an amount computed under it
#: to be re-derivable by a reader. The two rates are the arithmetic; the policy
#: identity and the rate source are what an amount is attributed to.
PRICING_RATE_FIELDS: tuple[str, ...] = ("input_usd_per_mtok", "output_usd_per_mtok")

#: Tokens a rate is quoted per. Stated here as well as in the pricing policy —
#: this module does not import it — and the suite asserts the two agree.
TOKENS_PER_RATE_UNIT = 1_000_000

#: The precision every amount here is derived under. Well above what a run's
#: amounts can need: a derived total that rounded would be a check against a
#: number that is nobody's arithmetic.
LEDGER_PRICING_PRECISION = 60

CONTROLS_FIELDS: tuple[str, ...] = (
    "max_provider_calls",
    "max_attempts_per_call",
    "expected_provider_calls",
    "token_hard_cap",
    "max_output_tokens",
    "cost_cap_usd",
    "cost_cap_policy",
    "turn_deadline_seconds",
    "wall_clock_deadline_seconds",
    "pricing",
    "pricing_digest_sha256",
)


@dataclass(frozen=True)
class LedgerControls:
    """The bounds a run was authorised to execute under.

    ``max_provider_calls`` is a safety bound and not a forecast:
    ``expected_provider_calls`` is what the scenario is measured to need, and
    the two are carried apart so a reader cannot mistake head-room for a
    prediction.

    ``pricing`` is the run's own pinned rate table, digested here for the same
    reason the settings are: an amount is only meaningful beside the rates it
    was computed at.

    ``max_output_tokens`` is the output ceiling every request pinned, and it is
    stated rather than left implicit inside ``token_hard_cap`` because it is
    half of every reservation the run ever authorised: without it a reader can
    add the recorded reservations up but cannot say whether any of them is the
    amount the run was authorised for.
    """

    max_provider_calls: int
    max_attempts_per_call: int
    expected_provider_calls: int
    token_hard_cap: int
    max_output_tokens: int
    cost_cap_usd: str
    cost_cap_policy: str
    turn_deadline_seconds: float
    wall_clock_deadline_seconds: float
    pricing: Mapping[str, Any]

    def __post_init__(self) -> None:
        _count(self.max_provider_calls, "controls.max_provider_calls", minimum=1)
        _count(self.max_attempts_per_call, "controls.max_attempts_per_call", minimum=1)
        _count(
            self.expected_provider_calls, "controls.expected_provider_calls", minimum=1
        )
        _count(self.token_hard_cap, "controls.token_hard_cap", minimum=1)
        _count(self.max_output_tokens, "controls.max_output_tokens", minimum=1)
        object.__setattr__(
            self,
            "cost_cap_usd",
            _canonical_amount(self.cost_cap_usd, "controls.cost_cap_usd"),
        )
        _member(self.cost_cap_policy, COST_CAP_POLICIES, "controls.cost_cap_policy")
        _seconds(self.turn_deadline_seconds, "controls.turn_deadline_seconds")
        _seconds(self.wall_clock_deadline_seconds, "controls.wall_clock_deadline_seconds")
        if self.expected_provider_calls > self.max_provider_calls:
            raise LedgerSchemaError(
                "controls: the expected call count exceeds the safety bound, which "
                "would make the bound a forecast the run is already known to break"
            )
        object.__setattr__(self, "pricing", _mapping(self.pricing, "controls.pricing"))
        for name in PRICING_RATE_FIELDS:
            self.rate(name)

    def rate(self, name: str) -> Decimal:
        """One pinned rate, as the exact decimal it was stated as.

        Read rather than remembered. Every amount a reader re-derives comes
        through here, so a table missing a rate — or holding one as a JSON
        number, which is a binary float to most readers — is refused at the
        header rather than at the first amount that depends on it.
        """
        value = dict(self.pricing).get(name)
        if not isinstance(value, str):
            raise LedgerSchemaError(
                f"controls.pricing.{name}: a run's pinned rate is an exact decimal "
                "string. Without it no amount on any row below can be re-derived, so "
                "the table is refused rather than carried"
            )
        try:
            rate = Decimal(value)
        except InvalidOperation as exc:
            raise LedgerSchemaError(
                f"controls.pricing.{name}: {value!r} is not a decimal rate"
            ) from exc
        if not rate.is_finite() or rate < 0:
            raise LedgerSchemaError(
                f"controls.pricing.{name}: a rate is finite and non-negative"
            )
        return rate

    @property
    def pricing_digest_sha256(self) -> str:
        return ledger_digest(dict(self.pricing))

    def as_dict(self) -> dict[str, Any]:
        return {
            **{
                name: getattr(self, name)
                for name in CONTROLS_FIELDS
                if name not in ("pricing", "pricing_digest_sha256")
            },
            "pricing": dict(self.pricing),
            "pricing_digest_sha256": self.pricing_digest_sha256,
        }

    @classmethod
    def from_row(cls, payload: Mapping[str, Any], context: str) -> LedgerControls:
        _exact_fields(payload, CONTROLS_FIELDS, context)
        controls = cls(
            **{
                name: payload[name]
                for name in CONTROLS_FIELDS
                if name != "pricing_digest_sha256"
            }
        )
        stated = _digest(payload["pricing_digest_sha256"], f"{context}.pricing_digest")
        if stated != controls.pricing_digest_sha256:
            raise LedgerChainError(
                f"{context}: the stated pricing digest is not the digest of the rate "
                "table beside it; an amount is only meaningful beside the rates it "
                "was computed at"
            )
        return controls


HEADER_FIELDS: tuple[str, ...] = (
    "execution_run_id",
    "operation_instance_id",
    "operation_id",
    "spec_digest_sha256",
    "scenario_id",
    "agent_id",
    "agent_kind",
    "engine_version",
    "intended_artifact_version",
    "started_at",
    "provider",
    "controls",
)


@dataclass(frozen=True)
class LedgerHeader:
    """What this run is, before it has done anything.

    Written before the first request leaves, so a journal that exists at all
    says which run made it — and a journal that exists with no calls in it is a
    run that was authorised and then did nothing, which is a different fact from
    a run nobody recorded.
    """

    execution_run_id: str
    operation_instance_id: str
    operation_id: str
    spec_digest_sha256: str
    scenario_id: str
    agent_id: str
    agent_kind: str
    started_at: str
    provider: LedgerProviderIdentity
    controls: LedgerControls
    intended_artifact_version: int = INTENDED_ARTIFACT_VERSION
    engine_version: str = ""

    def __post_init__(self) -> None:
        from operatebench.core.instance import operation_instance_id_problem
        from operatebench.version import OPERATEBENCH_VERSION

        check_execution_run_id(self.execution_run_id, "header.execution_run_id")
        problem = operation_instance_id_problem(self.operation_instance_id)
        if problem is not None:
            raise LedgerSchemaError(f"header.operation_instance_id {problem}")
        for name in ("operation_id", "scenario_id", "agent_id", "agent_kind"):
            _token(getattr(self, name), f"header.{name}")
        _digest(self.spec_digest_sha256, "header.spec_digest_sha256")
        _instant(self.started_at, "header.started_at")
        if not self.engine_version:
            object.__setattr__(self, "engine_version", OPERATEBENCH_VERSION)
        _token(self.engine_version, "header.engine_version")
        if self.intended_artifact_version != INTENDED_ARTIFACT_VERSION:
            raise LedgerSchemaError(
                f"header.intended_artifact_version: this build keeps provider "
                f"execution evidence against artefact {INTENDED_ARTIFACT_VERSION}, "
                f"got {self.intended_artifact_version!r}"
            )
        if not isinstance(self.provider, LedgerProviderIdentity):
            raise LedgerSchemaError("header.provider: not a provider identity")
        if not isinstance(self.controls, LedgerControls):
            raise LedgerSchemaError("header.controls: not a control block")

    def as_dict(self) -> dict[str, Any]:
        return {
            **{
                name: getattr(self, name)
                for name in HEADER_FIELDS
                if name not in ("provider", "controls")
            },
            "provider": self.provider.as_dict(),
            "controls": self.controls.as_dict(),
        }

    @classmethod
    def from_row(cls, payload: Mapping[str, Any], context: str) -> LedgerHeader:
        _exact_fields(payload, HEADER_FIELDS, context)
        raw_provider, raw_controls = payload["provider"], payload["controls"]
        if not isinstance(raw_provider, Mapping) or not isinstance(raw_controls, Mapping):
            raise LedgerSchemaError(f"{context}: provider and controls are objects")
        return cls(
            **{
                name: payload[name]
                for name in HEADER_FIELDS
                if name not in ("provider", "controls")
            },
            provider=LedgerProviderIdentity.from_row(raw_provider, f"{context}.provider"),
            controls=LedgerControls.from_row(raw_controls, f"{context}.controls"),
        )


# ------------------------------------------------- deriving what a row claims


def _priced(controls: LedgerControls, *, input_tokens: int, output_tokens: int) -> str:
    """Tokens at this run's pinned rates, as this build's one rendering.

    Written out here rather than imported from the pricing policy on purpose.
    A reader that re-derived an amount by calling the same object that computed
    it would be checking that a function agrees with itself; what makes a
    recorded amount evidence is that a second, independent statement of the
    arithmetic — over the run's own recorded tokens, rates and bounds — lands on
    the same number. The suite holds the two implementations to each other.
    """
    with localcontext() as context:
        context.prec = LEDGER_PRICING_PRECISION
        amount = (
            Decimal(input_tokens) * controls.rate("input_usd_per_mtok")
            + Decimal(output_tokens) * controls.rate("output_usd_per_mtok")
        ) / Decimal(TOKENS_PER_RATE_UNIT)
    return usd_text(amount)


def derive_reservation_usd(
    *, input_token_upper_bound: int, controls: LedgerControls
) -> str:
    """What a call of this size was authorised to cost, before it was made.

    The conservative bound: this call's own upper bound on input tokens, the
    whole of the run's output ceiling, at the run's pinned rates. Nothing about
    the recorded amount is consulted, which is the point — a reservation that
    was edited after the fact does not survive being recomputed from the
    numbers that produced it.
    """
    return _priced(
        controls,
        input_tokens=_count(input_token_upper_bound, "input_token_upper_bound"),
        output_tokens=controls.max_output_tokens,
    )


def derive_measured_usd(
    *, input_tokens: int, output_tokens: int, controls: LedgerControls
) -> str:
    """What the counts a provider reported come to at the run's pinned rates."""
    return _priced(
        controls,
        input_tokens=_count(input_tokens, "input_tokens"),
        output_tokens=_count(output_tokens, "output_tokens"),
    )


def check_call_amounts(call: LedgerCall, *, controls: LedgerControls) -> None:
    """Refuse a call row whose amounts do not follow from its own numbers.

    Three checks, and each of them is arithmetic rather than a comparison
    between two recorded fields:

    * a reservation is the call's input bound and the run's output ceiling at
      the run's rates;
    * a measured cost is the counts the provider reported at those same rates;
    * a forfeited reservation is the whole reservation, because that is what
      forfeiting one means — it is exposure the run must account for and never
      a measurement of what was spent.
    """
    where = f"call {call.call_index}"
    for attempt in call.attempts:
        stated = attempt.cost_reservation_usd
        if stated is not None:
            expected = derive_reservation_usd(
                input_token_upper_bound=call.input_token_upper_bound, controls=controls
            )
            if stated != expected:
                raise LedgerChainError(
                    f"{where} attempt {attempt.attempt_index}: states a reservation of "
                    f"{stated} USD where this call's own input bound of "
                    f"{call.input_token_upper_bound} token(s), the run's output ceiling "
                    f"of {controls.max_output_tokens} and the run's pinned rates come "
                    f"to {expected} USD. An amount that does not follow from the "
                    "numbers beside it is a claim rather than a measurement"
                )
        if attempt.cost_settlement == ATTEMPT_SETTLEMENT_MEASURED:
            if attempt.input_tokens is None or attempt.output_tokens is None:
                raise LedgerChainError(
                    f"{where} attempt {attempt.attempt_index}: settles as measured with "
                    "no counts to have measured it from"
                )
            expected = derive_measured_usd(
                input_tokens=attempt.input_tokens,
                output_tokens=attempt.output_tokens,
                controls=controls,
            )
            if attempt.cost_measured_usd != expected:
                raise LedgerChainError(
                    f"{where} attempt {attempt.attempt_index}: states a measured cost "
                    f"of {attempt.cost_measured_usd} USD where its own reported counts "
                    f"({attempt.input_tokens} in, {attempt.output_tokens} out) at the "
                    f"run's pinned rates come to {expected} USD"
                )
        if (
            attempt.cost_settlement == ATTEMPT_SETTLEMENT_FORFEITED
            and attempt.cost_forfeited_usd != stated
        ):
            raise LedgerChainError(
                f"{where} attempt {attempt.attempt_index}: forfeits "
                f"{attempt.cost_forfeited_usd} USD of a {stated} USD reservation. "
                "A forfeited reservation is the whole reservation: it is the "
                "exposure of an attempt whose actual cost is unknown, and a part "
                "of one is a number nothing measured"
            )


def check_call_admissible(
    call: LedgerCall,
    *,
    controls: LedgerControls,
    previous: LedgerCall | None,
) -> None:
    """Refuse a call row this run cannot have produced, before or after writing.

    What :class:`LedgerCall` cannot decide on its own: how many attempts the
    run's policy allowed, how many calls it was authorised to make, and where
    this call sits relative to the one before it. The coordinates
    ``(invocation_index, turn_index)`` strictly increase, so a repeated turn —
    two rows claiming one moment of one run — and a re-ordered one are refused
    rather than counted twice.
    """
    where = f"call {call.call_index}"
    if call.call_index >= controls.max_provider_calls:
        raise LedgerChainError(
            f"{where}: records a call against a bound of "
            f"{controls.max_provider_calls}. The bound is what the run was authorised "
            "to make, so a call past it is not a call this run could have made"
        )
    if len(call.attempts) > controls.max_attempts_per_call:
        raise LedgerChainError(
            f"{where}: records {len(call.attempts)} attempt(s) under a policy that "
            f"allows {controls.max_attempts_per_call}. A turn cannot have made a "
            "request its own run was not authorised to retry"
        )
    if call.terminal_reason == TURN_END_RETRIES_EXHAUSTED and (
        len(call.attempts) != controls.max_attempts_per_call
    ):
        raise LedgerChainError(
            f"{where}: states it exhausted its retries after {len(call.attempts)} of "
            f"{controls.max_attempts_per_call} permitted attempt(s). A turn that "
            "stopped with attempts left stopped for a reason it has to name"
        )
    if previous is None:
        return
    if (call.invocation_index, call.turn_index) <= (
        previous.invocation_index,
        previous.turn_index,
    ):
        raise LedgerChainError(
            f"{where}: states invocation {call.invocation_index} turn "
            f"{call.turn_index} after invocation {previous.invocation_index} turn "
            f"{previous.turn_index}. A run's turns move forwards and happen once, so a "
            "repeated or re-ordered coordinate is refused rather than recorded twice"
        )


def check_terminal_admissible(
    kind: str, exclusion_code: str | None, calls: Sequence[LedgerCall]
) -> None:
    """Refuse an ending the rows above it do not support.

    A ``scored`` run reached a business terminal, which means every call it
    made answered and produced a decision: one faulted call, one call with no
    decision, or one error this build never classified and the run did not
    reach one.

    An ``excluded`` run stopped, and where the code it stopped under names
    something the *provider* did, the rows have to show it: a fault, an
    unclassified attempt, or a turn that ran out before it could dispatch. The
    two codes that name this build's *own* bounds instead —
    :data:`EXCLUSION_BUDGET` and the deadline — need no such witness, because
    the thing that stopped the run is a control rather than an outage and the
    control leaves no row. Neither does a journal with no calls in it: nothing
    there contradicts a stop, because nothing there happened.

    Neither ``excluded`` nor ``aborted`` is ever scored, and there is no field
    on a terminal row that could make one look it.
    """
    if kind == TERMINAL_SCORED:
        for call in calls:
            if any(attempt.fault is not None for attempt in call.attempts):
                raise LedgerChainError(
                    f"terminal: a scored run states call {call.call_index} met "
                    f"{call.attempts[-1].fault!r}. A run that reached an evaluated "
                    "episode made no call it did not get an answer to, and a ledger "
                    "that says otherwise is not evidence of a scored run"
                )
            if any(
                attempt.outcome == ATTEMPT_OUTCOME_UNCLASSIFIED
                for attempt in call.attempts
            ):
                raise LedgerChainError(
                    f"terminal: a scored run states call {call.call_index} ended in an "
                    "error this build could not classify"
                )
            if call.decision is None:
                raise LedgerChainError(
                    f"terminal: a scored run states call {call.call_index} produced no "
                    "decision. Every call of a run that reached a business terminal is "
                    "a call the run took something from"
                )
        return
    if kind == TERMINAL_EXCLUDED:
        if exclusion_code == ABORTED_CODE:
            raise LedgerChainError(
                f"terminal: {ABORTED_CODE!r} is this module's own code for a run that "
                "was abandoned, and an excluded run states the code it was excluded "
                "under"
            )
        if exclusion_code in CONTROL_EXCLUSIONS or not calls:
            return
        if any(
            attempt.fault is not None or attempt.outcome == ATTEMPT_OUTCOME_UNCLASSIFIED
            for call in calls
            for attempt in call.attempts
        ):
            return
        if any(call.terminal_reason in TURN_END_EARLY_REASONS for call in calls):
            return
        raise LedgerChainError(
            f"terminal: a run excluded under {exclusion_code!r} records no fault, no "
            "unclassified attempt and no turn stopped by a local pre-dispatch "
            "control. An "
            "exclusion code with nothing behind it names a stop nothing in this "
            "journal witnessed"
        )


TOTALS_FIELDS: tuple[str, ...] = (
    "provider_calls",
    "attempts",
    "input_tokens",
    "output_tokens",
    "measured_cost_usd",
    "forfeited_reservation_usd",
    "reserved_usd",
    "fault_counts",
)


@dataclass(frozen=True)
class LedgerTotals:
    """What the call rows add up to. Three amounts, and never one.

    Re-derived from the rows whenever a ledger is read, and refused if the
    stated totals do not follow from them: a total that cannot be rebuilt from
    its own parts is a claim rather than a measurement.
    """

    provider_calls: int
    attempts: int
    input_tokens: int
    output_tokens: int
    measured_cost_usd: str
    forfeited_reservation_usd: str
    reserved_usd: str
    fault_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        for name in ("provider_calls", "attempts", "input_tokens", "output_tokens"):
            _count(getattr(self, name), f"totals.{name}")
        for name in ("measured_cost_usd", "forfeited_reservation_usd", "reserved_usd"):
            _amount(getattr(self, name), f"totals.{name}")
        counts = _mapping(self.fault_counts, "totals.fault_counts")
        for fault, count in counts.items():
            _token(fault, "totals.fault_counts")
            _count(count, f"totals.fault_counts.{fault}", minimum=1)
        object.__setattr__(self, "fault_counts", dict(sorted(counts.items())))

    def as_dict(self) -> dict[str, Any]:
        return {
            **{
                name: getattr(self, name)
                for name in TOTALS_FIELDS
                if name != "fault_counts"
            },
            "fault_counts": dict(self.fault_counts),
        }

    @classmethod
    def from_row(
        cls,
        payload: Mapping[str, Any],
        context: str,
        *,
        ledger_version: int = EXECUTION_LEDGER_VERSION,
    ) -> LedgerTotals:
        _exact_fields(payload, TOTALS_FIELDS, context, ledger_version=ledger_version)
        totals = cls(**{name: payload[name] for name in TOTALS_FIELDS})
        try:
            allowed = PROVIDER_FAULTS_BY_EXECUTION_LEDGER_VERSION[ledger_version]
        except KeyError as exc:
            raise LedgerSchemaError(
                f"{context}: execution ledger version {ledger_version!r} has no "
                "provider-fault vocabulary in this build"
            ) from exc
        for fault in totals.fault_counts:
            if fault not in allowed:
                raise LedgerSchemaError(
                    f"{context}.fault_counts: execution ledger version "
                    f"{ledger_version} does not own provider fault {fault!r}; "
                    f"allowed: {list(allowed)}"
                )
        return totals


def recompute_totals(calls: Sequence[LedgerCall]) -> LedgerTotals:
    """The totals the given call rows state, derived from nothing else."""
    attempts = [attempt for call in calls for attempt in call.attempts]
    faults: dict[str, int] = {}
    for attempt in attempts:
        if attempt.fault is not None:
            faults[attempt.fault] = faults.get(attempt.fault, 0) + 1
    return LedgerTotals(
        provider_calls=len(calls),
        attempts=len(attempts),
        input_tokens=sum(a.input_tokens or 0 for a in attempts),
        output_tokens=sum(a.output_tokens or 0 for a in attempts),
        measured_cost_usd=_sum(
            [a.cost_measured_usd for a in attempts if a.cost_measured_usd is not None]
        ),
        forfeited_reservation_usd=_sum(
            [a.cost_forfeited_usd for a in attempts if a.cost_forfeited_usd is not None]
        ),
        reserved_usd=_sum(
            [
                a.cost_reservation_usd
                for a in attempts
                if a.cost_reservation_usd is not None
            ]
        ),
        fault_counts=dict(sorted(faults.items())),
    )


TERMINAL_FIELDS: tuple[str, ...] = (
    "kind",
    "ended_at",
    "exclusion_code",
    "exclusion_detail",
    "totals",
)


@dataclass(frozen=True)
class LedgerTerminal:
    """How the run ended, and what it added up to.

    There is no status, no terminal outcome, no evaluation and no final state
    here, and there is deliberately no field one could be put in. An excluded
    run has no episode artefact by contract; if its ledger could state a
    business result, the contract would be a naming convention rather than a
    rule.
    """

    kind: str
    ended_at: str
    totals: LedgerTotals
    exclusion_code: str | None = None
    exclusion_detail: str = ""

    def __post_init__(self) -> None:
        _member(self.kind, TERMINAL_KINDS, "terminal.kind")
        _instant(self.ended_at, "terminal.ended_at")
        if not isinstance(self.totals, LedgerTotals):
            raise LedgerSchemaError("terminal.totals: not a totals block")
        if self.kind == TERMINAL_SCORED and self.exclusion_code is not None:
            raise LedgerSchemaError(
                "terminal: a scored run states no exclusion code; a run cannot be "
                "both scored and stopped"
            )
        if self.kind == TERMINAL_EXCLUDED and self.exclusion_code is None:
            raise LedgerSchemaError(
                "terminal: an excluded run states the code it was excluded under"
            )
        if self.kind == TERMINAL_ABORTED and self.exclusion_code != ABORTED_CODE:
            raise LedgerSchemaError(
                f"terminal: an aborted run states {ABORTED_CODE!r}, got "
                f"{self.exclusion_code!r}"
            )
        _optional_member(self.exclusion_code, EXCLUSION_CODES, "terminal.exclusion_code")
        detail = TERMINAL_DETAILS[self.kind]
        if not self.exclusion_detail:
            object.__setattr__(self, "exclusion_detail", detail)
        if self.exclusion_detail != detail:
            raise LedgerSchemaError(
                "terminal.exclusion_detail: a terminal row states this build's own "
                "fixed detail for its kind and nothing else, so no text from "
                "anywhere else can reach durable evidence through it"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            **{name: getattr(self, name) for name in TERMINAL_FIELDS if name != "totals"},
            "totals": self.totals.as_dict(),
        }

    @classmethod
    def from_row(
        cls,
        payload: Mapping[str, Any],
        context: str,
        *,
        ledger_version: int = EXECUTION_LEDGER_VERSION,
    ) -> LedgerTerminal:
        _exact_fields(payload, TERMINAL_FIELDS, context, ledger_version=ledger_version)
        raw_totals = payload["totals"]
        if not isinstance(raw_totals, Mapping):
            raise LedgerSchemaError(f"{context}.totals: expected a JSON object")
        return cls(
            **{name: payload[name] for name in TERMINAL_FIELDS if name != "totals"},
            totals=LedgerTotals.from_row(
                raw_totals,
                f"{context}.totals",
                ledger_version=ledger_version,
            ),
        )


@dataclass(frozen=True)
class ExecutionLedgerAudit:
    """What a journal on disk actually proves, including when it proves little.

    ``status`` is the whole point. It is the terminal row's kind only when a
    terminal row verified; otherwise it is ``"incomplete"``, and
    :attr:`scored` is false. There is no path through this object on which a
    partial journal reports a scored run.

    :attr:`chain_tip_sha256` is the digest of the last row that verified.
    :attr:`ledger_digest_sha256` is that value **only for a journal that ends in
    a verified terminal row**, and ``None`` otherwise — see
    :func:`execution_ledger_digest`.
    """

    header: LedgerHeader
    calls: tuple[LedgerCall, ...]
    totals: LedgerTotals
    terminal: LedgerTerminal | None = None
    rows_verified: int = 0
    truncated_tail: bool = False
    chain_tip_sha256: str = GENESIS_DIGEST
    # Optional only so callers of this public result type that predate ledger
    # version reporting can still construct it. ``read_execution_ledger`` always
    # supplies the version independently verified from the journal rows.
    ledger_version: int | None = None

    @property
    def status(self) -> str:
        return STATUS_INCOMPLETE if self.terminal is None else self.terminal.kind

    @property
    def complete(self) -> bool:
        return self.terminal is not None

    @property
    def scored(self) -> bool:
        return self.status == TERMINAL_SCORED

    @property
    def ledger_digest_sha256(self) -> str | None:
        """The digest of this complete journal, or ``None`` if it is not one.

        The terminal row's own digest. Because every row states the digest of
        the row before it, that one value covers the header, every call row and
        the terminal row's derived totals transitively: change any byte of any
        row and this number changes, and a journal that ends anywhere other than
        a verified terminal row has no such value at all.

        ``None`` rather than the tip of a partial chain, deliberately. A binding
        that could name the tip of an incomplete journal would let an artefact
        cite a run that never finished as though it had.
        """
        return None if self.terminal is None else self.chain_tip_sha256

    def summary(self) -> dict[str, Any]:
        return {
            "ledger_version": self.ledger_version,
            "execution_run_id": self.header.execution_run_id,
            "operation_instance_id": self.header.operation_instance_id,
            "scenario_id": self.header.scenario_id,
            "model": self.header.provider.model,
            "settings_digest_sha256": self.header.provider.settings_digest_sha256,
            "pricing_digest_sha256": self.header.controls.pricing_digest_sha256,
            "ledger_digest_sha256": self.ledger_digest_sha256,
            "status": self.status,
            "complete": self.complete,
            "scored": self.scored,
            "truncated_tail": self.truncated_tail,
            "rows_verified": self.rows_verified,
            "exclusion_code": (
                None if self.terminal is None else self.terminal.exclusion_code
            ),
            "totals": self.totals.as_dict(),
        }


def execution_ledger_digest(path: Path | str) -> str:
    """The digest of one complete journal, read and verified from its own bytes.

    The independent form of :attr:`ExecutionLedgerAudit.ledger_digest_sha256`,
    for a caller holding a path and nothing else: it re-reads the file, re-walks
    the chain, and refuses a journal that does not end in a verified terminal
    row rather than returning the tip of a prefix.
    """
    audit = read_execution_ledger(path, require_complete=True)
    digest = audit.ledger_digest_sha256
    assert digest is not None  # require_complete guarantees a terminal row
    return digest


class IncompleteExecutionLedgerError(ExecutionLedgerError):
    """A journal was read strictly and does not end in a verified terminal row.

    Carries the prefix that *did* verify, because the prefix is the evidence: a
    run that died after forty calls made forty calls, and a reader that threw
    that away to report a clean failure would be discarding the only record of
    what was spent.
    """

    def __init__(self, audit: ExecutionLedgerAudit, detail: str) -> None:
        super().__init__(detail)
        self.audit = audit


# --------------------------------------------------------------------- the file


def _open_ledger_directory(directory: Path) -> int:
    """Open the ledger's directory by walking to it, and hold the descriptor.

    Every component is opened from the root with ``O_DIRECTORY | O_NOFOLLOW``
    against the descriptor of the component before it, so a symbolic link
    anywhere on the way is refused by the kernel rather than found by a test
    that ran earlier. What comes back is a descriptor for the directory that
    was *checked*, and the ledger is created against that descriptor.

    That is the whole of the fix for the check/use race. A path checked and
    then re-opened by name is a path anything that can rename a component gets
    to choose the second time; a descriptor names one inode for as long as it
    is held, whatever the name above it is made to mean afterwards.
    """
    absolute = Path(os.path.abspath(directory))
    try:
        fd = os.open(absolute.anchor or os.sep, os.O_RDONLY | os.O_DIRECTORY)
    except OSError as exc:
        raise LedgerPathError(
            "the execution ledger's filesystem anchor could not be opened; evidence "
            "is written only through a directory chain this build can verify"
        ) from exc
    for component in absolute.parts[1:]:
        try:
            below = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=fd,
            )
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.ELOOP, errno.EMLINK):
                raise LedgerPathError(
                    f"the execution ledger's location reaches through a symbolic link "
                    f"at {component!r}. A link is somebody else's decision about where "
                    "these bytes land, and evidence is written where the operator said "
                    "or nowhere"
                ) from exc
            raise LedgerPathError(
                f"the execution ledger's directory could not be opened at "
                f"{component!r} ({type(exc).__name__}): an "
                "operator names the location, and this build does not create one. Got "
                f"{directory}"
            ) from exc
        os.close(fd)
        fd = below
    try:
        _check_open_directory(fd, directory)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _check_open_directory(fd: int, directory: Path) -> None:
    """Owner and mode, read off the descriptor rather than off the path."""
    info = os.fstat(fd)
    if info.st_uid != os.geteuid():
        raise LedgerPathError(
            "the execution ledger's directory is owned by another user; this build "
            "writes evidence only where the running user owns the directory"
        )
    mode = stat.S_IMODE(info.st_mode)
    if mode != 0o700:
        raise LedgerPathError(
            f"the execution ledger's directory is mode {mode:04o} and must be 0700. "
            "A ledger names the runs a machine made and what they cost, and is "
            "readable by its owner alone. Got {0}".format(directory)
        )


def _create_ledger_file(dirfd: int, name: str) -> int:
    """Create the ledger inside the directory that descriptor names, or refuse.

    ``O_CREAT | O_EXCL | O_NOFOLLOW`` against the held descriptor: an existing
    file is never appended to, a symbolic link — dangling or not — is never
    followed, and two runs cannot interleave into one journal. There is no
    "does it exist" test before this, because that test *is* the race: the
    kernel's own exclusive create is the only check that cannot be overtaken.
    """
    try:
        return os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=dirfd,
        )
    except FileExistsError as exc:
        raise LedgerPathError(
            f"{name} already exists, or is a symbolic link. An execution ledger is "
            "written once: appending to a file this run did not create would merge "
            "two runs' evidence into one chain"
        ) from exc
    except OSError as exc:
        raise LedgerPathError(
            f"the execution ledger could not be created exclusively at {name} "
            f"({type(exc).__name__})"
        ) from exc


class ExecutionLedgerJournal:
    """The writer: one row at a time, flushed and fsynced before the next call.

    Opened exclusively — ``O_CREAT | O_EXCL | O_NOFOLLOW`` — so an existing file
    is never appended to, a symbolic link is never followed, and two runs cannot
    interleave into one journal. The file is created 0600 and the directory must
    already be 0700.

    The instance is usable as a context manager. Leaving the block closes the
    file; it does *not* invent a terminal row, because a run whose finaliser
    never ran is exactly the run whose journal must read as incomplete.
    """

    def __init__(self, fd: int, path: Path, header: LedgerHeader) -> None:
        self._fd = fd
        self._path = path
        self._header = header
        self._ledger_version = EXECUTION_LEDGER_VERSION
        self._calls: list[LedgerCall] = []
        self._row_index = 0
        self._previous = GENESIS_DIGEST
        self._terminal: LedgerTerminal | None = None
        self._closed = False
        self._poisoned: str | None = None
        self._append(ROW_HEADER, header.as_dict())

    @classmethod
    def open(cls, path: Path, header: LedgerHeader) -> ExecutionLedgerJournal:
        """Create the journal and write its header row, before any call is made."""
        if not isinstance(header, LedgerHeader):
            raise LedgerSchemaError(
                "an execution ledger opens on a validated header, not on "
                f"{type(header).__name__}"
            )
        path = Path(path)
        name = path.name
        if not name or name in (os.curdir, os.pardir) or os.sep in name:
            raise LedgerPathError(
                f"{path} does not name a file to write inside the directory the "
                "operator gave"
            )
        dirfd = _open_ledger_directory(path.parent)
        try:
            fd = _create_ledger_file(dirfd, name)
        except BaseException:
            os.close(dirfd)
            raise
        try:
            journal = cls(fd, path, header)
            os.fsync(dirfd)
            dirfd_to_close = dirfd
            dirfd = -1
            os.close(dirfd_to_close)
            return journal
        except BaseException:
            with suppress(OSError):
                os.close(fd)
            if dirfd >= 0:
                with suppress(OSError):
                    os.close(dirfd)
            raise

    @classmethod
    def open_at(
        cls, path: Path, header: LedgerHeader, *, dir_fd: int
    ) -> ExecutionLedgerJournal:
        """Create a journal in the exact already-authorised directory inode."""
        if not isinstance(header, LedgerHeader):
            raise LedgerSchemaError(
                "an execution ledger opens on a validated header, not on "
                f"{type(header).__name__}"
            )
        path = Path(path)
        name = os.fspath(path)
        if (
            not name
            or name in (os.curdir, os.pardir)
            or "/" in name
            or "\\" in name
            or "\x00" in name
        ):
            raise LedgerPathError(
                f"{path} does not name a file to write inside the directory the "
                "operator gave"
            )
        fd: int | None = None
        try:
            _check_open_directory(dir_fd, Path("."))
            fd = _create_ledger_file(dir_fd, name)
            os.fchmod(fd, 0o600)
            journal = cls(fd, path, header)
            os.fsync(dir_fd)
            return journal
        except BaseException:
            if fd is not None:
                with suppress(OSError):
                    os.close(fd)
            raise

    def __enter__(self) -> ExecutionLedgerJournal:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def header(self) -> LedgerHeader:
        return self._header

    @property
    def ledger_version(self) -> int:
        """The schema identity written into every row of this journal."""
        return self._ledger_version

    @property
    def calls_recorded(self) -> int:
        return len(self._calls)

    @property
    def calls(self) -> tuple[LedgerCall, ...]:
        """Every call row made durable so far, in the order it was written."""
        return tuple(self._calls)

    @property
    def terminal(self) -> LedgerTerminal | None:
        """The terminal row, once one is written. ``None`` until then."""
        return self._terminal

    @property
    def rows_written(self) -> int:
        return self._row_index

    @property
    def finalized(self) -> bool:
        return self._terminal is not None

    @property
    def poisoned(self) -> bool:
        """Whether a row failed to become durable and this writer is finished."""
        return self._poisoned is not None

    @property
    def chain_tip(self) -> str:
        """The digest of the last row this writer made durable."""
        return self._previous

    @property
    def ledger_digest_sha256(self) -> str | None:
        """This journal's digest, once it has a terminal row and not before.

        The same value :func:`read_execution_ledger` derives from the bytes, so
        a caller that binds to it here and a reader that verifies it later are
        stating one number rather than two that have to agree.
        """
        return None if self._terminal is None else self._previous

    def record_call(self, call: LedgerCall) -> None:
        """Append one call row, flushed and fsynced before this returns.

        Every check the reader will make of this row is made here first, and it
        is made *before* the append. A row this build would refuse to read is a
        row it will not write: evidence that has to be written and then
        disbelieved is evidence that was written.
        """
        if not isinstance(call, LedgerCall):
            raise LedgerSchemaError(
                f"an execution ledger records a validated call, not {type(call).__name__}"
            )
        assert_current_provider_ledger_compatibility()
        call.validate_for_ledger_version(self._ledger_version, f"call {call.call_index}")
        if self._terminal is not None:
            raise LedgerStateError(
                "this execution ledger has been finalised; a call recorded after the "
                "terminal row would be a call the run's own totals do not cover"
            )
        if call.call_index != len(self._calls):
            raise LedgerStateError(
                f"call indexes are contiguous from zero; this journal holds "
                f"{len(self._calls)} call(s) and was offered index {call.call_index}"
            )
        controls = self._header.controls
        check_call_admissible(
            call,
            controls=controls,
            previous=self._calls[-1] if self._calls else None,
        )
        check_call_amounts(call, controls=controls)
        self._append(ROW_CALL, call.as_dict())
        self._calls.append(call)

    def finalize(
        self,
        kind: str,
        *,
        ended_at: str,
        exclusion_code: str | None = None,
    ) -> LedgerTerminal:
        """Write the terminal row, with totals derived from the rows above it."""
        if self._terminal is not None:
            raise LedgerStateError(
                "this execution ledger has already been finalised; a second terminal "
                "row would let one run state two endings"
            )
        check_terminal_admissible(kind, exclusion_code, tuple(self._calls))
        terminal = LedgerTerminal(
            kind=kind,
            ended_at=ended_at,
            exclusion_code=exclusion_code,
            totals=recompute_totals(tuple(self._calls)),
        )
        self._append(ROW_TERMINAL, terminal.as_dict())
        self._terminal = terminal
        return terminal

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self._fd)

    def _append(self, kind: str, payload: Mapping[str, Any]) -> None:
        if self._poisoned is not None:
            raise LedgerStateError(self._poisoned)
        if self._closed:
            raise LedgerStateError("this execution ledger is closed")
        row: dict[str, Any] = {
            "ledger_version": self._ledger_version,
            "row_index": self._row_index,
            "row_kind": kind,
            "previous_row_digest_sha256": self._previous,
            kind: dict(payload),
        }
        assert_no_secret_material(row, f"row {self._row_index}")
        row["row_digest_sha256"] = row_digest(row)
        line = json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
        self._append_durably(line.encode("utf-8"), kind)
        self._previous = row["row_digest_sha256"]
        self._row_index += 1

    def _append_durably(self, payload: bytes, kind: str) -> None:
        """Write one row whole and fsync it, or leave the file as it was.

        The offset before the row is captured first, because it is the only
        thing that makes a failure recoverable: on any failure the file is
        truncated back to it, so what remains is the prefix that was already
        durable rather than a row that is half of one.

        The writer is then poisoned permanently. Not for this row — for the
        journal. A writer that failed to make a row durable cannot know what
        reached the device, so every later row would sit on an offset nothing
        verified, and a terminal row above one of those would read as a
        complete run.
        """
        offset = os.lseek(self._fd, 0, os.SEEK_CUR)
        try:
            self._write_all(payload)
            os.fsync(self._fd)
        except OSError as exc:
            rolled_back = self._roll_back(offset)
            self._poisoned = (
                f"this execution ledger's {kind} row could not be made durable "
                f"({type(exc).__name__}), so the writer is finished: nothing further "
                "is appended, finalised or abandoned through it. "
                + (
                    "The partial row was truncated away and what remains is the "
                    "verified prefix"
                    if rolled_back
                    else "The partial row could not be truncated away either, so the "
                    "file's tail is bytes this build could neither complete nor remove"
                )
            )
            raise LedgerWriteError(self._poisoned, rolled_back=rolled_back) from exc

    def _write_all(self, payload: bytes) -> None:
        """Every byte of one row, however many writes that takes.

        ``os.write`` may accept fewer bytes than it is given and may be
        interrupted before it accepts any. Treating either as a completed row
        is what leaves a partial line for the next row to be appended onto, so
        the loop continues until the row is whole and refuses a write that
        accepted nothing rather than spinning on it.
        """
        view = memoryview(payload)
        written = 0
        while written < len(view):
            try:
                count = os.write(self._fd, view[written:])
            except InterruptedError:
                continue
            if count <= 0:
                raise OSError(
                    errno.EIO,
                    "the execution ledger's file accepted none of the bytes offered",
                )
            written += count

    def _roll_back(self, offset: int) -> bool:
        """Truncate the failed row away, and say whether that worked."""
        try:
            os.ftruncate(self._fd, offset)
            os.lseek(self._fd, offset, os.SEEK_SET)
            os.fsync(self._fd)
        except OSError:
            return False
        return True


# ------------------------------------------------------------------- the reader


class _UndecodableLedgerRow(Exception):
    """Row bytes that do not form one complete UTF-8 JSON value."""


class _DuplicateLedgerKey(Exception):
    """Private parser control flow carrying no key or source input."""


_DUPLICATE_LEDGER_KEY = object()


def _parse_json_without_duplicate_retention(text: str) -> Any:
    """Decode JSON, returning an input-free marker for duplicate object keys."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: dict[str, Any] = {}
        for key, value in items:
            if key in seen:
                raise _DuplicateLedgerKey
            seen[key] = value
        return seen

    try:
        return json.loads(text, object_pairs_hook=pairs)
    except _DuplicateLedgerKey:
        return _DUPLICATE_LEDGER_KEY


def _incomplete_utf8_suffix(exc: UnicodeDecodeError) -> bool:
    """Return whether ``exc`` is solely a valid code point cut short at EOF."""
    if exc.end != len(exc.object) or exc.reason != "unexpected end of data":
        return False
    suffix = bytes(exc.object[exc.start : exc.end])
    if not suffix:
        return False
    lead = suffix[0]
    if 0xC2 <= lead <= 0xDF:
        width = 2
        second_low, second_high = 0x80, 0xBF
    elif 0xE0 <= lead <= 0xEF:
        width = 3
        second_low = 0xA0 if lead == 0xE0 else 0x80
        second_high = 0x9F if lead == 0xED else 0xBF
    elif 0xF0 <= lead <= 0xF4:
        width = 4
        second_low = 0x90 if lead == 0xF0 else 0x80
        second_high = 0x8F if lead == 0xF4 else 0xBF
    else:
        return False
    if len(suffix) >= width:
        return False
    if len(suffix) >= 2 and not second_low <= suffix[1] <= second_high:
        return False
    return all(0x80 <= byte <= 0xBF for byte in suffix[2:])


def _is_recoverable_json_object_text_prefix(text: str) -> bool:
    """Recognise strict prefixes of one JSON object with an iterative PDA.

    This accepts only text that can be extended (never edited) into one JSON
    object.  The explicit container stack makes runtime linear in input length
    and memory linear in nesting depth without Python recursion.
    """
    if not text or text[0] != "{":
        return text == ""

    # Phases are deliberately separate after a comma: a trailing comma cannot
    # borrow the initial container's permission to close while remaining a
    # purported prefix.
    stack: list[tuple[str, str]] = [("object", "key_or_end")]
    lexical: str | None = None
    string_role = ""
    unicode_digits = 0
    complete = False
    index = 1

    def set_phase(phase: str) -> None:
        kind, _ = stack[-1]
        stack[-1] = (kind, phase)

    def begin_value(character: str) -> bool:
        nonlocal lexical, string_role
        set_phase("comma_or_end")
        if character == '"':
            lexical = "string"
            string_role = "value"
        elif character == "{":
            stack.append(("object", "key_or_end"))
        elif character == "[":
            stack.append(("array", "value_or_end"))
        elif character in "tfn":
            lexical = {
                "t": "literal:true:1",
                "f": "literal:false:1",
                "n": "literal:null:1",
            }[character]
        elif character == "-" or (character.isdigit() and character.isascii()):
            lexical = "number"
        else:
            return False
        return True

    while index < len(text):
        character = text[index]

        if lexical == "string":
            if character == '"':
                lexical = None
                if string_role == "key":
                    set_phase("colon")
            elif character == "\\":
                lexical = "escape"
            elif ord(character) < 0x20:
                return False
            index += 1
            continue
        if lexical == "escape":
            if character == "u":
                lexical = "unicode"
                unicode_digits = 0
            elif character in '"\\/bfnrt':
                lexical = "string"
            else:
                return False
            index += 1
            continue
        if lexical == "unicode":
            if character not in "0123456789abcdefABCDEF":
                return False
            unicode_digits += 1
            if unicode_digits == 4:
                lexical = "string"
            index += 1
            continue
        if lexical is not None and lexical.startswith("literal:"):
            _, target, position_text = lexical.split(":")
            position = int(position_text)
            if character != target[position]:
                return False
            position += 1
            lexical = None if position == len(target) else f"literal:{target}:{position}"
            index += 1
            continue
        if lexical == "number":
            start = index - 1
            cursor = start
            if text[cursor] == "-":
                cursor += 1
                if cursor == len(text):
                    return True
            if cursor == len(text):
                return True
            if text[cursor] == "0":
                cursor += 1
                if (
                    cursor < len(text)
                    and text[cursor].isdigit()
                    and text[cursor].isascii()
                ):
                    return False
            elif "1" <= text[cursor] <= "9":
                cursor += 1
                while cursor < len(text) and "0" <= text[cursor] <= "9":
                    cursor += 1
            else:
                return False
            if cursor < len(text) and text[cursor] == ".":
                cursor += 1
                if cursor == len(text):
                    return True
                if not "0" <= text[cursor] <= "9":
                    return False
                cursor += 1
                while cursor < len(text) and "0" <= text[cursor] <= "9":
                    cursor += 1
            if cursor < len(text) and text[cursor] in "eE":
                cursor += 1
                if cursor == len(text):
                    return True
                if text[cursor] in "+-":
                    cursor += 1
                    if cursor == len(text):
                        return True
                if not "0" <= text[cursor] <= "9":
                    return False
                cursor += 1
                while cursor < len(text) and "0" <= text[cursor] <= "9":
                    cursor += 1
            if cursor < len(text) and text[cursor] not in " \t\r\n,]}":
                return False
            lexical = None
            index = cursor
            continue

        if complete:
            return False
        if character in " \t\r\n":
            index += 1
            continue

        kind, phase = stack[-1]
        if kind == "object":
            if phase in ("key_or_end", "key"):
                if character == "}" and phase == "key_or_end":
                    stack.pop()
                    complete = not stack
                elif character == '"':
                    lexical = "string"
                    string_role = "key"
                else:
                    return False
            elif phase == "colon":
                if character != ":":
                    return False
                set_phase("value")
            elif phase == "value":
                if not begin_value(character):
                    return False
            else:
                if character == ",":
                    set_phase("key")
                elif character == "}":
                    stack.pop()
                    complete = not stack
                else:
                    return False
        else:
            if phase in ("value_or_end", "value"):
                if character == "]" and phase == "value_or_end":
                    stack.pop()
                    complete = not stack
                elif not begin_value(character):
                    return False
            else:
                if character == ",":
                    set_phase("value")
                elif character == "]":
                    stack.pop()
                    complete = not stack
                else:
                    return False
        index += 1

    return not complete


def _is_recoverable_json_object_prefix(fragment: bytes) -> bool:
    """Return whether bytes can be extended into exactly one UTF-8 JSON object."""
    try:
        text = fragment.decode("utf-8")
    except UnicodeDecodeError as exc:
        if not _incomplete_utf8_suffix(exc):
            return False
        decoded = fragment[: exc.start].decode("utf-8")
        # Every valid non-ASCII scalar is legal in the same JSON positions.  A
        # representative scalar proves the cut code point is inside a string,
        # rather than merely following some independently extendable prefix.
        return _is_recoverable_json_object_text_prefix(decoded + "\u0080")
    return _is_recoverable_json_object_text_prefix(text)


def _decode(line: bytes, context: str) -> dict[str, Any]:
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError:
        raise _UndecodableLedgerRow(f"{context}: not one UTF-8 JSON object") from None
    try:
        decoded = _parse_json_without_duplicate_retention(text)
    except json.JSONDecodeError:
        raise _UndecodableLedgerRow(f"{context}: not one JSON object") from None
    if decoded is _DUPLICATE_LEDGER_KEY:
        # The public exception is raised only after every reference to parser input
        # has been dropped from this retained frame. The private sentinel was caught
        # inside the parser helper and therefore cannot survive as context or cause.
        del line, text
        raise LedgerSchemaError(f"{context}: duplicate object key") from None
    if not isinstance(decoded, dict):
        raise LedgerSchemaError(f"{context}: a ledger row is a JSON object")
    return decoded


def read_execution_ledger(
    path: Path | str, *, require_complete: bool = False, dir_fd: int | None = None
) -> ExecutionLedgerAudit:
    """Read a journal, verify its chain, and say exactly what it proves.

    A row is accepted only if it decodes, satisfies the schema exactly, sits at
    the next index, back-links to the row before it and re-digests to the digest
    it states. A *final* non-newline fragment is dropped only when an explicit
    UTF-8 and JSON parser-state check proves it is a strict prefix of one object;
    :attr:`ExecutionLedgerAudit.truncated_tail` then says so. Malformed bytes or
    grammar are a :class:`LedgerSchemaError` through both the path and descriptor
    APIs, and any non-empty bytes after a verified terminal row
    are refused because a journal ends once. Any *earlier* bad line is likewise
    a refusal, because a journal with a hole in the middle is not a prefix of
    anything.

    The totals in a terminal row are re-derived from the call rows above it and
    the row is refused if they disagree. ``require_complete`` turns the absence
    of a verified terminal row into :class:`IncompleteExecutionLedgerError`,
    which carries the verified prefix.
    """
    path = Path(path)
    if dir_fd is None:
        raw = path.read_bytes()
    else:
        name = os.fspath(path)
        if (
            not name
            or name in (os.curdir, os.pardir)
            or "/" in name
            or "\\" in name
            or "\x00" in name
        ):
            raise LedgerPathError(
                "the execution ledger could not be read safely at the requested basename"
            )
        descriptor: int | None = None
        try:
            descriptor = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dir_fd
            )
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
                raise OSError(errno.EPERM, "ledger identity or mode changed")
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
        except OSError as exc:
            raise LedgerPathError(
                f"the execution ledger could not be read safely at {path.name}"
            ) from exc
        finally:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
    lines = raw.split(b"\n")
    trailing_newline = raw.endswith(b"\n")
    if trailing_newline:
        lines = lines[:-1]

    header: LedgerHeader | None = None
    calls: list[LedgerCall] = []
    terminal: LedgerTerminal | None = None
    ledger_version: int | None = None
    previous = GENESIS_DIGEST
    verified = 0
    truncated = False

    for index, line in enumerate(lines):
        last = index == len(lines) - 1
        tolerate = last and not trailing_newline
        if terminal is not None and line:
            raise LedgerChainError(
                f"row {index}: follows the terminal row. A run ends once, and "
                "non-empty bytes after its ending are refused"
            )
        try:
            row = _decode(line, f"row {index}")
        except _UndecodableLedgerRow as exc:
            if tolerate and _is_recoverable_json_object_prefix(line):
                truncated = True
                break
            raise LedgerSchemaError(str(exc)) from None

        stated_version = row.get("ledger_version")
        if (
            type(stated_version) is not int
            or stated_version not in SUPPORTED_EXECUTION_LEDGER_VERSIONS
        ):
            raise LedgerSchemaError(
                f"row {index}: this build reads execution ledger versions "
                f"{list(SUPPORTED_EXECUTION_LEDGER_VERSIONS)}, and the row states "
                f"{stated_version!r}"
            )
        if ledger_version is None:
            ledger_version = stated_version
        elif stated_version != ledger_version:
            raise LedgerSchemaError(
                f"row {index}: states execution ledger version {stated_version}, "
                f"but the journal began as version {ledger_version}; one chain "
                "cannot change schema identity between rows"
            )
        _exact_fields(
            row,
            (
                "ledger_version",
                "row_index",
                "row_kind",
                "previous_row_digest_sha256",
                "row_digest_sha256",
                str(row.get("row_kind")),
            ),
            f"row {index}",
            ledger_version=stated_version,
        )
        kind = _member(row["row_kind"], ROW_KINDS, f"row {index}.row_kind")
        assert_no_secret_material(row, f"row {index}")
        if row["row_index"] != index:
            raise LedgerChainError(
                f"row {index}: states index {row['row_index']!r}. Rows are "
                "contiguous from zero, so a gap, a repeat or a swap is refused "
                "rather than read past"
            )
        if row["previous_row_digest_sha256"] != previous:
            raise LedgerChainError(
                f"row {index}: does not back-link to the row before it. The chain "
                "is what makes an edited, deleted or re-ordered row detectable"
            )
        if row_digest(row) != row["row_digest_sha256"]:
            raise LedgerChainError(
                f"row {index}: does not hash to the digest it states, so it is "
                "not the row this build wrote"
            )

        if kind == ROW_HEADER:
            if index != 0 or header is not None:
                raise LedgerChainError(f"row {index}: a journal has one header, first")
            header = LedgerHeader.from_row(row[ROW_HEADER], f"row {index}.header")
        elif kind == ROW_CALL:
            if header is None:
                raise LedgerChainError(f"row {index}: a call row precedes the header")
            assert ledger_version is not None
            call = LedgerCall.from_row(
                row[ROW_CALL],
                f"row {index}.call",
                ledger_version=ledger_version,
            )
            if call.call_index != len(calls):
                raise LedgerChainError(
                    f"row {index}: states call index {call.call_index} where "
                    f"{len(calls)} is the next one. Call indexes are contiguous from "
                    "zero, so a re-indexed row cannot be slipped in"
                )
            check_call_admissible(
                call,
                controls=header.controls,
                previous=calls[-1] if calls else None,
            )
            check_call_amounts(call, controls=header.controls)
            calls.append(call)
        else:
            if header is None:
                raise LedgerChainError(f"row {index}: a terminal row precedes the header")
            stated = LedgerTerminal.from_row(
                row[ROW_TERMINAL],
                f"row {index}.terminal",
                ledger_version=ledger_version,
            )
            check_terminal_admissible(stated.kind, stated.exclusion_code, tuple(calls))
            derived = recompute_totals(tuple(calls))
            if stated.totals.as_dict() != derived.as_dict():
                raise LedgerChainError(
                    f"row {index}: the terminal totals do not follow from the call "
                    "rows above them. A total that cannot be rebuilt from its own "
                    "parts is a claim, and this one is refused rather than reported"
                )
            terminal = stated
        previous = row["row_digest_sha256"]
        verified += 1

    if header is None:
        raise LedgerChainError(
            f"{path.name}: no verified header row, so there is no run to report. An "
            "empty or wholly unreadable journal proves nothing, including nothing "
            "about what a run spent"
        )
    assert ledger_version is not None  # a verified header row supplied it
    audit = ExecutionLedgerAudit(
        header=header,
        calls=tuple(calls),
        totals=recompute_totals(tuple(calls)),
        terminal=terminal,
        rows_verified=verified,
        truncated_tail=truncated,
        # ``previous`` is the digest of the last row that verified, which after
        # a terminal row is the digest of the whole journal.
        chain_tip_sha256=previous,
        ledger_version=ledger_version,
    )
    if require_complete and terminal is None:
        raise IncompleteExecutionLedgerError(
            audit,
            f"{path.name}: {verified} row(s) verify and none of them is a terminal "
            f"row, so this run is {STATUS_INCOMPLETE!r}. The verified prefix is "
            "carried on this error: the calls it records were made and are not "
            "discarded by the run failing to finish",
        )
    return audit


def verify_execution_ledger(path: Path | str) -> ExecutionLedgerAudit:
    """Read and verify a journal, returning what it proves. An alias with intent."""
    return read_execution_ledger(path)


__all__ = [
    "ABORTED_CODE",
    "ATTEMPT_FIELDS",
    "CALL_FIELDS",
    "CONTROLS_FIELDS",
    "CONTROL_EXCLUSIONS",
    "COST_CAP_POLICIES",
    "COST_CAP_POLICY_REFUSE",
    "DECISION_FIELDS",
    "EXCLUSION_BUDGET",
    "EXCLUSION_CODES",
    "EXCLUSION_DEADLINE",
    "EXECUTION_LEDGER_VERSION",
    "EXECUTION_LEDGER_VERSION_V2",
    "EXECUTION_RUN_ID_PREFIX",
    "GENESIS_DIGEST",
    "HEADER_FIELDS",
    "INTENDED_ARTIFACT_VERSION",
    "LEDGER_PRICING_PRECISION",
    "MAX_LEDGER_TEXT_CHARACTERS",
    "PRICING_RATE_FIELDS",
    "PROVIDER_FIELDS",
    "ROW_CALL",
    "ROW_HEADER",
    "ROW_KINDS",
    "ROW_TERMINAL",
    "STATUS_INCOMPLETE",
    "STOP_CLASSIFICATIONS",
    "STOP_COMPLETED",
    "STOP_TRUNCATED",
    "SUPPORTED_EXECUTION_LEDGER_VERSIONS",
    "TERMINAL_ABORTED",
    "TERMINAL_DETAILS",
    "TERMINAL_EXCLUDED",
    "TERMINAL_FIELDS",
    "TERMINAL_KINDS",
    "TERMINAL_SCORED",
    "TOKENS_PER_RATE_UNIT",
    "TOTALS_FIELDS",
    "ExecutionLedgerAudit",
    "ExecutionLedgerError",
    "ExecutionLedgerJournal",
    "IncompleteExecutionLedgerError",
    "LedgerAttempt",
    "LedgerCall",
    "LedgerChainError",
    "LedgerControls",
    "LedgerDecision",
    "LedgerHeader",
    "LedgerPathError",
    "LedgerProviderIdentity",
    "LedgerSchemaError",
    "LedgerSecretError",
    "LedgerStateError",
    "LedgerTerminal",
    "LedgerTotals",
    "LedgerWriteError",
    "assert_no_secret_material",
    "check_call_admissible",
    "check_call_amounts",
    "check_execution_run_id",
    "check_terminal_admissible",
    "derive_measured_usd",
    "derive_reservation_usd",
    "execution_ledger_digest",
    "execution_run_id_problem",
    "ledger_digest",
    "new_execution_run_id",
    "read_execution_ledger",
    "recompute_totals",
    "row_digest",
    "verify_execution_ledger",
]
