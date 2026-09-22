"""The bundle: an episode artefact, its execution ledger, and whether they agree.

An artefact-7 record carries a *binding* — a run identity, a ledger digest, a
provider identity, totals and one call index per decision. Its own reader can
establish that the binding is **coherent**: the right shape, the right
cardinality, the exact contiguous mapping, and no contradiction with the
execution record and the decision tape beside it. What that reader cannot do,
and deliberately does not attempt, is say whether the journal holds what the
binding says it holds — the journal is a different file, and
:func:`~operatebench.artifact.replay_artifact` is required never to open it.

This module is the one place that does open it, and it is a separate module for
that reason. Nothing in :mod:`operatebench.artifact` imports this, so "replay
reaches the sidecar" is not a mistake somebody can make by editing a line: the
dependency does not exist in that direction.

**Every refusal is a named class.** A bundle whose run identities differ is a
different failure from one whose totals differ, and an operator reading the
exception has to be able to tell which happened without re-deriving it.

**The digest is checked last, and that is deliberate.** The ledger digest is the
coarse answer — it says *something* differs and never *what*. Running the
identity, totals and per-call checks first means a bundle that disagrees in a
way this build can explain gets explained, and the digest check remains as the
backstop for a difference nothing above it names. A caller who only wants the
coarse answer has
:func:`~operatebench.execution_ledger.execution_ledger_digest` and can compare
it themselves.

**What a passing audit does not establish.** It establishes that two local files
describe one run: the same execution identity, the same provider and settings,
the same rate table, the same totals, and, call by call, the same request the
artefact says was hashed and the same decision the artefact says came back. It
establishes nothing about the provider. A party who can run this build can
produce both files from the same observations, and no digest over a
self-describing local file distinguishes a provider that answered from one that
was never asked. Closing that needs evidence the provider signs, which neither
of these documents holds.

Nothing here opens a socket, reads a credential or imports a provider SDK, and
nothing here writes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from operatebench.artifact import (
    ARTIFACT_VERSION,
    content_digest,
    read_artifact,
    validate_artifact,
)
from operatebench.core.errors import ArtifactError, OperateBenchError
from operatebench.execution_ledger import (
    TERMINAL_SCORED,
    ExecutionLedgerAudit,
    ExecutionLedgerError,
    LedgerCall,
    read_execution_ledger,
)

#: Said wherever a passing audit is reported. A bundle audit is a statement
#: about two files agreeing, and it is worth saying plainly rather than leaving
#: to be discovered.
BUNDLE_NOTE = (
    "This audit establishes that an episode artefact and an execution ledger "
    "describe one run: the same execution identity, provider, settings, rate "
    "table and totals, and the same request and decision on every call. It is "
    "not provider attestation. Both documents are produced locally, and a "
    "content digest answers whether two records carry the same content, not who "
    "could have written them."
)


class ExecutionBundleError(OperateBenchError):
    """An artefact and a ledger were not shown to be one run. Every refusal is one."""


class BundleContractError(ExecutionBundleError):
    """The artefact is not one this build can audit a bundle for.

    A record below contract 7, one whose binding is absent because the run
    reached no provider, or one that does not validate at all. In each case
    there is no bundle to audit rather than a bundle that failed.
    """


class BundleLedgerStateError(ExecutionBundleError):
    """The journal is not a complete, scored one, or does not read at all.

    Also the wrapper for every
    :class:`~operatebench.execution_ledger.ExecutionLedgerError` a read raises.
    A caller of a bundle surface asked one question and gets one
    family of answers; letting a ledger error escape here would make "the
    sidecar is broken" and "the sidecar disagrees" two exception hierarchies a
    caller has to know about to handle one failure.
    """


class BundleIdentityMismatchError(ExecutionBundleError):
    """The two documents do not name the same run, provider, model or settings."""


class BundleLedgerVersionMismatchError(BundleIdentityMismatchError):
    """The binding and independently verified sidecar name different contracts."""


class BundleTotalsMismatchError(ExecutionBundleError):
    """The binding's totals are not the totals the journal's own rows derive."""


class BundleCallBindingError(ExecutionBundleError):
    """A call the binding indexes is not the call the artefact recorded there."""


class BundleDigestMismatchError(ExecutionBundleError):
    """The journal's digest is not the digest the binding states.

    The backstop. Reached only when nothing above it explained the difference,
    which means the two files differ somewhere this build has no more specific
    name for.
    """


@dataclass(frozen=True)
class ExecutionBundleAudit:
    """What one artefact and one ledger were shown to agree about."""

    execution_run_id: str
    execution_ledger_digest_sha256: str
    provider: str
    api: str
    model: str
    settings_digest_sha256: str
    pricing_digest_sha256: str
    provider_calls: int
    attempts: int
    measured_cost_usd: str
    forfeited_reservation_usd: str
    input_tokens: int
    output_tokens: int
    decision_call_index: tuple[int, ...]
    ledger_path: str
    # Optional only so callers of this public result type that predate version
    # reporting can still construct it. Production audits always populate it
    # from the independently verified ``ExecutionLedgerAudit``.
    execution_ledger_version: int | None = None

    @property
    def ok(self) -> bool:
        """Always true on an instance that exists.

        There is no failing :class:`ExecutionBundleAudit`: a bundle that does
        not bind raises, and nothing here returns a report with ``ok=False``
        that a caller could forget to read. The property exists so a summary
        reads the same way whether it was produced by this build or by
        something checking it.
        """
        return True

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "provider_attested": False,
            "execution_run_id": self.execution_run_id,
            "execution_ledger_version": self.execution_ledger_version,
            "execution_ledger_digest_sha256": self.execution_ledger_digest_sha256,
            "provider": self.provider,
            "api": self.api,
            "model": self.model,
            "settings_digest_sha256": self.settings_digest_sha256,
            "pricing_digest_sha256": self.pricing_digest_sha256,
            "provider_calls": self.provider_calls,
            "attempts": self.attempts,
            "measured_cost_usd": self.measured_cost_usd,
            "forfeited_reservation_usd": self.forfeited_reservation_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "decision_call_index": list(self.decision_call_index),
            "ledger": self.ledger_path,
            "note": BUNDLE_NOTE,
        }


def _read_artifact_for_bundle(artifact: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return validate_artifact(artifact, "bundle artefact")
    except ArtifactError as exc:
        raise BundleContractError(
            f"this artefact does not read as a run artefact, so there is no bundle "
            f"to audit: {exc}"
        ) from exc


def _read_ledger_for_bundle(
    ledger_path: Path | str, *, dir_fd: int | None = None
) -> ExecutionLedgerAudit:
    try:
        return read_execution_ledger(ledger_path, require_complete=True, dir_fd=dir_fd)
    except ExecutionLedgerError as exc:
        raise BundleLedgerStateError(
            f"{Path(ledger_path).name} does not read as a complete execution "
            f"ledger, so it witnesses nothing this artefact can be bound to: {exc}"
        ) from exc
    except OSError as exc:
        raise BundleLedgerStateError(
            f"the execution ledger could not be read at {Path(ledger_path)}: {exc}"
        ) from exc


def _require(condition: bool, error: type[ExecutionBundleError], message: str) -> None:
    if not condition:
        raise error(message)


def _check_identity(
    body: Mapping[str, Any], binding: Mapping[str, Any], ledger: ExecutionLedgerAudit
) -> None:
    """The two documents name one run, asked through one thing, under one table."""
    _require(
        binding["execution_ledger_version"] == ledger.ledger_version,
        BundleLedgerVersionMismatchError,
        "the artefact's execution ledger contract does not match the verified "
        "sidecar contract",
    )
    _require(
        binding["execution_run_id"] == ledger.header.execution_run_id,
        BundleIdentityMismatchError,
        f"the artefact binds execution run {binding['execution_run_id']!r} and this "
        f"ledger records {ledger.header.execution_run_id!r}. Two valid documents "
        "about two runs are still two runs, and pairing them would attribute one "
        "run's spend to another's decisions",
    )
    identity = ledger.header.provider
    artifact_max_output = body["agent_execution"]["max_output_tokens"]
    _require(
        artifact_max_output == ledger.header.controls.max_output_tokens,
        BundleIdentityMismatchError,
        "the artefact records agent_execution.max_output_tokens="
        f"{artifact_max_output!r} and the ledger authorises controls.max_output_tokens="
        f"{ledger.header.controls.max_output_tokens!r}. One run cannot execute and "
        "account for requests under different output ceilings",
    )
    for name, recorded in (
        ("provider", identity.provider),
        ("api", identity.api),
        ("model", identity.model),
        ("settings_digest_sha256", identity.settings_digest_sha256),
        ("pricing_digest_sha256", ledger.header.controls.pricing_digest_sha256),
    ):
        _require(
            binding[name] == recorded,
            BundleIdentityMismatchError,
            f"the artefact binds {name}={binding[name]!r} and the ledger records "
            f"{recorded!r}. An answer is what it is because of who produced it, "
            "under which settings and at which rates, so a binding that disagrees "
            "about any of the three is not a binding to this journal",
        )


def _check_totals(binding: Mapping[str, Any], ledger: ExecutionLedgerAudit) -> None:
    """The binding's numbers are the ones the journal's own rows add up to.

    Compared against :attr:`ExecutionLedgerAudit.totals`, which the reader
    re-derived from the call rows rather than read out of the terminal row. A
    binding checked against a *stated* total would be checking two claims
    against each other.
    """
    totals = ledger.totals
    for name, recorded in (
        ("provider_calls", totals.provider_calls),
        ("attempts", totals.attempts),
        ("input_tokens", totals.input_tokens),
        ("output_tokens", totals.output_tokens),
        ("measured_cost_usd", totals.measured_cost_usd),
        ("forfeited_reservation_usd", totals.forfeited_reservation_usd),
    ):
        _require(
            binding[name] == recorded,
            BundleTotalsMismatchError,
            f"the artefact binds {name}={binding[name]!r} and this ledger's own "
            f"rows come to {recorded!r}. A total that does not follow from the rows "
            "beside it is a claim rather than a measurement",
        )


def _check_calls(
    body: Mapping[str, Any],
    binding: Mapping[str, Any],
    ledger: ExecutionLedgerAudit,
) -> tuple[int, ...]:
    """Call by call: the same request, and the same decision off the same answer."""
    attempts: Sequence[Mapping[str, Any]] = list(body["agent_execution"]["attempts"])
    decisions: Sequence[Mapping[str, Any]] = list(body["decisions"])
    indexes = tuple(int(index) for index in binding["decision_call_index"])
    _require(
        len(indexes) == len(decisions) == len(attempts),
        BundleCallBindingError,
        f"this artefact records {len(decisions)} decision(s) against "
        f"{len(attempts)} attempt(s) and a mapping of {len(indexes)} call(s)",
    )
    _require(
        len(ledger.calls) == len(indexes),
        BundleCallBindingError,
        f"this ledger records {len(ledger.calls)} call(s) and the artefact maps "
        f"{len(indexes)} decision(s) onto them; a decision with no call, or a call "
        "with no decision, is a tape and a journal describing different runs",
    )
    for position, index in enumerate(indexes):
        _require(
            0 <= index < len(ledger.calls),
            BundleCallBindingError,
            f"decision {position} is mapped onto call {index}, which this ledger "
            "does not record",
        )
        _check_one_call(
            ledger.calls[index], attempts[position], decisions[position], position
        )
    return indexes


def _check_one_call(
    call: LedgerCall,
    attempt: Mapping[str, Any],
    decision: Mapping[str, Any],
    position: int,
) -> None:
    where = f"decision {position} / ledger call {call.call_index}"
    _require(
        call.request_digest_sha256 == attempt["request_digest_sha256"],
        BundleCallBindingError,
        f"{where}: the ledger row hashed request "
        f"{call.request_digest_sha256!r} and the artefact's attempt at the same "
        f"position hashed {attempt['request_digest_sha256']!r}. The request "
        "identity is what ties a recorded answer to the prompt that produced it, "
        "so a row and an attempt that disagree about it describe two requests",
    )
    _require(
        (call.invocation_index, call.turn_index)
        == (attempt["invocation_index"], attempt["turn_index"]),
        BundleCallBindingError,
        f"{where}: the ledger row is invocation {call.invocation_index} turn "
        f"{call.turn_index} and the artefact's attempt is invocation "
        f"{attempt['invocation_index']} turn {attempt['turn_index']}",
    )
    _require(
        call.decision is not None,
        BundleCallBindingError,
        f"{where}: the ledger row records no decision, so nothing on it is the "
        "decision the artefact says came from it",
    )
    assert call.decision is not None
    outcome = decision["outcome"]
    _require(
        call.decision.decision_digest_sha256 == content_digest(dict(outcome)),
        BundleCallBindingError,
        f"{where}: the ledger row states a decision digest that is not the digest "
        "of the decision the artefact's tape records at the same position; the "
        "journal and the tape describe two different answers",
    )
    _require(
        call.decision.kind == str(outcome.get("kind")),
        BundleCallBindingError,
        f"{where}: the ledger row records a {call.decision.kind!r} decision and "
        f"the artefact's tape records {outcome.get('kind')!r}",
    )
    answering = call.attempts[-1]
    _require(
        answering.response_normalized_digest_sha256 is not None,
        BundleCallBindingError,
        f"{where}: the ledger row states no normalized digest for the answer its "
        "decision is claimed to have come from, so nothing binds the two",
    )


def audit_execution_bundle(
    artifact: Mapping[str, Any],
    ledger_path: Path | str,
    *,
    ledger_dir_fd: int | None = None,
) -> ExecutionBundleAudit:
    """Prove one artefact and one ledger are one run, or refuse by name.

    Both documents are validated in full first — an artefact that does not read
    and a journal that does not verify are refusals of their own rather than
    inputs to a comparison. What follows is, in order: the run identity, the
    independently verified ledger contract, provider identity and the two
    digests that pin what the answers were produced under; the totals, against
    the ones re-derived from the rows; each call, against the attempt and the
    decision the artefact records at the same position; and last, the journal's
    digest against the one the binding states.

    See the module docstring for what a pass does and does not establish.
    """
    body = _read_artifact_for_bundle(artifact)
    version = int(body["artifact_version"])
    _require(
        version >= ARTIFACT_VERSION,
        BundleContractError,
        f"this artefact was written under contract {version}, which carries no "
        f"provider execution binding; a bundle is audited at contract "
        f"{ARTIFACT_VERSION} and above",
    )
    binding = body["provider_execution"]
    _require(
        binding is not None,
        BundleContractError,
        "this artefact carries no provider execution binding, so it reached no "
        "provider and there is no bundle to audit. A ledger offered beside one is "
        "evidence about some other run",
    )
    assert binding is not None

    ledger = _read_ledger_for_bundle(ledger_path, dir_fd=ledger_dir_fd)
    _require(
        ledger.status == TERMINAL_SCORED,
        BundleLedgerStateError,
        f"{Path(ledger_path).name} ends {ledger.status!r} and a scored episode is "
        f"bound to a {TERMINAL_SCORED!r} journal. A run that stopped at the "
        "provider boundary has this journal and no episode artefact, so an "
        "artefact offered beside one describes a different run",
    )

    _check_identity(body, binding, ledger)
    _check_totals(binding, ledger)
    indexes = _check_calls(body, binding, ledger)

    digest = ledger.ledger_digest_sha256
    _require(
        digest == binding["execution_ledger_digest_sha256"],
        BundleDigestMismatchError,
        f"the artefact binds execution ledger digest "
        f"{binding['execution_ledger_digest_sha256']!r} and this journal's chain "
        f"comes to {digest!r}. Nothing above this named a difference, so the two "
        "files differ somewhere this build has no more specific refusal for",
    )
    assert digest is not None

    return ExecutionBundleAudit(
        execution_run_id=str(binding["execution_run_id"]),
        execution_ledger_version=ledger.ledger_version,
        execution_ledger_digest_sha256=digest,
        provider=str(binding["provider"]),
        api=str(binding["api"]),
        model=str(binding["model"]),
        settings_digest_sha256=str(binding["settings_digest_sha256"]),
        pricing_digest_sha256=str(binding["pricing_digest_sha256"]),
        provider_calls=int(binding["provider_calls"]),
        attempts=int(binding["attempts"]),
        measured_cost_usd=str(binding["measured_cost_usd"]),
        forfeited_reservation_usd=str(binding["forfeited_reservation_usd"]),
        input_tokens=int(binding["input_tokens"]),
        output_tokens=int(binding["output_tokens"]),
        decision_call_index=indexes,
        ledger_path=Path(ledger_path).name,
    )


def audit_execution_bundle_files(
    artifact_path: Path | str,
    ledger_path: Path | str,
    *,
    artifact_dir_fd: int | None = None,
    ledger_dir_fd: int | None = None,
) -> ExecutionBundleAudit:
    """Audit a bundle from two paths. The operator-facing form."""
    try:
        artifact = read_artifact(artifact_path, dir_fd=artifact_dir_fd)
    except ArtifactError as exc:
        raise BundleContractError(
            f"{Path(artifact_path)} does not read as a run artefact: {exc}"
        ) from exc
    return audit_execution_bundle(artifact, ledger_path, ledger_dir_fd=ledger_dir_fd)


__all__ = [
    "BUNDLE_NOTE",
    "BundleCallBindingError",
    "BundleContractError",
    "BundleDigestMismatchError",
    "BundleIdentityMismatchError",
    "BundleLedgerStateError",
    "BundleLedgerVersionMismatchError",
    "BundleTotalsMismatchError",
    "ExecutionBundleAudit",
    "ExecutionBundleError",
    "audit_execution_bundle",
    "audit_execution_bundle_files",
]
