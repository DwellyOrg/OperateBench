"""Run identity: the immutable manifest written before the first episode.

A run is only reproducible if what it *was* is fixed before anything happens.
The manifest states every input that changes execution or grading — which suite
at which pinned digest, which scaffold at which pinned digest, which adapter at
which version and settings, how many trials, which limits, which episodes in
which order — and that is the run's *configuration*.

Three separations are load-bearing:

* **Configuration identity is not execution identity.** ``configuration_id`` is
  the hash of everything that changes execution or grading and of nothing else,
  so the same plan is the same configuration however often and whenever it is
  started. That is what a resume can check. It is *not* a name for one run: two
  independent invocations of one plan are two executions, and recording them
  under a single id left their evidence indistinguishable and their provider
  dates unrecoverable. So every new run directory also mints an
  ``execution_id`` — a version-4 UUID from the platform's cryptographic
  generator — and stamps ``created_at_utc``, and every row and report carries
  both.
* **Creation time is execution evidence, not a configuration input, and not an
  immutability claim.** It is excluded from ``configuration_id`` because a run
  started at noon and the same run started at midnight are the same experiment.
  It is bound into ``execution_digest`` because an execution record whose
  timestamp could be edited independently of its id is not a record of when
  anything happened. What that digest proves is that the manifest's execution
  fields have not drifted *from each other*; it is a self-contained unkeyed
  hash in a writable directory, so it proves nothing about external
  immutability and this build does not claim otherwise.
* **Secrets never arrive.** Adapter settings are screened recursively and a
  secret-bearing key is *refused*, not redacted. Redaction relies on the writer
  recognising every shape a credential can take; refusal only relies on the key
  name, and fails closed when it is unsure.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from boundarybench.adapter import (
    ADAPTER_CONTRACT_VERSION,
    IN_PROCESS_PROVIDERS,
    AdapterIdentity,
)
from boundarybench.budget import BudgetError, CostControls
from boundarybench.evaluator import EVALUATOR_CONTRACT_VERSION
from boundarybench.freezing import deep_freeze, to_json
from boundarybench.jsonsafe import (
    JsonSafetyError,
    NestingDepthError,
    canonical_json_text,
    check_text,
    ensure_json_safe,
    ensure_raw_json_depth,
)
from boundarybench.scaffold import Scaffold
from boundarybench.suite import SuiteReport
from boundarybench.version import __version__

#: The run manifest file format this build reads and writes.
#:
#: ``2`` added ``track`` and made ``status`` and ``scope`` follow from it.
#:
#: ``3`` split the single ``run_id`` into ``configuration_id`` (deterministic,
#: wall-clock free) and ``execution_id``/``created_at_utc`` (unique per new run
#: directory, bound together by ``execution_digest``). Version-2 manifests name
#: one id for two different things and are refused rather than reinterpreted.
#:
#: ``4`` added ``cost_controls``: the cost cap, the episode ceiling and the
#: pricing policy a run's spend is measured against. They are inside
#: configuration identity because they change what a run does — a run that stops
#: at USD 100 is not the run that stops at USD 10 — so a manifest without the
#: block describes a run with no enforced limits and is refused rather than read
#: as an unlimited one.
#:
#: ``5`` made the stored pricing block self-describing and exact. It now states
#: the provider and the token unit its rates are quoted in as well as the rates
#: themselves, and it is compared against the shipped policy key-exact,
#: type-exact and value-exact when it is read. A version-4 manifest states two
#: fewer fields than the block this build validates, so it is refused by version
#: rather than read with the missing halves assumed.
RUN_MANIFEST_SCHEMA_VERSION = 5

#: The runner contract this build implements: what an episode loop does, how
#: limits are counted and how outcomes are classified.
#:
#: ``0.2.0`` moved four things a row's meaning depends on: a row records which
#: execution produced it and when the episode ran, it carries structured
#: provider-attempt evidence, it states whether it is a completed semantic
#: sample, and a run-wide terminal configuration failure stops the run instead
#: of producing twelve identical rows.
#:
#: ``0.3.0`` moved two more. An episode's ``completed_at_utc`` is derived from
#: its recorded start plus the monotonic duration it measured, not read from the
#: wall clock a second time, so a clock correction can no longer invert a row's
#: window. And the run-terminal set narrowed to exactly the two failures proven
#: to hold for every remaining episode — a refused credential and a provider
#: that cannot find the pinned model — so a rejected request now fails its own
#: episode and leaves the plan running. A run under each produces a different
#: number of rows from the same provider behaviour.
#: ``0.4.0`` moved three more. A run carries hard cost and episode limits, it
#: refuses to dispatch a provider request its remaining authorised budget cannot
#: cover and records that refusal as a run-terminal row, and every row states the
#: conservative exposure of the attempts it made that returned no usage — which
#: is what a resume recovers so it cannot forget what the run before it spent.
#: ``0.5.0`` moved three more, all of them about money a run may spend. The
#: guard a run executes under is bound to the ``cost_controls`` its own manifest
#: records — and to the adapter that will ask it — before any request is
#: dispatched, so a run can no longer spend under limits its durable record does
#: not state. A response measuring more than the reservation that authorised it
#: ends the run as a recorded breach instead of being committed as an ordinary
#: success. And a resumed ledger has its measured cost re-derived from its token
#: counts and its exposure re-derived from its attempts' reservations, rather
#: than being read back from totals the same file states. A run under ``0.4.0``
#: can complete where a ``0.5.0`` run stops, and can resume from a ledger a
#: ``0.5.0`` run refuses.
#: ``0.6.0`` moved one more, and it is about the same money. A resumed ledger's
#: per-attempt reservations are now *rebuilt* from the request each turn sent —
#: a projection of this manifest, the scaffold it pins, the compiled variant and
#: the replayed trajectory — instead of being accepted anywhere inside the range
#: a request of this run could have reserved. Exposure is then summed from those
#: rebuilt amounts rather than from the field being validated, so a row whose
#: reservations and whose total were edited together no longer agrees with
#: itself. A ``0.5.0`` run resumes from ledgers a ``0.6.0`` run refuses, and a
#: ``0.6.0`` run refuses to start at all against a request mapping, output
#: ceiling or provider whose requests this build cannot reproduce.
#: ``0.7.0`` moved where one whole class of failure is decided. A turn that names
#: a fact key outside the set the case offered is now refused during response
#: validation, as an argument that does not satisfy the schema the model was
#: shown, instead of being dispatched into the environment and refused there. A
#: ``0.6.0`` run recorded those episodes as the model choosing an inadmissible
#: action; a ``0.7.0`` run records them as the model failing the protocol, which
#: is the honest reading once the admissible keys are part of the interface —
#: see :data:`~boundarybench.scaffold.FACT_AFFORDANCE_CONTRACT`. The same
#: provider behaviour therefore produces a different outcome class, a different
#: error class and a different sample status under each.
#: ``0.8.0`` moved what an acquisition turn can *mean*. Every request for
#: information now resolves through the Cube's authored query-resolution
#: registry, which answers with one of four closed outcomes; a turn may therefore
#: legitimately obtain a handle that carries no value, and an episode that asked
#: three questions and was told the answers are not recorded is a completed
#: semantic sample rather than a failure. A ``0.7.0`` run could only ever be
#: given facts, so its rows cannot describe the same protocol and a resume across
#: the boundary is refused.
#: ``0.9.0`` moved what an acquisition turn *records*. A request for information
#: the episode already holds returns the same handle and reveals nothing, as it
#: always did, and is now recorded as a step of its own rather than being
#: dropped. Under ``0.8.0`` that turn was spent and never filed, so an episode's
#: turn count could exceed its step log by however many questions the model
#: repeated and the row was unreadable against the one rule that makes a
#: trajectory audit evidence: every completed model action is in it. A ``0.8.0``
#: run and a ``0.9.0`` run therefore write different trajectories, different step
#: counts and different counter relationships from the same provider behaviour,
#: and a resume across the boundary is refused.
RUNNER_CONTRACT_VERSION = "0.9.0"

# -- what kind of run this is ------------------------------------------------
#
# The kind of run is an explicit, stored, hashed field. It is *derived*
# from the adapter's provider rather than supplied, because a caller-chosen
# track is one more thing that can be wrong; and it is re-derived wherever a
# manifest is loaded, supplied or executed, because a stored field is a claim
# until something checks it.

#: No service was contacted: the adapter is in this process.
RUN_TRACK_SYNTHETIC_FAKE = "synthetic_infrastructure_smoke"
#: The plan is to execute against a model hosted by an external provider.
RUN_TRACK_PROVIDER_EXECUTION = "external_provider_execution"

RUN_STATUS_SYNTHETIC_FAKE = "SYNTHETIC_INFRASTRUCTURE_SMOKE"
RUN_STATUS_PROVIDER_EXECUTION = "EXTERNAL_PROVIDER_EXECUTION"

#: Unchanged, word for word, from the build that shipped it: a fake run's honest
#: account of itself was never the problem.
RUN_SCOPE_SYNTHETIC_FAKE = (
    "Synthetic infrastructure smoke run over a two-Cube methodology spike with a "
    "deterministic fake adapter. This is not a model benchmark result: no model "
    "was executed, no score is computed and no ranking of any kind is supported."
)

#: What a provider run may honestly say about itself. It executed; it measured
#: nothing that supports a comparison; and the manifest is a plan rather than a
#: receipt.
RUN_SCOPE_PROVIDER_EXECUTION = (
    "Execution of the same synthetic two-Cube methodology spike suite against a "
    "model hosted by an external provider, through this build's provider "
    "adapter, under the same scaffold, evaluator, manifest and ledger as the "
    "fake track. This is an execution record and not a model benchmark result: "
    "the semantic n is 2 Cubes / 12 variants, so no score, no ranking, no "
    "aggregate statistic and no public benchmark claim is computed or supported "
    "at this size. Creating this manifest fixes the plan and does not assert "
    "that any provider call was made, returned or completed; only ledger rows "
    "record episodes that actually ran."
)

#: Said in every report, because a plan and a receipt look alike on disk.
RUN_PLAN_NOTE = (
    "A run manifest is the plan, written before the first episode. Its existence "
    "records what this build was configured to execute and never that anything "
    "was executed: only ledger rows record episodes that ran, and within a row "
    "only a non-null usage field records a provider response that actually "
    "returned. 'planned' and 'completed' in the counts above are therefore "
    "different claims and are never read as one."
)


@dataclass(frozen=True)
class RunIdentityScope:
    """The three fields that say what kind of run this is, as one unit.

    Kept together rather than as three loose constants because they are only
    ever correct together: a manifest carrying one track's status beside
    another's scope describes no run this build can execute, and deriving all
    three from one lookup is what makes that unrepresentable.
    """

    track: str
    status: str
    scope: str


#: Every track this build can plan, by name. A stored track that is not a key
#: here was written by a different build and is refused rather than read.
RUN_TRACKS: Mapping[str, RunIdentityScope] = MappingProxyType(
    {
        RUN_TRACK_SYNTHETIC_FAKE: RunIdentityScope(
            track=RUN_TRACK_SYNTHETIC_FAKE,
            status=RUN_STATUS_SYNTHETIC_FAKE,
            scope=RUN_SCOPE_SYNTHETIC_FAKE,
        ),
        RUN_TRACK_PROVIDER_EXECUTION: RunIdentityScope(
            track=RUN_TRACK_PROVIDER_EXECUTION,
            status=RUN_STATUS_PROVIDER_EXECUTION,
            scope=RUN_SCOPE_PROVIDER_EXECUTION,
        ),
    }
)


def run_track_for_provider(provider: str) -> RunIdentityScope:
    """Which track a run against this provider is on. Total, and closed.

    The in-process providers are a closed set of reserved names this build owns
    (:data:`~boundarybench.adapter.IN_PROCESS_PROVIDERS`); nothing else can be
    one, so everything else is an external service. Defaulting the *other* way
    — treating an unrecognised provider as a fake — is the failure mode this
    function exists to prevent, because it is the one that produces a false
    claim rather than an over-cautious one.
    """
    if provider in IN_PROCESS_PROVIDERS:
        return RUN_TRACKS[RUN_TRACK_SYNTHETIC_FAKE]
    return RUN_TRACKS[RUN_TRACK_PROVIDER_EXECUTION]


#: Zero, and pinned. A silent retry would turn a provider failure into a second
#: sample of the same episode, which is not a thing this run can measure.
#:
#: A read-only proxy, not a dict. Both of these constants are inside run
#: identity: every manifest hashes them, every stored manifest is checked
#: against them and every execution re-derives its provenance from them. A
#: module-level dict is writable by anything that can import the module, so one
#: assignment would silently redefine what "this build" means for every run in
#: the process — including the ones already on disk.
RETRY_POLICY: Mapping[str, Any] = MappingProxyType({"retries": 0, "policy": "none"})

#: The four contracts this build implements. A run manifest that names any other
#: combination was produced by a different build and cannot be executed here.
BUILD_CONTRACT_VERSIONS: Mapping[str, str] = MappingProxyType(
    {
        "package": __version__,
        "runner": RUNNER_CONTRACT_VERSION,
        "evaluator": EVALUATOR_CONTRACT_VERSION,
        "adapter": ADAPTER_CONTRACT_VERSION,
    }
)

#: Key fragments that mark a value as a credential. Matched against the key with
#: every non-alphanumeric character stripped, so ``api_key``, ``apiKey``,
#: ``API-KEY`` and ``x_api_key`` are all one rule.
_SECRET_FRAGMENTS: tuple[str, ...] = (
    "apikey",
    "accesskey",
    "secretkey",
    "privatekey",
    "secret",
    "token",
    "password",
    "passwd",
    "authorization",
    "credential",
    "bearer",
    "sessionkey",
)

#: Settings whose *name* trips the screen above and which are demonstrably not
#: credentials. One entry, matched exactly against the flattened key, and only
#: honoured when the value is a plain integer.
#:
#: The pairing is what makes this safe rather than a hole. ``max_tokens`` is the
#: Anthropic Messages API's own name for the output ceiling and it is a
#: result-affecting setting that has to be pinned in run identity, so it has to
#: be sayable; but every credential is a *string*, and every setting an operator
#: can supply from the command line is a string too (``--setting`` never guesses
#: a type). So this exemption is unreachable from operator input, and a key
#: named ``max_output_tokens`` carrying anything a secret could be is still
#: refused by name.
_SECRET_EXEMPT_INTEGER_KEYS: frozenset[str] = frozenset({"maxoutputtokens"})

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

# -- execution identity ------------------------------------------------------
#
# A configuration is a plan and can be described. An execution is a thing that
# happened once, and the only honest way to name one is to mint a fresh name for
# it. The generator is the platform's cryptographic source rather than a counter
# or a clock: an id derived from anything observable would collide between two
# operators who started the same plan at the same second, which is exactly the
# case that has to stay distinguishable.

#: A version-4 UUID in canonical lowercase form. Checked by *shape*, and the
#: shape includes the version and variant nibbles, so a stored id that was not
#: minted as a random UUID is refused rather than read.
_EXECUTION_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

#: RFC 3339 in UTC, to fixed microsecond precision, with the ``Z`` offset. One
#: rendering, so two timestamps are comparable as strings and a digest over one
#: cannot be defeated by re-rendering the same instant a different way.
_UTC_TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")

#: Reads the wall clock. Injected everywhere it is used so a test can state the
#: instant a record claims instead of observing whatever the machine says.
WallClock = Callable[[], datetime]

#: Mints one execution id. Injected for the same reason.
ExecutionIdFactory = Callable[[], str]


def utc_now() -> datetime:
    """The wall clock, in UTC, as an aware datetime. The default injection."""
    return datetime.now(UTC)


def new_execution_id() -> str:
    """One fresh execution id, from the platform's cryptographic generator."""
    return str(uuid.uuid4())


def format_utc_timestamp(moment: datetime) -> str:
    """Render one instant in the single stored form, or refuse it.

    A naive datetime is refused rather than assumed to be UTC. "Assume UTC" is
    right on a server and wrong on a laptop, and the failure is silent: the
    record would state an instant that is hours from the one that happened.
    """
    if not isinstance(moment, datetime) or moment.tzinfo is None:
        raise RunManifestFormatError(
            f"a run timestamp must be a timezone-aware datetime, got {moment!r}; a "
            "naive datetime is refused rather than assumed to be UTC, because "
            "assuming wrongly records an instant that never happened"
        )
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def check_utc_timestamp(value: Any, name: str) -> str:
    """Prove a stored timestamp is the one rendering this build writes."""
    if not isinstance(value, str) or not _UTC_TIMESTAMP_PATTERN.fullmatch(value):
        raise RunManifestFormatError(
            f"{name} must be an RFC 3339 UTC timestamp of the form "
            f"'YYYY-MM-DDTHH:MM:SS.ffffffZ', got {value!r}"
        )
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f%z")
    except ValueError as exc:
        raise RunManifestFormatError(f"{name} is not a real instant: {value!r}") from exc
    return value


def check_execution_id(value: Any, name: str = "execution_id") -> str:
    """Prove a stored execution id is a canonical version-4 UUID."""
    if not isinstance(value, str) or not _EXECUTION_ID_PATTERN.fullmatch(value):
        raise RunManifestFormatError(
            f"{name} must be a lowercase canonical version-4 UUID, got {value!r}; "
            "an execution id names one thing that happened once and is minted from "
            "the platform's cryptographic generator, never derived from the plan"
        )
    return value


def execution_binding_payload(
    *, configuration_id: str, execution_id: str, created_at_utc: str
) -> dict[str, str]:
    """The three execution facts that are only meaningful together.

    Hashed as one unit so none of them can be edited independently of the
    others: an execution id repointed at a different configuration, or a
    creation time moved to another provider date, breaks the digest.
    """
    return {
        "configuration_id": configuration_id,
        "execution_id": execution_id,
        "created_at_utc": created_at_utc,
    }


class RunManifestError(ValueError):
    """Base class for every run-manifest failure."""


class RunManifestFormatError(RunManifestError):
    """The manifest, or a parameter offered for one, is malformed."""


class RunSecretError(RunManifestError):
    """An adapter setting looks like a credential and must never be persisted."""


class RunManifestMismatchError(RunManifestError):
    """A stored manifest is not the run that was requested, or has been edited."""


class RunPathError(RunManifestError):
    """An output path is unsafe or unusable as a run directory."""


class RunLockError(RunManifestError):
    """Another writer holds this run, or the lock itself cannot be taken."""


class RunIOError(RunManifestError):
    """The filesystem refused an operation this run depends on.

    Named rather than left as an ``OSError`` so the CLI can report it as a run
    failure instead of a traceback: an unwritable output directory is an
    operator mistake, not a crash.
    """


# -- parameters --------------------------------------------------------------


@dataclass(frozen=True)
class RunLimits:
    """The budget one episode may spend before it is cut off.

    Three independent ceilings, because they bound different things. ``max_turns``
    bounds how many decisions the agent gets. ``max_messages`` bounds the traffic
    that buys those decisions — one turn is one request and one response, so a
    turn costs :data:`~boundarybench.runner.MESSAGES_PER_TURN` messages, and a
    scaffold that ever sends more than one message per decision would blow a
    message budget while the turn budget still looked generous.
    ``episode_timeout_seconds`` bounds wall-clock, which neither of the others
    can: a single call can block for longer than a whole run is allowed to take.
    """

    max_turns: int
    max_messages: int
    episode_timeout_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_turns": self.max_turns,
            "max_messages": self.max_messages,
            "episode_timeout_seconds": self.episode_timeout_seconds,
        }


@dataclass(frozen=True)
class EpisodePlanEntry:
    """One planned episode: which variant, which trial."""

    variant_id: str
    trial_index: int

    @property
    def episode_id(self) -> str:
        """Stable and readable: the plan entry is its own name."""
        return f"{self.variant_id}#t{self.trial_index:03d}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "variant_id": self.variant_id,
            "trial_index": self.trial_index,
        }


# -- validation --------------------------------------------------------------


def _positive_int(value: Any, name: str) -> int:
    # bool is an int subclass, so `trials=True` must not read as one trial.
    if type(value) is not int or value < 1:
        raise RunManifestFormatError(f"{name} must be a positive integer, got {value!r}")
    return value


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunManifestFormatError(
            f"{name} must be a positive, finite number, got {value!r}"
        )
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise RunManifestFormatError(
            f"{name} must be a positive, finite number, got {value!r}"
        )
    return number


def normalized_adapter_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The one normalisation adapter settings pass through, wherever they arrive.

    Used both when a manifest is built and when a live adapter's settings are
    compared against it, so the comparison cannot fail — or pass — because two
    call sites normalised differently. The result is plain built-ins: it is what
    gets compared and hashed, and the manifest freezes its own copy separately.

    Structure is proven serialisable *before* the secret screen walks it, so a
    self-referential or unreasonably deep settings mapping is a named
    ``RunManifestError`` rather than an interpreter recursion failure from an
    arbitrary frame.
    """
    try:
        ensure_json_safe(settings, "adapter_settings")
    except JsonSafetyError as exc:
        raise RunManifestFormatError(str(exc)) from exc
    normalized = _normalize_settings(dict(settings), "adapter_settings")
    assert isinstance(normalized, dict)
    return normalized


def _normalize_settings(value: Any, context: str) -> Any:
    """Screen adapter settings for secrets and put them in canonical order.

    Only that. Whether the value can be serialised at all was settled by
    :func:`normalized_adapter_settings` before this walk started, so there is
    exactly one definition of JSON-safety in the build and this is not a second
    one drifting alongside it.
    """
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            flattened = _NON_ALNUM.sub("", key.lower())
            # ``bool`` is an ``int`` subclass, so the type test is exact: nothing
            # is exempted on the strength of being ``True``.
            exempt = flattened in _SECRET_EXEMPT_INTEGER_KEYS and type(item) is int
            for fragment in _SECRET_FRAGMENTS:
                if fragment in flattened and not exempt:
                    raise RunSecretError(
                        f"{context}: adapter setting {key!r} looks like a "
                        f"credential ({fragment!r}); secrets are refused rather "
                        "than redacted, so a run manifest can never carry one. "
                        "Pass credentials through the environment instead."
                    )
            normalized[key] = _normalize_settings(item, f"{context}.{key}")
        return dict(sorted(normalized.items()))
    if not isinstance(value, (str, bytes)) and isinstance(value, Sequence):
        return [
            _normalize_settings(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    return value


# -- the manifest ------------------------------------------------------------


@dataclass(frozen=True)
class RunManifest:
    """One execution of one fixed configuration.

    ``configuration_id`` answers "what was this run configured to do?" and is a
    pure function of the fields below it. ``execution_id``, ``created_at_utc``
    and ``execution_digest`` answer "which run was this, and when did it start?"
    and are minted once per new run directory.
    """

    schema_version: int
    configuration_id: str
    execution_id: str
    created_at_utc: str
    execution_digest: str
    suite_id: str
    suite_benchmark_version: str
    suite_content_digest: str
    scaffold_id: str
    scaffold_version: str
    scaffold_content_digest: str
    adapter: AdapterIdentity
    adapter_settings: Mapping[str, Any]
    trials: int
    limits: RunLimits
    #: The hard cost and episode limits this run was authorised to spend, and
    #: the pricing policy its spend is measured at. Inside configuration
    #: identity, so a resume under different limits is refused rather than
    #: silently permitted.
    cost_controls: CostControls
    episode_plan: tuple[EpisodePlanEntry, ...]
    package_version: str
    runner_contract_version: str
    evaluator_contract_version: str
    adapter_contract_version: str
    run_track: str
    run_status: str
    run_scope: str

    def __post_init__(self) -> None:
        """Freeze the settings, and refuse a manifest that misdescribes its run.

        ``configuration_id`` is a hash of ``configuration_payload()``, and every
        other field of this dataclass is frozen by the dataclass itself — except
        the *contents* of ``adapter_settings``, which is a mapping. A frozen
        dataclass holding a live dict is not immutable: the settings could be
        edited after the id was computed, and the manifest would then describe a
        run its own id does not name. Freezing happens here rather than only in
        :func:`build_run_manifest` so a directly constructed manifest and a
        loaded one carry the same guarantee.

        The track check is here for the same reason. ``RunManifest`` is public
        and its fields are ordinary arguments, so a caller can assemble one that
        names Anthropic as its provider and the fake track as its scope, and
        hash it perfectly — every digest check in the build would then agree
        with a manifest whose central claim about itself is false. Coherence is
        therefore a property of *holding* one of these at all, checked at the
        single point every built, loaded and supplied manifest passes through.
        """
        object.__setattr__(self, "adapter_settings", deep_freeze(self.adapter_settings))
        expected = run_track_for_provider(self.adapter.provider)
        actual = (self.run_track, self.run_status, self.run_scope)
        if actual != (expected.track, expected.status, expected.scope):
            raise RunManifestMismatchError(
                f"a run against provider {self.adapter.provider!r} is on the "
                f"{expected.track!r} track, but this manifest states track "
                f"{self.run_track!r} with status {self.run_status!r}. A run's track, "
                "status and scope are what it claims to be; they follow from the "
                "provider it runs against and are never stated independently of it."
            )

    @property
    def retry_policy(self) -> Mapping[str, Any]:
        return RETRY_POLICY

    @property
    def planned_episode_ids(self) -> tuple[str, ...]:
        return tuple(entry.episode_id for entry in self.episode_plan)

    @property
    def contract_versions(self) -> Mapping[str, str]:
        """The four contracts a result is only comparable within.

        Stated once here because run identity, every ledger row and the resume
        check all have to agree on what "the same contracts" means.
        """
        return {
            "package": self.package_version,
            "runner": self.runner_contract_version,
            "evaluator": self.evaluator_contract_version,
            "adapter": self.adapter_contract_version,
        }

    @property
    def run_id(self) -> str:
        """Compatibility alias for :attr:`configuration_id`. Documented, not stored.

        The alias answers with deterministic configuration identity so callers'
        comparisons remain meaningful. It is deliberately *not* a stored field:
        a serialised
        ``run_id`` is exactly the conflation this schema version removed, and a
        reader of the file has to be told which of the two ids it is holding.
        """
        return self.configuration_id

    def configuration_payload(self) -> dict[str, Any]:
        """Everything that changes execution or grading, and nothing else.

        The three execution fields are excluded on purpose and not by omission:
        when a run started, and which run it was, do not change what the run
        does, so two executions of one plan must agree here or the resume check
        has nothing to compare. ``configuration_id`` itself is excluded because
        it is the hash of this payload.
        """
        return {
            "schema_version": self.schema_version,
            "suite": {
                "suite_id": self.suite_id,
                "benchmark_version": self.suite_benchmark_version,
                "suite_content_digest": self.suite_content_digest,
            },
            "scaffold": {
                "scaffold_id": self.scaffold_id,
                "scaffold_version": self.scaffold_version,
                "content_digest": self.scaffold_content_digest,
            },
            "adapter": self.adapter.as_dict(),
            # ``to_json``, not ``dict``: the stored settings are deep-frozen, so a
            # shallow copy would hand read-only proxies and tuples to the JSON
            # encoder, which cannot serialise either.
            "adapter_settings": to_json(self.adapter_settings),
            "trials": self.trials,
            "limits": self.limits.as_dict(),
            "cost_controls": self.cost_controls.as_dict(),
            "retry_policy": dict(RETRY_POLICY),
            "episode_plan": [entry.as_dict() for entry in self.episode_plan],
            "contract_versions": dict(self.contract_versions),
            # All three, not just the track they follow from. A reader of the
            # stored file gets the claim in full rather than a key it has to
            # look up in a build it may not have, and an editor who rewrites one
            # of them has to rewrite the provider too — which is the check.
            "track": self.run_track,
            "status": self.run_status,
            "scope": self.run_scope,
        }

    def execution_payload(self) -> dict[str, str]:
        """The three execution facts, as one hashed unit."""
        return execution_binding_payload(
            configuration_id=self.configuration_id,
            execution_id=self.execution_id,
            created_at_utc=self.created_at_utc,
        )

    def as_dict(self) -> dict[str, Any]:
        """The stored form: the configuration, and the execution that ran it."""
        payload = self.configuration_payload()
        payload["configuration_id"] = self.configuration_id
        payload["execution"] = {
            **self.execution_payload(),
            "execution_digest": self.execution_digest,
        }
        return payload


def canonical_json(payload: Any, context: str = "run record") -> str:
    """The one encoding used for every digest and every persisted record.

    The payload is proven serialisable first, so a lone surrogate, a cycle or an
    over-deep structure is a named failure instead of a ``UnicodeEncodeError``
    or a ``RecursionError`` raised from inside the encoder.
    """
    try:
        return canonical_json_text(payload, context)
    except JsonSafetyError as exc:
        raise RunManifestFormatError(str(exc)) from exc


def run_identity_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json(payload, "run identity payload").encode("utf-8")
    ).hexdigest()


def configuration_digest(payload: Mapping[str, Any]) -> str:
    """The deterministic name of one configuration."""
    return run_identity_digest(payload)


def execution_digest_of(
    *, configuration_id: str, execution_id: str, created_at_utc: str
) -> str:
    """Bind an execution's id and creation time to the configuration it ran.

    This is an internal-consistency digest and nothing more. It detects an edit
    to one of the three fields that was not carried through to the others, which
    is the realistic corruption; it cannot detect a rewrite of all four together,
    because it is unkeyed and lives in the same writable file. Creation time is
    therefore evidence of what this build recorded, and never a proof that the
    record has not been replaced since.
    """
    return run_identity_digest(
        execution_binding_payload(
            configuration_id=configuration_id,
            execution_id=execution_id,
            created_at_utc=created_at_utc,
        )
    )


def verify_manifest_identity(manifest: RunManifest) -> None:
    """Re-derive a supplied manifest's identity before anything trusts it.

    A ``RunManifest`` handed in by a caller is a claim about a run in exactly
    the way a stored one is, and the stored one has been rehashed since the
    first review. This is the same object seen from the other side:
    ``RunManifest`` is a frozen dataclass, so ``dataclasses.replace`` produces
    one with new limits and the *old* ``configuration_id``. Comparing that id
    against the run on disk then agrees, and every episode executes under a
    budget the immutable manifest never pinned while the ledger records the
    original identity — a run that claims twelve turns and spent one.

    Both digests are re-derived, because they answer different questions and
    each is forgeable on its own. The configuration digest proves the manifest
    describes the inputs its name claims; the execution digest proves its
    execution id and creation time still belong to that configuration and to
    each other, so a row's timestamps cannot be repointed at a different run
    while every other check keeps agreeing.

    Rehashing happens before a directory is created, a lock is taken or a row is
    read.
    """
    recomputed = configuration_digest(manifest.configuration_payload())
    if recomputed != manifest.configuration_id:
        raise RunManifestMismatchError(
            f"the supplied run manifest does not hash to its own configuration_id "
            f"({manifest.configuration_id} recorded, {recomputed} computed). A "
            "manifest is a claim about a run until it is re-derived; this one "
            "describes different inputs from the ones its identity names, so nothing "
            "may execute under it."
        )
    check_execution_id(manifest.execution_id)
    check_utc_timestamp(manifest.created_at_utc, "created_at_utc")
    recomputed_execution = execution_digest_of(
        configuration_id=manifest.configuration_id,
        execution_id=manifest.execution_id,
        created_at_utc=manifest.created_at_utc,
    )
    if recomputed_execution != manifest.execution_digest:
        raise RunManifestMismatchError(
            f"the run manifest's execution record does not hash to its own "
            f"execution_digest ({manifest.execution_digest} recorded, "
            f"{recomputed_execution} computed). The execution id, the creation "
            "timestamp and the configuration they belong to are recorded as one "
            "unit precisely so none of them can be moved without the others."
        )


def check_episode_limit(planned: int, controls: CostControls) -> None:
    """Refuse a plan larger than the number of episodes that were authorised.

    Checked here, where the plan is first assembled, so a run whose plan exceeds
    its authorisation never reaches a directory, a lock or a provider. The
    boundary is equality: a plan of exactly the authorised number is exactly
    what was authorised, and refusing it would stop a run that fits.
    """
    if controls.max_episodes is not None and planned > controls.max_episodes:
        raise RunManifestFormatError(
            f"this configuration plans {planned} episode(s) and the run was "
            f"authorised for at most {controls.max_episodes}. The plan is refused "
            "here, before a run directory or a provider client exists, because an "
            "episode ceiling that only stopped execution would already have "
            "recorded a plan nobody approved."
        )


def build_run_manifest(
    *,
    suite: SuiteReport,
    scaffold: Scaffold,
    provider: str,
    model: str,
    implementation: str,
    adapter_version: str,
    adapter_settings: Mapping[str, Any],
    trials: int,
    limits: RunLimits,
    cost_controls: CostControls | None = None,
    now: WallClock = utc_now,
    execution_id_factory: ExecutionIdFactory = new_execution_id,
) -> RunManifest:
    """Fix a run's configuration, and mint a candidate execution for it.

    The execution minted here is a *candidate*: it names this invocation, and it
    is only kept if this invocation turns out to be the one that creates the run
    directory. :func:`open_run_session` discards it in favour of the stored one
    when the directory already holds this configuration, so generating a fresh
    id can never make an exact resume impossible — see there.

    The clock and the id generator are parameters so a test can state the
    execution a record claims rather than observe whatever the machine did.
    """
    checked_trials = _positive_int(trials, "trials")
    checked_limits = RunLimits(
        max_turns=_positive_int(limits.max_turns, "max_turns"),
        max_messages=_positive_int(limits.max_messages, "max_messages"),
        episode_timeout_seconds=_positive_number(
            limits.episode_timeout_seconds, "episode_timeout_seconds"
        ),
    )
    settings = normalized_adapter_settings(adapter_settings)
    for name, value in (
        ("provider", provider),
        ("model", model),
        ("implementation", implementation),
        ("adapter_version", adapter_version),
    ):
        try:
            check_text(value, f"adapter {name}")
        except JsonSafetyError as exc:
            raise RunManifestFormatError(str(exc)) from exc

    plan = tuple(
        EpisodePlanEntry(variant_id=variant.variant_id, trial_index=trial)
        for cube in suite.cubes
        for variant in cube.cube.variants
        for trial in range(1, checked_trials + 1)
    )
    # A run with no limits still says so, rather than omitting the block: the
    # absence of a cap is a fact about the run and a reader has to be told it.
    controls = cost_controls or CostControls(
        max_cost_usd=None, max_episodes=None, price=None
    )
    check_episode_limit(len(plan), controls)

    # Derived, never a parameter. A caller who could pass the track could pass
    # the wrong one, and the wrong one is precisely the defect this replaces.
    identity_scope = run_track_for_provider(provider)

    created_at_utc = format_utc_timestamp(now())
    execution_id = check_execution_id(execution_id_factory())

    manifest = RunManifest(
        schema_version=RUN_MANIFEST_SCHEMA_VERSION,
        configuration_id="",
        execution_id=execution_id,
        created_at_utc=created_at_utc,
        execution_digest="",
        suite_id=suite.manifest.suite_id,
        suite_benchmark_version=suite.manifest.benchmark_version,
        suite_content_digest=suite.content_digest,
        scaffold_id=scaffold.scaffold_id,
        scaffold_version=scaffold.scaffold_version,
        scaffold_content_digest=scaffold.content_digest,
        adapter=AdapterIdentity(
            provider=provider,
            model=model,
            implementation=implementation,
            version=adapter_version,
        ),
        adapter_settings=settings,
        trials=checked_trials,
        limits=checked_limits,
        cost_controls=controls,
        episode_plan=plan,
        package_version=__version__,
        runner_contract_version=RUNNER_CONTRACT_VERSION,
        evaluator_contract_version=EVALUATOR_CONTRACT_VERSION,
        adapter_contract_version=ADAPTER_CONTRACT_VERSION,
        run_track=identity_scope.track,
        run_status=identity_scope.status,
        run_scope=identity_scope.scope,
    )
    identity = configuration_digest(manifest.configuration_payload())
    return RunManifest(
        **{
            **{
                field: getattr(manifest, field) for field in manifest.__dataclass_fields__
            },
            "configuration_id": identity,
            "execution_digest": execution_digest_of(
                configuration_id=identity,
                execution_id=execution_id,
                created_at_utc=created_at_utc,
            ),
        }
    )


# -- persistence -------------------------------------------------------------

MANIFEST_FILENAME = "run_manifest.json"
LEDGER_FILENAME = "episodes.jsonl"
#: The advisory lock one writer holds for the whole run. Never read as data.
LOCK_FILENAME = ".run.lock"


@dataclass(frozen=True)
class RunPaths:
    """Where a run lives on disk."""

    root: Path
    manifest_path: Path
    ledger_path: Path
    lock_path: Path


def check_run_path_safety(root: Path) -> None:
    """Refuse the run root, any file in it, or *any ancestor*, that is a symlink.

    Checking only the final component is not enough: a link standing in for a
    parent directory redirects the whole run just as effectively, and the
    operator auditing ``--output-dir`` would never see it. Every existing
    component of the absolute path is checked, so the run root can only be
    reached by walking real directories.
    """
    absolute = root if root.is_absolute() else Path(os.getcwd()) / root
    for ancestor in reversed(absolute.parents):
        if ancestor.is_symlink():
            raise RunPathError(
                f"{ancestor} is a symlink on the path to {root}; a run directory "
                "must be reachable through real directories only, so the manifest "
                "and ledger cannot be redirected"
            )
    if absolute.is_symlink():
        raise RunPathError(
            f"{root} is a symlink; a run directory must be a real directory so "
            "the manifest and ledger cannot be redirected"
        )
    if absolute.exists() and not absolute.is_dir():
        raise RunPathError(f"{root} exists and is not a directory")
    for name in (MANIFEST_FILENAME, LEDGER_FILENAME, LOCK_FILENAME):
        path = absolute / name
        if path.is_symlink():
            raise RunPathError(
                f"{path} is a symlink; run records are written in place, never "
                "through a link"
            )


def resolve_run_paths(output_dir: str | Path) -> RunPaths:
    """Resolve, and prove safe, the three files a run writes."""
    root = Path(output_dir)
    check_run_path_safety(root)
    return RunPaths(
        root=root,
        manifest_path=root / MANIFEST_FILENAME,
        ledger_path=root / LEDGER_FILENAME,
        lock_path=root / LOCK_FILENAME,
    )


def _fsync_directory(directory: Path) -> None:
    handle = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def write_run_manifest(paths: RunPaths, manifest: RunManifest) -> None:
    """Write the manifest atomically, and only if there is not one already.

    The payload goes to a temporary file created *exclusively* under an
    unpredictable name — :func:`tempfile.mkstemp` opens with ``O_CREAT|O_EXCL``,
    so a symlink left at a guessable scratch path is never followed and cannot
    redirect the write onto a file the operator did not choose. The temporary is
    flushed, fsynced and then linked into place; ``os.link`` fails when the
    destination exists, so the manifest is created exactly once and a crash
    mid-write leaves the scratch file rather than half a manifest.

    The identity is re-derived here, immediately before the bytes are produced,
    and not only by whoever assembled the manifest. This is the last point at
    which a manifest is still a claim; after it, the file on disk *is* the run.
    A manifest whose payload no longer hashes to its recorded ``run_id`` would
    otherwise be written under a stale identity, and every ledger row appended
    afterwards would assert that identity as provenance.
    """
    verify_manifest_identity(manifest)
    rendered = json.dumps(manifest.as_dict(), indent=2, ensure_ascii=False) + "\n"
    descriptor, name = tempfile.mkstemp(
        dir=paths.root, prefix=f".{MANIFEST_FILENAME}.", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, paths.manifest_path)
        except FileExistsError as exc:
            raise RunManifestMismatchError(
                f"{paths.manifest_path} appeared while this run was starting; a run "
                "manifest is written once and never overwritten"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(paths.root)


def _stored_mapping(raw: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = raw[key]
    if not isinstance(value, Mapping):
        raise RunManifestFormatError(f"{context}: {key!r} must be a JSON object")
    return value


def _stored_text(raw: Mapping[str, Any], key: str, context: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise RunManifestFormatError(f"{context}: {key!r} must be a non-empty string")
    return value


def _stored_int(raw: Mapping[str, Any], key: str, context: str) -> int:
    value = raw[key]
    # bool is an int subclass, so `true` must not be read as one.
    if type(value) is not int or value < 1:
        raise RunManifestFormatError(f"{context}: {key!r} must be a positive integer")
    return value


def _stored_digest(raw: Mapping[str, Any], key: str, context: str) -> str:
    """A pinned digest is a digest, not merely a non-empty string."""
    value = raw[key]
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise RunManifestFormatError(
            f"{context}: {key!r} must be a lowercase 64-character SHA-256 hex "
            f"digest, got {value!r}"
        )
    return value


def _require_keys(raw: Mapping[str, Any], allowed: tuple[str, ...], context: str) -> None:
    missing = [key for key in allowed if key not in raw]
    if missing:
        raise RunManifestFormatError(f"{context}: missing required field(s) {missing}")
    unknown = sorted(set(map(str, raw)) - set(allowed))
    if unknown:
        raise RunManifestFormatError(
            f"{context}: unknown field(s) {unknown}; allowed: {sorted(allowed)}"
        )


_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "schema_version",
    "configuration_id",
    "execution",
    "suite",
    "scaffold",
    "adapter",
    "adapter_settings",
    "trials",
    "limits",
    "cost_controls",
    "retry_policy",
    "episode_plan",
    "contract_versions",
    "track",
    "status",
    "scope",
)
_EXECUTION_KEYS = (
    "configuration_id",
    "execution_id",
    "created_at_utc",
    "execution_digest",
)
_SUITE_KEYS = ("suite_id", "benchmark_version", "suite_content_digest")
_SCAFFOLD_KEYS = ("scaffold_id", "scaffold_version", "content_digest")
_ADAPTER_KEYS = ("provider", "model", "implementation", "version")
_LIMIT_KEYS = ("max_turns", "max_messages", "episode_timeout_seconds")
_PLAN_KEYS = ("episode_id", "variant_id", "trial_index")
_CONTRACT_KEYS = ("package", "runner", "evaluator", "adapter")


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    for key, _value in pairs:
        if key in seen:
            raise RunManifestFormatError(
                f"duplicate JSON key {key!r}; a run manifest states each key once"
            )
        seen.add(key)
    return dict(pairs)


def _check_stored_track(
    raw: Mapping[str, Any], provider: str, context: str
) -> RunIdentityScope:
    """Prove the stored track, status and scope are the ones this provider gets.

    Each of the three is compared to what :func:`run_track_for_provider` derives
    from the provider the *same file* records, and named individually when it
    disagrees. Rehashing after an edit cannot get past this: the digest only
    proves the file is self-consistent, and a manifest that quietly relabels an
    Anthropic run as a fake infrastructure smoke is perfectly self-consistent.

    An unknown track is refused before the comparison, because a build that read
    a track it does not implement would be guessing at what the run claimed.
    """
    track = raw["track"]
    if not isinstance(track, str) or track not in RUN_TRACKS:
        raise RunManifestFormatError(
            f"{context}: 'track' must be one of {sorted(RUN_TRACKS)}, got {track!r}; "
            "a run manifest naming a track this build does not implement was "
            "written by a different build and is refused rather than reinterpreted"
        )
    expected = run_track_for_provider(provider)
    for key, value in (
        ("track", expected.track),
        ("status", expected.status),
        ("scope", expected.scope),
    ):
        if raw[key] != value:
            raise RunManifestMismatchError(
                f"{context}: a run against provider {provider!r} is on the "
                f"{expected.track!r} track, so its {key!r} is fixed by this build "
                f"and this manifest states something else. The digest cannot catch "
                "this: it is recomputable by anyone who can edit the file, so the "
                "track, status and scope are checked against the provider instead."
            )
    return expected


def _reject_constant(name: str) -> Any:
    raise RunManifestFormatError(f"{name} is not a finite JSON value")


def stored_cost_controls(
    raw: Mapping[str, Any], provider: str, model: str, context: str
) -> CostControls:
    """Read the stored limits, and re-derive their price from this build.

    The price is looked up again rather than trusted from the file. Recomputing
    an unkeyed digest is free to anyone who can edit the manifest, so a stored
    rate proves nothing; the shipped price table is the authority, and a run
    whose recorded policy version or digest is not this build's is refused
    explicitly rather than quietly re-priced under a table it never used.
    """
    try:
        return CostControls.from_stored(
            _stored_mapping(raw, "cost_controls", context), provider=provider, model=model
        )
    except BudgetError as exc:
        raise RunManifestFormatError(f"{context}: {exc}") from exc


def load_run_manifest(path: str | Path) -> RunManifest:
    """Read a stored manifest strictly under the public JSON depth limit."""
    return _load_run_manifest(Path(path))


def _load_run_manifest(path: Path) -> RunManifest:
    context = str(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise RunManifestFormatError(f"{context} is not valid UTF-8: {exc}") from exc
    except OSError as exc:
        raise RunManifestFormatError(
            f"cannot read run manifest {context}: {exc}"
        ) from exc
    try:
        ensure_raw_json_depth(text, context)
        raw = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except NestingDepthError as exc:
        raise RunManifestFormatError(str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise RunManifestFormatError(f"{context} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise RunManifestFormatError(f"{context}: a run manifest must be a JSON object")
    try:
        ensure_json_safe(raw, f"{context}: run manifest")
    except JsonSafetyError as exc:
        raise RunManifestFormatError(str(exc)) from exc

    # The version is read before the field set, and before any field is
    # interpreted. A manifest from an older build is missing fields this one
    # requires, and reporting that as "missing required field(s)" describes the
    # symptom: what happened is that the file was written by a build whose
    # manifests mean something different, and it is refused as that.
    version = raw.get("schema_version")
    if type(version) is not int or version != RUN_MANIFEST_SCHEMA_VERSION:
        raise RunManifestFormatError(
            f"{context}: unsupported schema_version {version!r}; this build reads "
            f"{RUN_MANIFEST_SCHEMA_VERSION}. A run manifest written by a different "
            "build is refused rather than reinterpreted: its fields do not mean "
            "what this build's fields mean, and the ones it does not carry are "
            "not absent limits but unrecorded ones"
        )

    _require_keys(raw, _TOP_LEVEL_KEYS, context)

    configuration_id = _stored_digest(raw, "configuration_id", context)

    # The execution block is read strictly and checked against itself before the
    # configuration it names is rehashed. Its three facts are only meaningful as
    # a unit — an execution id belongs to one configuration and to one instant —
    # so each is proven well-formed, then all three are proven to be the ones the
    # stored digest was taken over.
    execution = _stored_mapping(raw, "execution", context)
    _require_keys(execution, _EXECUTION_KEYS, f"{context}: execution")
    execution_id = check_execution_id(execution["execution_id"])
    created_at_utc = check_utc_timestamp(execution["created_at_utc"], "created_at_utc")
    bound_configuration = _stored_digest(
        execution, "configuration_id", f"{context}: execution"
    )
    stored_execution_digest = _stored_digest(
        execution, "execution_digest", f"{context}: execution"
    )
    recomputed_execution = execution_digest_of(
        configuration_id=bound_configuration,
        execution_id=execution_id,
        created_at_utc=created_at_utc,
    )
    if recomputed_execution != stored_execution_digest:
        raise RunManifestMismatchError(
            f"{context}: the stored execution record does not hash to its recorded "
            f"execution_digest ({stored_execution_digest} recorded, "
            f"{recomputed_execution} computed). Its execution id or creation "
            "timestamp has been edited since the run was created; this run cannot "
            "be resumed."
        )
    if bound_configuration != configuration_id:
        raise RunManifestMismatchError(
            f"{context}: the execution record names configuration "
            f"{bound_configuration}, but this manifest states {configuration_id}. "
            "An execution is an execution *of* one configuration, so a rehashed "
            "execution block pointing at another one is refused: recomputing an "
            "unkeyed digest is free, and agreeing with the rest of the file is not."
        )

    suite = _stored_mapping(raw, "suite", context)
    _require_keys(suite, _SUITE_KEYS, f"{context}: suite")
    scaffold = _stored_mapping(raw, "scaffold", context)
    _require_keys(scaffold, _SCAFFOLD_KEYS, f"{context}: scaffold")
    adapter = _stored_mapping(raw, "adapter", context)
    _require_keys(adapter, _ADAPTER_KEYS, f"{context}: adapter")

    # The track and the provider are checked against each other before anything
    # else about the file is trusted. Both are stored, so a self-consistent
    # forgery — one that rehashes cleanly after the edit — is caught here or not
    # at all: the digest agrees with whatever the file now says.
    provider = _stored_text(adapter, "provider", f"{context}: adapter")
    model = _stored_text(adapter, "model", f"{context}: adapter")
    _check_stored_track(raw, provider, context)
    track = _stored_text(raw, "track", context)
    status = _stored_text(raw, "status", context)
    scope = _stored_text(raw, "scope", context)

    retries = raw["retry_policy"]
    # Not ordinary dict equality: `False == 0` in Python, so `{"retries": false}`
    # would compare equal to the pinned policy and then be silently reprojected
    # to zero. Types are checked exactly.
    if (
        not isinstance(retries, Mapping)
        or set(retries) != set(RETRY_POLICY)
        or type(retries["retries"]) is not int
        or retries["retries"] != RETRY_POLICY["retries"]
        or type(retries["policy"]) is not str
        or retries["policy"] != RETRY_POLICY["policy"]
    ):
        raise RunManifestFormatError(
            f"{context}: 'retry_policy' must be exactly {dict(RETRY_POLICY)!r} with "
            f"those types, got {retries!r}"
        )

    limits = _stored_mapping(raw, "limits", context)
    _require_keys(limits, _LIMIT_KEYS, f"{context}: limits")
    contracts = _stored_mapping(raw, "contract_versions", context)
    _require_keys(contracts, _CONTRACT_KEYS, f"{context}: contract_versions")

    controls = stored_cost_controls(raw, provider, model, context)

    plan_section = raw["episode_plan"]
    if isinstance(plan_section, (str, bytes)) or not isinstance(plan_section, Sequence):
        raise RunManifestFormatError(f"{context}: 'episode_plan' must be a list")
    if not plan_section:
        raise RunManifestFormatError(f"{context}: 'episode_plan' must not be empty")
    plan: list[EpisodePlanEntry] = []
    seen: set[str] = set()
    for index, item in enumerate(plan_section):
        entry_context = f"{context}: episode_plan[{index}]"
        if not isinstance(item, Mapping):
            raise RunManifestFormatError(f"{entry_context} must be a JSON object")
        _require_keys(item, _PLAN_KEYS, entry_context)
        entry = EpisodePlanEntry(
            variant_id=_stored_text(item, "variant_id", entry_context),
            trial_index=_stored_int(item, "trial_index", entry_context),
        )
        if item["episode_id"] != entry.episode_id:
            raise RunManifestFormatError(
                f"{entry_context}: episode_id {item['episode_id']!r} is not the id "
                f"of this plan entry ({entry.episode_id!r})"
            )
        if entry.episode_id in seen:
            raise RunManifestFormatError(
                f"{entry_context}: duplicate episode_id {entry.episode_id!r}"
            )
        seen.add(entry.episode_id)
        plan.append(entry)
    # Re-derived rather than trusted: the stored plan and the stored ceiling are
    # both editable, and a manifest naming more episodes than it authorised is
    # refused here rather than executed up to the ceiling and reported as though
    # the rest had never been planned.
    check_episode_limit(len(plan), controls)

    manifest = RunManifest(
        schema_version=version,
        configuration_id=configuration_id,
        execution_id=execution_id,
        created_at_utc=created_at_utc,
        execution_digest=stored_execution_digest,
        suite_id=_stored_text(suite, "suite_id", f"{context}: suite"),
        suite_benchmark_version=_stored_text(
            suite, "benchmark_version", f"{context}: suite"
        ),
        suite_content_digest=_stored_digest(
            suite, "suite_content_digest", f"{context}: suite"
        ),
        scaffold_id=_stored_text(scaffold, "scaffold_id", f"{context}: scaffold"),
        scaffold_version=_stored_text(
            scaffold, "scaffold_version", f"{context}: scaffold"
        ),
        scaffold_content_digest=_stored_digest(
            scaffold, "content_digest", f"{context}: scaffold"
        ),
        adapter=AdapterIdentity(
            provider=provider,
            model=model,
            implementation=_stored_text(adapter, "implementation", f"{context}: adapter"),
            version=_stored_text(adapter, "version", f"{context}: adapter"),
        ),
        adapter_settings=normalized_adapter_settings(
            _stored_mapping(raw, "adapter_settings", context)
        ),
        trials=_stored_int(raw, "trials", context),
        cost_controls=controls,
        limits=RunLimits(
            max_turns=_stored_int(limits, "max_turns", f"{context}: limits"),
            max_messages=_stored_int(limits, "max_messages", f"{context}: limits"),
            episode_timeout_seconds=_positive_number(
                limits["episode_timeout_seconds"], f"{context}: episode_timeout_seconds"
            ),
        ),
        episode_plan=tuple(plan),
        package_version=_stored_text(contracts, "package", f"{context}: contracts"),
        runner_contract_version=_stored_text(
            contracts, "runner", f"{context}: contracts"
        ),
        evaluator_contract_version=_stored_text(
            contracts, "evaluator", f"{context}: contracts"
        ),
        adapter_contract_version=_stored_text(
            contracts, "adapter", f"{context}: contracts"
        ),
        # Read back from the file rather than substituted from the derivation
        # above: what is rehashed below has to be what the bytes on disk say,
        # or the digest check would be verifying this build's opinion of the
        # manifest instead of the manifest.
        run_track=track,
        run_status=status,
        run_scope=scope,
    )

    # Every stored field bar the two digests and the execution block is in the
    # configuration payload, so this single rehash covers the rest of the file,
    # and the execution block was covered above. What none of it is is
    # tamper-proofing: an editor who can change a field can recompute an unkeyed
    # digest of it just as easily, and claiming otherwise would overstate what a
    # manifest sitting in a writable directory can prove about itself.
    recomputed = configuration_digest(manifest.configuration_payload())
    if recomputed != configuration_id:
        raise RunManifestMismatchError(
            f"{context}: the stored manifest does not hash to its recorded "
            f"configuration_id ({configuration_id} recorded, {recomputed} computed). "
            "Its contents have been edited since the run was created; this run "
            "cannot be resumed."
        )
    return manifest


@dataclass(frozen=True)
class RunSession:
    """An open run: safe paths, an exclusive lock, and a validated manifest.

    Holding one of these is the *permission* to read or write a run's records.
    Everything from opening the manifest to the final ledger validation happens
    inside it, so two writers can never interleave a read of the completed set
    with an append to it.
    """

    paths: RunPaths
    manifest: RunManifest
    resumed: bool
    _descriptor: int

    def _release(self) -> None:
        """Close the descriptor and mark this handle spent. Called once, on exit.

        Closing alone is not enough. The descriptor number stays in the frozen
        field after the ``flock`` is gone, so a handle kept past the ``with``
        block still looked held — and the kernel may hand the same number to an
        unrelated file, which would make the check worse than useless. The field
        is therefore invalidated *before* the close, so no window exists in which
        the handle reads as valid and the lock is not.
        """
        descriptor = self._descriptor
        object.__setattr__(self, "_descriptor", -1)
        if descriptor >= 0:
            os.close(descriptor)

    def assert_held(self) -> None:
        """Prove the lock is still ours before touching the run's records."""
        if self._descriptor < 0:
            raise RunLockError(
                f"the run lock for {self.paths.root} has been released; a run's "
                "records are only read or written while its lock is held"
            )


@contextmanager
def open_run_session(
    output_dir: str | Path, manifest: RunManifest
) -> Iterator[RunSession]:
    """Create or re-open a run directory under an exclusive lock.

    A directory that already holds a *different* run is an error rather than a
    fresh start: silently starting over would discard evidence.

    The lock is an advisory ``flock`` on a dedicated file. It is deliberately
    not a sentinel the writer creates and deletes: a sentinel outlives a crashed
    process and has to be cleaned up by hand, whereas the kernel drops an
    ``flock`` when the holder's descriptors close. This is a Linux/POSIX
    mechanism and is the supported and tested configuration for this build; a
    Windows port would need an equivalent, which this build does not claim.
    """
    # Before a directory, a lock or a manifest: a caller's manifest is checked
    # against itself first, so a run whose identity does not describe its own
    # inputs never reaches the point of creating anything.
    verify_manifest_identity(manifest)
    paths = resolve_run_paths(output_dir)
    try:
        paths.root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RunIOError(f"cannot create run directory {paths.root}: {exc}") from exc
    # Re-checked after creation: the directory may have been swapped for a link
    # between the first check and the mkdir.
    check_run_path_safety(paths.root)

    try:
        descriptor = os.open(
            paths.lock_path,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
    except OSError as exc:
        raise RunLockError(f"cannot open the run lock {paths.lock_path}: {exc}") from exc
    session: RunSession | None = None
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RunLockError(
                f"{paths.root} is already running: another writer holds its lock. "
                "A run has exactly one writer, so this invocation refuses rather "
                "than appending alongside it."
            ) from exc

        if paths.manifest_path.exists():
            stored = load_run_manifest(paths.manifest_path)
            # Canonical equality over the *configuration*, not over the whole
            # file. Every configuration field is inside ``configuration_id``, so
            # two self-consistent configurations with one id are the same plan —
            # and demanding the payloads match outright means the resume never
            # rests on that argument holding. The execution fields are excluded
            # by construction: this invocation minted a candidate execution
            # before it could know a directory existed, and comparing that
            # against the stored one would make every resume impossible.
            if canonical_json(stored.configuration_payload()) != canonical_json(
                manifest.configuration_payload()
            ):
                raise RunManifestMismatchError(
                    f"{paths.root} already holds configuration {stored.configuration_id}"
                    f", but the requested configuration is {manifest.configuration_id}. "
                    "Use a different output directory, or repeat the exact request "
                    "that created this one."
                )
            # The stored execution wins, whole. A run directory holds exactly one
            # execution: its id and its creation time are what every row already
            # appended asserts, so this invocation joins that execution rather
            # than appending a second one's rows into the same ledger.
            active = stored
            resumed = True
        else:
            write_run_manifest(paths, manifest)
            active = manifest
            resumed = False
        session = RunSession(
            paths=paths,
            manifest=active,
            resumed=resumed,
            _descriptor=descriptor,
        )
        yield session
    finally:
        if session is None:
            os.close(descriptor)
        else:
            # Releasing through the session is what invalidates the handle, so a
            # reference kept past this block cannot be used to read or append.
            session._release()


__all__ = [
    "BUILD_CONTRACT_VERSIONS",
    "LEDGER_FILENAME",
    "LOCK_FILENAME",
    "MANIFEST_FILENAME",
    "RETRY_POLICY",
    "RUNNER_CONTRACT_VERSION",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "RUN_PLAN_NOTE",
    "RUN_SCOPE_PROVIDER_EXECUTION",
    "RUN_SCOPE_SYNTHETIC_FAKE",
    "RUN_STATUS_PROVIDER_EXECUTION",
    "RUN_STATUS_SYNTHETIC_FAKE",
    "RUN_TRACKS",
    "RUN_TRACK_PROVIDER_EXECUTION",
    "RUN_TRACK_SYNTHETIC_FAKE",
    "EpisodePlanEntry",
    "ExecutionIdFactory",
    "RunIOError",
    "RunIdentityScope",
    "RunLimits",
    "RunLockError",
    "RunManifest",
    "RunManifestError",
    "RunManifestFormatError",
    "RunManifestMismatchError",
    "RunPathError",
    "RunPaths",
    "RunSecretError",
    "RunSession",
    "WallClock",
    "build_run_manifest",
    "canonical_json",
    "check_episode_limit",
    "check_execution_id",
    "check_run_path_safety",
    "check_utc_timestamp",
    "configuration_digest",
    "execution_binding_payload",
    "execution_digest_of",
    "format_utc_timestamp",
    "load_run_manifest",
    "new_execution_id",
    "normalized_adapter_settings",
    "open_run_session",
    "resolve_run_paths",
    "run_identity_digest",
    "run_track_for_provider",
    "stored_cost_controls",
    "utc_now",
    "verify_manifest_identity",
    "write_run_manifest",
]
