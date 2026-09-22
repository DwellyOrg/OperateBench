"""The one-cell Lifecycle execution-integrity canary.

Run directly or as a module::

    uv run python tools/run_lifecycle_v1_canary.py --offline-preflight \\
        --output-dir DIR --cost-cap-usd 0.52 \\
        --input-usd-per-mtok 0.20 --output-usd-per-mtok 1.20

    uv run python -m tools.run_lifecycle_v1_canary --offline-preflight \\
        --output-dir DIR --cost-cap-usd 0.52 \\
        --input-usd-per-mtok 0.20 --output-usd-per-mtok 1.20

**This is not a provider CLI.** It runs exactly one cell — ``openai`` /
Responses / ``gpt-5.6-luna`` / scenario ``V1`` / one episode — and there is no
argument that selects a provider, an API, a model, a scenario, an agent or a
number of episodes, because those are not command-line decisions. Nothing here
loops over cells; the word "batch" does not describe anything this file does.

**Two modes, and one of them is the default.** ``--offline-preflight`` builds a
real ``openai.OpenAI`` client over an in-process ``httpx.MockTransport``: the
request is serialised by the real SDK, the answer is parsed into real
``openai.types.responses`` models, and the answers are the shipped reference
agent's own decisions. It reads no credential and refuses to run beside one.

``--live`` is possible only when **every** one of these holds, checked in this
order and before anything is constructed:

1. the exact authorisation ``OPERATEBENCH_LIVE_CELL_AUTHORIZATION`` names *this*
   cell and its call envelope — provider, model, scenario, episode count and
   fifty-call cap — and nothing else;
2. no endpoint or tenancy override is in the environment;
3. this build's fixed controls are the pinned ones — fifty calls, one attempt
   per call, no SDK retries, the fixed token ceiling, the thirty-second turn
   deadline and the forty-five-minute wall clock;
4. the operator states this canary's exact fixed envelope: USD 0.20 input and
   USD 1.20 output per million tokens, with an exact USD 0.52 cost cap. Those
   rates retain operator provenance; this build does not claim they are any
   vendor's current prices;
5. the output directory already exists and is mode 0700;
6. the pessimistic worst case fits inside that fixed cap; and
7. every output name is free and reservable, after which ``OPENAI_API_KEY`` is
   read exactly once. Credential presence is not inspected before the envelope
   and output checks pass.

A conjunction rather than a switch: any one of them absent and the run is
refused *before* a client exists, which is what makes "unauthorised paths
construct nothing" a checkable property rather than an intention. Every one of
them is exercised by ``tests/test_lifecycle_v1_canary_gate.py``, and no test in
this repository makes a live call.

**The credential is borrowed, not held.** It is read once, after every check
above has passed, and both it and the authorisation are removed from the process
environment immediately — before the client is built, not after the run. The
``finally`` block drops the local reference and removes them again, so a child
process spawned mid-run, a crash handler that dumps the environment, and a
library that re-reads ``os.environ`` all see an environment with no credential
in it. Relying on the process exiting would be relying on the one thing a
failure mode is most likely to take away. The key is never logged, never
written and never put in a summary; no header is read anywhere in this file.

**The output is owned before it is earned.** Both names a run may write —
the ledger and the episode artefact — are checked free and reserved against the
output directory's own descriptor before a client exists, so an occupied name, a
symlink standing at one, or a second run already holding one is a refusal at
zero dispatch rather than a discovery made after the calls have been paid for.
The reservation is a hidden sentinel that says what it is not; it is never the
output file created early, because a run killed mid-flight must leave real
evidence or nothing, never a plausible-looking artefact. The reservations are
released on every path out, and the final writers still create their own files
exclusively, at 0600, following no link.

**A failure says which check failed, and nothing a provider wrote.** The report
carries a fixed class and code from this build's own vocabulary and the fixed
sentence that code means. An exception raised by a client, an SDK or a provider
is never rendered: it may quote a request header, a response body or the
credential it was rejected for. It stays in memory for whoever is holding a
debugger, and is never printed and never filed. A scanner over everything this
tool prints and everything it writes is the second line rather than the first.

On success the tool writes an artefact 8, the execution ledger that witnessed
it, an audited bundle over the pair, and a zero-provider playback. On any
provider fault, cap refusal or timeout the ledger is kept, **no episode artefact
is written**, and the exit status is non-zero.

The result is an execution-integrity canary at n=1 over a synthetic fixture. It
is not a model score and it is not a leaderboard entry, and the summary says so
in those words.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import openai

from operatebench.agents.evidence import (
    EvidenceRecordingModelAgent,
    EvidenceRunIdentity,
    ProviderEvidenceRecorder,
    WireCaptureTransport,
    provider_identity_for,
)
from operatebench.agents.model import MAX_OUTPUT_TOKENS, MODEL_PROTOCOL_VERSION
from operatebench.agents.openai_responses import (
    LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION,
    LIFECYCLE_RETRY_POLICY,
    OpenAIResponsesTransport,
)
from operatebench.agents.pricing import (
    RATE_SOURCE_OPERATOR,
    LifecycleCostGuard,
    LifecyclePricingPolicy,
)
from operatebench.agents.transport import ProviderFailure
from operatebench.artifact import (
    ARTIFACT_VERSION,
    read_artifact,
    replay_artifact,
    write_artifact,
)
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.execution_bundle import audit_execution_bundle_files
from operatebench.execution_ledger import (
    COST_CAP_POLICY_REFUSE,
    EXECUTION_LEDGER_VERSION,
    TERMINAL_SCORED,
    LedgerControls,
    read_execution_ledger,
)
from operatebench.providers.cost import usd_text
from operatebench.providers.openai_responses import (
    GPT_5_6_LUNA_MODEL,
    GPT_5_6_LUNA_PROFILE,
    OPENAI_BASE_URL,
    PARALLEL_TOOL_CALLS,
    REJECTED_VARIABLES,
    STORE,
    TOOL_CHOICE,
    check_client_endpoint,
    check_request_profile,
)
from operatebench.runner import run_episode
from operatebench.version import OPERATEBENCH_VERSION

if TYPE_CHECKING or __package__:
    from tools.preflight_lifecycle_provider_evidence import ScriptedProvider
else:
    # Direct operator invocation places tools/, not the repository root, on
    # sys.path, so the sibling module is importable by its local name.
    from preflight_lifecycle_provider_evidence import ScriptedProvider

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The fixture this canary runs. The shipped example, by relative path.
FIXTURE = REPO_ROOT / "examples" / "operatebench" / "maintenance_v0_1.yaml"

# ---------------------------------------------------------------- the one cell

#: The whole cell, pinned. Every one of these is a constant rather than an
#: argument default: a default is a decision somebody can change at a shell
#: prompt, and which model this build is authorised to spend money against is
#: not that kind of decision.
CANARY_PROVIDER = "openai"
CANARY_API = "responses"
CANARY_MODEL = GPT_5_6_LUNA_MODEL
CANARY_PROFILE_ID = GPT_5_6_LUNA_PROFILE.profile_id
CANARY_SCENARIO_ID = "V1"
CANARY_AGENT_ID = "openai-gpt-5-6-luna"
CANARY_EPISODES = 1
CANARY_OPERATION_VERSION = "0.6.0"

#: What the output is, said in the words it has to be reported in.
CANARY_CLAIM = (
    "n=1 synthetic execution-integrity canary: one scenario, one model, one "
    "episode, over an authored fixture. It is evidence that this build's "
    "provider execution was recorded, bound and verifiable — it is not a model "
    "score, not a capability measurement and not a leaderboard entry."
)

# ------------------------------------------------------------- the live gate

#: The variable that authorises a live call, and the only value that does.
#: Pinned to the cell and envelope — provider, model, scenario, episode count and
#: provider-call cap — so an authorisation granted for this run cannot be reused
#: for another one or for a build with a different spending envelope.
LIVE_AUTHORIZATION_VARIABLE = "OPERATEBENCH_LIVE_CELL_AUTHORIZATION"
LIVE_AUTHORIZATION_VALUE = (
    f"{CANARY_PROVIDER}:{CANARY_MODEL}:{CANARY_SCENARIO_ID}:{CANARY_EPISODES}:"
    "max-calls-50"
)

#: The variable a real credential is read from, and nowhere else.
CREDENTIAL_VARIABLE = "OPENAI_API_KEY"

#: The placeholder handed to the SDK offline. Not a credential, not read from
#: anywhere, and it never leaves this process.
PLACEHOLDER_KEY = "canary-mock-transport-not-a-credential"

# ----------------------------------------------------------------- the bounds

MAX_PROVIDER_CALLS = 50
EXPECTED_PROVIDER_CALLS = 46
MAX_ATTEMPTS_PER_CALL = 1
SDK_MAX_RETRIES = 0

#: The exact pricing envelope this canary is authorised to exercise. These are
#: build-owned canary controls, not claims about any vendor's current prices.
FIXED_CANARY_POLICY_ID = "operator_pinned_v1"
FIXED_CANARY_RATE_SOURCE = RATE_SOURCE_OPERATOR
FIXED_CANARY_INPUT_USD_PER_MTOK = Decimal("0.20")
FIXED_CANARY_OUTPUT_USD_PER_MTOK = Decimal("1.20")
FIXED_CANARY_COST_CAP_USD = Decimal("0.52")

#: The largest request this scaffold has been measured to produce, and the
#: absolute worst-case token exposure computed from it. Fixed by this build:
#: there is no flag that moves either, and moving one is a code change.
LARGEST_REQUEST_TOKEN_BOUND = 27_241
TOKEN_HARD_CAP = MAX_PROVIDER_CALLS * (LARGEST_REQUEST_TOKEN_BOUND + MAX_OUTPUT_TOKENS)

TURN_DEADLINE_SECONDS = 30.0
WALL_CLOCK_DEADLINE_SECONDS = 2700.0

LEDGER_NAME = "execution_ledger.ndjson"
ARTIFACT_NAME = "episode_artifact.json"

#: Every name in the output directory this run may create, on any path it can
#: take. A successful run writes all three; a faulted or abandoned one writes the
#: ledger alone; neither writes anything else. The bundle audit and the summary
#: are computed and printed, never filed; the third name is a non-scored sidecar,
#: and ``TestEveryOutputNameIsReservedBeforeAnythingIsDispatched`` holds this
#: tuple to exactly what a passing run leaves behind.
OUTPUT_NAMES = (LEDGER_NAME, ARTIFACT_NAME, "execution_ledger.partial.ndjson")

#: What a reservation is called. Not a name any reader could take for evidence:
#: hidden, suffixed with what it is not, and carrying neither of the extensions
#: this build's two documents use. A reservation is never the output file
#: created early — an artefact that exists before the run that justifies it is
#: exactly the misleading object a crash must not leave behind.
RESERVATION_SUFFIX = ".canary-reservation-not-evidence"

RESERVATION_BYTES = (
    b"operatebench one-cell canary: an output name reserved by a run in "
    b"progress.\nThis file is not evidence. It is removed when that run ends; "
    b"one left\nbehind means a run was killed, and the next run refuses rather "
    b"than\nguessing which.\n"
)

MODE_OFFLINE = "offline_preflight"
MODE_LIVE = "live"

# ------------------------------------------------- what a failure may say

#: A failure report names a class and a code, and the code's meaning comes out
#: of the fixed table below. Nothing else is rendered: an exception raised by an
#: SDK, a client or a provider is prose written elsewhere, and it may quote a
#: request header, a response body or the credential it was rejected for.
FAILURE_CLASS_REFUSAL = "refusal"
FAILURE_CLASS_CELL = "cell_failure"
FAILURE_CLASS_INTERNAL = "internal_error"

CODE_UNCLASSIFIED = "unclassified"
CODE_UNEXPECTED_INTERNAL_ERROR = "unexpected_internal_error"
CODE_OUTPUT_WITHHELD_BY_SECRET_SCANNER = "output_withheld_by_secret_scanner"
CODE_FIXED_CANARY_PRICING_ENVELOPE_REQUIRED = "fixed_canary_pricing_envelope_required"

#: The whole vocabulary, and the fixed sentence each term means. Every string
#: here is written in this file and holds no runtime value — no path, no
#: environment value, no exception text, no number this build did not pin.
FAILURE_DETAILS: Mapping[str, str] = {
    CODE_UNCLASSIFIED: (
        "this run failed a check that reported no code. Treat it as a refusal "
        "and read the exit status, not this report"
    ),
    CODE_UNEXPECTED_INTERNAL_ERROR: (
        "this build raised an error it does not classify. Nothing this run was "
        "told is reported here, because an unclassified error's text is written "
        "by whatever raised it"
    ),
    CODE_OUTPUT_WITHHELD_BY_SECRET_SCANNER: (
        "the report this run was about to print matched a credential shape, so "
        "it was withheld in full rather than printed with the match removed"
    ),
    # Refusals: the run never dispatched.
    "offline_run_beside_live_variables": (
        "the offline preflight refuses to run beside environment variables that "
        "name a credential, an authorisation or an endpoint override"
    ),
    "live_authorization_absent_or_wrong": (
        "a live run requires the pinned per-cell authorisation variable to state "
        "exactly this provider, model, scenario, episode count and provider-call cap"
    ),
    "endpoint_or_tenancy_override_present": (
        "the environment states an endpoint or tenancy override, which changes "
        "where a request goes or which account it is billed to"
    ),
    "credential_absent": (
        "a live run reads its credential from this build's one named variable "
        "and the environment states none"
    ),
    CODE_FIXED_CANARY_PRICING_ENVELOPE_REQUIRED: (
        "this canary accepts only its build-owned fixed policy identity, rate "
        "source, input rate, output rate and cost cap. The rates record operator "
        "provenance and are not a claim about any vendor's current prices"
    ),
    "pinned_exact_cell_envelope_changed": (
        "this build's provider, API, model, request profile, scenario, agent, "
        "workload expectation or interface identity is not the exact authorised cell"
    ),
    "pinned_call_bound_changed": (
        "this build's fixed provider-call bound is not the one this canary is "
        "authorised for"
    ),
    "pinned_attempt_policy_changed": (
        "this build's retry policy is not one attempt per call with no hidden retry"
    ),
    "pinned_sdk_retries_changed": (
        "the SDK's own retry loop is not off, so a turn's evidence would "
        "describe an unknown number of requests"
    ),
    "pinned_token_ceiling_changed": (
        "this build's fixed token ceiling is not the one this canary is authorised for"
    ),
    "pinned_turn_deadline_changed": (
        "this build's fixed per-turn deadline is not the one this canary is "
        "authorised for, and the ledger would record a control nobody agreed to"
    ),
    "pinned_wall_clock_deadline_changed": (
        "this build's fixed wall-clock deadline is not the one this canary is "
        "authorised for, and the ledger would record a control nobody agreed to"
    ),
    "sdk_client_retries_changed": (
        "the SDK client was constructed with a retry count this build does not own"
    ),
    "output_directory_absent": (
        "the output directory does not exist. It is named by the operator and is "
        "not created here"
    ),
    "output_directory_not_a_directory": ("the output location is not a directory"),
    "output_directory_not_owned": (
        "the output directory is owned by another user; this build writes "
        "evidence only where the running user owns the directory"
    ),
    "output_directory_mode_not_0700": (
        "the output directory is not mode 0700. A ledger names the runs a "
        "machine made and what they cost, and is readable by its owner alone"
    ),
    "output_directory_unopenable": (
        "the output directory could not be opened as a directory, without "
        "following a link, at the moment the reservation was taken"
    ),
    "output_target_exists": (
        "a name this run would write already exists in the output directory. "
        "Evidence is written once, and this run refuses before it dispatches "
        "rather than after it has spent"
    ),
    "output_target_unreadable": (
        "a name this run would write could not be examined in the output directory"
    ),
    "output_reserved_by_another_run": (
        "a name this run would write is reserved by another run, or by a run "
        "that was killed while holding it"
    ),
    "output_not_reservable": (
        "a name this run would write could not be reserved exclusively in the "
        "output directory"
    ),
    "worst_case_exceeds_cap": (
        "the pessimistic worst case at the stated rates exceeds the stated cost "
        "cap. The run is refused before anything is dispatched"
    ),
    # Cell failures: the run dispatched and produced no evidence a pass needs.
    "provider_boundary_fault": (
        "the cell stopped at the provider boundary. Its execution ledger is "
        "retained and no episode artefact is written: a run that stopped there "
        "produced no business outcome, so there is nothing to score"
    ),
    "wall_clock_deadline_exceeded": (
        "the cell passed its wall-clock bound and was abandoned before its next "
        "turn. Its execution ledger is retained and no episode artefact is "
        "written"
    ),
    "ledger_not_scored": ("the execution ledger does not read as a scored terminal"),
    "ledger_recorded_faults": (
        "the cell recorded provider fault(s); a passing canary records none"
    ),
    "call_bound_exceeded": (
        "the cell made more provider calls than this build's bound allows"
    ),
    "artifact_version_mismatch": (
        "the episode artefact is not the version this build writes"
    ),
    "artifact_did_not_replay": (
        "the episode artefact did not replay against its own specification"
    ),
    "replay_reached_provider": (
        "the replay reached the provider; a replay carries the binding forward "
        "and dispatches nothing"
    ),
    "evidence_failed_secret_scan": (
        "a file this run wrote matched a credential shape. The run is reported "
        "as failed and the match itself is not printed"
    ),
}


class CanaryRefusal(Exception):
    """A condition this canary requires was not met.

    The *code* is what an operator is shown; the detail is for a developer
    reading a traceback in memory, and never reaches a report or a file. Both
    exist because "which check refused" and "which value it saw" are different
    facts with different audiences, and only the first one is safe to render.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


class CanaryFailure(Exception):
    """The cell ran and did not produce the evidence a pass requires."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


class CanaryDeadlineExceeded(CanaryFailure):
    """The cell passed its wall-clock bound and was abandoned mid-flight."""

    def __init__(self, detail: str = "") -> None:
        super().__init__("wall_clock_deadline_exceeded", detail)


# ------------------------------------------------------------ the secret scan

#: Shapes that mean "this is a credential, or the header one travels in".
#: Defence in depth rather than the control: the control is that nothing but
#: this build's own fixed strings is ever rendered. A scanner is what catches
#: the case where that stops being true.
#:
#: Deliberately shape-based and narrow. A rule broad enough to catch every
#: opaque string would catch this build's own content digests, and a scanner
#: that fires on evidence is one somebody turns off.
SECRET_SHAPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("credential_prefix", re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}")),
    ("api_key_assignment", re.compile(r"(?i)\bx?-?api[_\-]?key\b\s*[:=]\s*\S")),
    ("authorization_assignment", re.compile(r"(?i)\bauthorization\b\s*[:=]\s*\S")),
)


def scan_for_secrets(text: str) -> list[str]:
    """Name every credential shape ``text`` matches. Empty means none."""
    return [name for name, shape in SECRET_SHAPES if shape.search(text)]


class DeadlinedModelAgent(EvidenceRecordingModelAgent):
    """The recording agent, refusing to begin a turn past the run's wall clock.

    Checked *before* each turn rather than after the run, because a bound tested
    only at the end is a report rather than a control. The per-turn deadline and
    the call bound already multiply to exactly this figure, so in practice this
    is the second of two independent enforcements — which is the point: a bound
    with one enforcement point stops being one the moment that point is bypassed.

    Raising here unwinds through :func:`~operatebench.runner.run_episode`, which
    finalises the journal as ``aborted`` on the way out. The evidence of what was
    spent survives; the episode artefact is never written, because a run that was
    abandoned reached no business terminal.
    """

    def __init__(self, *args: Any, deadline_at: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._deadline_at = deadline_at

    def decide(self, observation: Any) -> Any:
        if time.monotonic() >= self._deadline_at:
            raise CanaryDeadlineExceeded(
                f"passed the wall-clock bound of {WALL_CLOCK_DEADLINE_SECONDS:.0f}s"
            )
        return super().decide(observation)


def wall_clock_deadline_at(started_at: float) -> float:
    """The instant this run's wall clock expires.

    A seam, not a setting. The bound itself is one of this build's pinned
    controls and :func:`check_pinned_controls` refuses a build that moved it, so
    a test that needs to observe an *overrun* moves the deadline instant rather
    than the bound — the bound has to stay unmovable for the refusal above it to
    mean anything.
    """
    return started_at + WALL_CLOCK_DEADLINE_SECONDS


# ------------------------------------------------------------------ the gates


def check_pinned_controls() -> None:
    """Refuse if this build's own fixed controls are not the pinned ones.

    A self-check rather than an argument check. If a later edit raises the call
    bound, turns a retry back on, moves the token ceiling or moves either
    deadline, this run is no longer the run the authorisation was granted for,
    and it stops here rather than spending against a control nobody agreed to.

    Both deadlines are checked, and independently. They are not comfort
    settings: the per-turn deadline and the call bound are what make fifty calls
    a bound on time as well as on count, the wall clock is the second,
    independent enforcement of the same thing, and both are written into the
    ledger as the controls this run was made under. A build whose ledger would
    record a figure nobody authorised is refused here — before a transport, a
    client or a request exists.
    """
    # Independent literals are the authorization oracle. In particular, never
    # derive this side from the fixed constants: a coherent source edit must not
    # be able to move both the run and the check that authorizes it.
    if (
        CANARY_PROVIDER != "openai"
        or CANARY_API != "responses"
        or CANARY_MODEL != "gpt-5.6-luna"
        or CANARY_PROFILE_ID != "gpt56luna_reasoning_none_sampling_omitted_v2"
        or CANARY_SCENARIO_ID != "V1"
        or CANARY_AGENT_ID != "openai-gpt-5-6-luna"
        or CANARY_EPISODES != 1
        or EXPECTED_PROVIDER_CALLS != 46
        or LIVE_AUTHORIZATION_VALUE != "openai:gpt-5.6-luna:V1:1:max-calls-50"
        or GPT_5_6_LUNA_PROFILE.profile_id
        != "gpt56luna_reasoning_none_sampling_omitted_v2"
        or GPT_5_6_LUNA_PROFILE.reasoning != {"effort": "none"}
        or GPT_5_6_LUNA_PROFILE.temperature is not None
        or GPT_5_6_LUNA_PROFILE.omitted_fields() != ("temperature", "top_p")
        or STORE is not False
        or PARALLEL_TOOL_CALLS is not False
        or TOOL_CHOICE != "required"
        or OPERATEBENCH_VERSION != "0.12.0"
        or CANARY_OPERATION_VERSION != "0.6.0"
        or MODEL_PROTOCOL_VERSION != "operatebench.model.v4"
        or LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION
        != "lifecycle_openai_responses_model_request_v9"
        or ARTIFACT_VERSION != 8
        or EXECUTION_LEDGER_VERSION != 3
    ):
        raise CanaryRefusal("pinned_exact_cell_envelope_changed")
    if (
        FIXED_CANARY_POLICY_ID != "operator_pinned_v1"
        or RATE_SOURCE_OPERATOR != "operator_supplied_pinned_rates"
        or FIXED_CANARY_RATE_SOURCE != "operator_supplied_pinned_rates"
        or Decimal("0.20") != FIXED_CANARY_INPUT_USD_PER_MTOK
        or Decimal("1.20") != FIXED_CANARY_OUTPUT_USD_PER_MTOK
        or Decimal("0.52") != FIXED_CANARY_COST_CAP_USD
    ):
        raise CanaryRefusal(CODE_FIXED_CANARY_PRICING_ENVELOPE_REQUIRED)
    if MAX_PROVIDER_CALLS != 50:
        raise CanaryRefusal(
            "pinned_call_bound_changed", f"authorised for 50, states {MAX_PROVIDER_CALLS}"
        )
    if MAX_ATTEMPTS_PER_CALL != 1 or LIFECYCLE_RETRY_POLICY.max_attempts != 1:
        raise CanaryRefusal(
            "pinned_attempt_policy_changed",
            f"policy allows {LIFECYCLE_RETRY_POLICY.max_attempts}, tool states "
            f"{MAX_ATTEMPTS_PER_CALL}",
        )
    if SDK_MAX_RETRIES != 0:
        raise CanaryRefusal("pinned_sdk_retries_changed", f"states {SDK_MAX_RETRIES}")
    if (
        LARGEST_REQUEST_TOKEN_BOUND != 27_241
        or MAX_OUTPUT_TOKENS != 4_096
        or TOKEN_HARD_CAP != 1_566_850
    ):
        raise CanaryRefusal(
            "pinned_token_ceiling_changed",
            "authorised for request bound 27241, output bound 4096 and token "
            f"hard cap 1566850; states {LARGEST_REQUEST_TOKEN_BOUND}, "
            f"{MAX_OUTPUT_TOKENS} and {TOKEN_HARD_CAP}",
        )

    if TURN_DEADLINE_SECONDS != 30.0:
        raise CanaryRefusal(
            "pinned_turn_deadline_changed",
            f"states {TURN_DEADLINE_SECONDS}, not 30.0",
        )
    if WALL_CLOCK_DEADLINE_SECONDS != 2700.0:
        raise CanaryRefusal(
            "pinned_wall_clock_deadline_changed",
            f"states {WALL_CLOCK_DEADLINE_SECONDS}, not 2700.0",
        )


def check_fixed_canary_pricing_envelope(
    policy: LifecyclePricingPolicy, *, cap: Decimal
) -> None:
    """Refuse anything outside this build's exact canary pricing envelope.

    This is an authorisation check, not a claim that the fixed rates are a
    vendor's current prices. It deliberately compares every independent term
    before a credential is read, a client or transport is built, an output name
    is reserved, or a request can be dispatched.
    """
    if (
        type(policy) is not LifecyclePricingPolicy
        or policy.policy_id != "operator_pinned_v1"
        or policy.rate_source != "operator_supplied_pinned_rates"
        or policy.input_usd_per_mtok != Decimal("0.20")
        or policy.output_usd_per_mtok != Decimal("1.20")
        or type(cap) is not Decimal
        or cap != Decimal("0.52")
    ):
        raise CanaryRefusal(CODE_FIXED_CANARY_PRICING_ENVELOPE_REQUIRED)


def check_spec_identity(spec: Any) -> None:
    """Refuse a specification outside the pinned operation interface."""
    spec.verify_identity()
    if spec.operation_version != "0.6.0":
        raise CanaryRefusal("pinned_exact_cell_envelope_changed")


def check_live_authorization(environ: Mapping[str, str]) -> None:
    """Refuse a live run that nothing in the environment authorises.

    Order matters. This checks only the authorisation and endpoint/tenancy
    overrides. The credential is not looked for here: the fixed pricing envelope
    and output ownership must pass before even credential presence is read.
    """
    stated = environ.get(LIVE_AUTHORIZATION_VARIABLE)
    if stated != LIVE_AUTHORIZATION_VALUE:
        raise CanaryRefusal(
            "live_authorization_absent_or_wrong",
            f"{LIVE_AUTHORIZATION_VARIABLE} does not state {LIVE_AUTHORIZATION_VALUE!r}",
        )
    present = sorted(name for name in REJECTED_VARIABLES if environ.get(name))
    if present:
        raise CanaryRefusal("endpoint_or_tenancy_override_present", f"{present}")


def refuse_live_environment(environ: Mapping[str, str]) -> list[str]:
    """Refuse an *offline* run standing beside anything that points at a service."""
    named = [CREDENTIAL_VARIABLE, LIVE_AUTHORIZATION_VARIABLE, *REJECTED_VARIABLES]
    present = sorted(name for name in named if environ.get(name))
    if present:
        raise CanaryRefusal("offline_run_beside_live_variables", f"{present}")
    return named


def check_output_directory(directory: Path) -> Path:
    """Refuse an output location this build will not write evidence to.

    Not created here. Evidence is written where somebody said, or nowhere: a
    tool that made its own directory would decide where a machine's record of
    what it spent lives, and nobody asked it to.
    """
    try:
        info = directory.lstat()
    except OSError as exc:
        raise CanaryRefusal(
            "output_directory_absent", f"{directory} ({type(exc).__name__})"
        ) from exc
    if not stat.S_ISDIR(info.st_mode):
        raise CanaryRefusal("output_directory_not_a_directory", str(directory))
    if info.st_uid != os.geteuid():
        raise CanaryRefusal("output_directory_not_owned", str(directory))
    mode = stat.S_IMODE(info.st_mode)
    if mode != 0o700:
        raise CanaryRefusal("output_directory_mode_not_0700", f"{directory} {mode:04o}")
    return directory


# ------------------------------------------------------ reserving the output


def reservation_name_for(name: str) -> str:
    """What this run's claim on an output name is called while it holds it."""
    return f".{name}{RESERVATION_SUFFIX}"


def _open_output_directory(directory: Path) -> int:
    """Hold the directory by descriptor, and re-check owner and mode off it.

    Everything the reservation does afterwards is done against this descriptor
    rather than by path. A path checked and then used by name is a path anything
    that can rename a component gets to choose the second time; a descriptor
    names one inode for as long as it is held.
    """
    try:
        fd = os.open(
            directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
    except OSError as exc:
        raise CanaryRefusal(
            "output_directory_unopenable", f"{directory} ({type(exc).__name__})"
        ) from exc
    try:
        info = os.fstat(fd)
        if info.st_uid != os.geteuid():
            raise CanaryRefusal("output_directory_not_owned", str(directory))
        if stat.S_IMODE(info.st_mode) != 0o700:
            raise CanaryRefusal(
                "output_directory_mode_not_0700",
                f"{directory} {stat.S_IMODE(info.st_mode):04o}",
            )
    except BaseException:
        os.close(fd)
        raise
    return fd


def _check_target_free(dirfd: int, name: str) -> None:
    """Refuse if anything at all already stands at an output name.

    ``lstat`` rather than ``exists``: a symbolic link, dangling or not, is
    something standing at the name, and the point is to refuse it here — at zero
    dispatch — rather than have the writer's ``O_EXCL`` refuse it after a run
    has been paid for.
    """
    try:
        os.lstat(name, dir_fd=dirfd)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise CanaryRefusal(
            "output_target_unreadable", f"{name} ({type(exc).__name__})"
        ) from exc
    raise CanaryRefusal("output_target_exists", name)


def _take_reservation(dirfd: int, name: str) -> int:
    """Claim one output name exclusively, and hold the descriptor.

    ``O_CREAT | O_EXCL | O_NOFOLLOW`` against the held directory descriptor, at
    0600. The kernel's exclusive create is the whole of the race: two runs
    started at once, one wins the name and the other refuses before it has
    constructed a client. Creating the *reservation* rather than the output
    itself is what keeps that true without leaving a file a reader could mistake
    for evidence if this process is killed while holding it.
    """
    reservation = reservation_name_for(name)
    try:
        fd = os.open(
            reservation,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=dirfd,
        )
    except FileExistsError as exc:
        raise CanaryRefusal("output_reserved_by_another_run", reservation) from exc
    except OSError as exc:
        raise CanaryRefusal(
            "output_not_reservable", f"{reservation} ({type(exc).__name__})"
        ) from exc
    try:
        os.write(fd, RESERVATION_BYTES)
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        with suppress(OSError):
            os.unlink(reservation, dir_fd=dirfd)
        raise
    return fd


@contextmanager
def reserve_output_names(directory: Path) -> Iterator[None]:
    """Own every name this run may write, before this run builds anything.

    Two passes, in this order. First every output name is checked free, so a
    directory that already holds one of them refuses without having claimed any
    of the others; then every name is reserved. A collision in either pass
    releases what this run took and leaves what it did not — a run that refuses
    does not tidy away another run's claim.

    The descriptors are held for the run's whole lifetime and the reservations
    are removed on the way out, on every path. The final writers still create
    their own files by ``O_CREAT | O_EXCL | O_NOFOLLOW`` at 0600: the
    reservation makes the collision cost nothing, and does not replace the
    exclusive create that makes evidence write-once.
    """
    dirfd = _open_output_directory(directory)
    held: list[tuple[str, int]] = []
    try:
        for name in OUTPUT_NAMES:
            _check_target_free(dirfd, name)
        for name in OUTPUT_NAMES:
            held.append((name, _take_reservation(dirfd, name)))
        yield
    finally:
        for name, fd in held:
            with suppress(OSError):
                os.close(fd)
            with suppress(OSError):
                os.unlink(reservation_name_for(name), dir_fd=dirfd)
        os.close(dirfd)


def check_worst_case(policy: LifecyclePricingPolicy, *, cap: Decimal) -> dict[str, Any]:
    """Refuse the run if its pessimistic worst case does not fit the cap."""
    worst = policy.worst_case_usd(
        calls=MAX_PROVIDER_CALLS,
        input_token_bound=LARGEST_REQUEST_TOKEN_BOUND,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    if worst > cap:
        raise CanaryRefusal(
            "worst_case_exceeds_cap",
            f"worst case {usd_text(worst)} USD over cap {usd_text(cap)} USD",
        )
    return {
        "worst_case_usd": usd_text(worst),
        "cost_cap_usd": usd_text(cap),
        "token_hard_cap": TOKEN_HARD_CAP,
        "pricing_digest_sha256": policy.digest_sha256,
        "rate_source": policy.rate_source,
    }


# ------------------------------------------------------------ the two clients


def open_live_transport() -> httpx.BaseTransport:
    """The one function in this file that would open a socket.

    Separated from :func:`open_live_client` so the two things a gate must
    prevent are separately observable: constructing a client around a
    credential, and being in a position to send. The canary's own test suite
    replaces both with tripwires and asserts neither is reached on any
    unauthorised path.

    ``retries=0`` because this build owns its retry policy at two levels
    already; a third one underneath would make a turn's recorded evidence
    describe an unknown number of requests.
    """
    return httpx.HTTPTransport(retries=0)


def open_live_client(api_key: str, *, transport: httpx.BaseTransport) -> openai.OpenAI:
    """Build the SDK client for a live run, over the transport it is given.

    The transport is a parameter rather than something built in here, and that
    is what makes the wire capture unconditional: the client cannot be
    constructed around a socket this run is not observing. A gate that built the
    client and then declined to use it would already have handed a credential to
    an SDK, which is why every check runs before this is called.
    """
    return openai.OpenAI(
        api_key=api_key,
        base_url=OPENAI_BASE_URL,
        max_retries=SDK_MAX_RETRIES,
        http_client=httpx.Client(transport=transport, timeout=TURN_DEADLINE_SECONDS),
    )


def open_offline_client(handler: Callable[[httpx.Request], httpx.Response]) -> Any:
    """Build the SDK client for the offline preflight. No socket, no credential."""
    return openai.OpenAI(
        api_key=PLACEHOLDER_KEY,
        base_url=OPENAI_BASE_URL,
        max_retries=SDK_MAX_RETRIES,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


# -------------------------------------------------------------------- the cell


def controls_for(policy: LifecyclePricingPolicy, *, cap: Decimal) -> LedgerControls:
    return LedgerControls(
        max_provider_calls=MAX_PROVIDER_CALLS,
        max_attempts_per_call=MAX_ATTEMPTS_PER_CALL,
        expected_provider_calls=EXPECTED_PROVIDER_CALLS,
        token_hard_cap=TOKEN_HARD_CAP,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        cost_cap_usd=usd_text(cap),
        cost_cap_policy=COST_CAP_POLICY_REFUSE,
        turn_deadline_seconds=TURN_DEADLINE_SECONDS,
        wall_clock_deadline_seconds=WALL_CLOCK_DEADLINE_SECONDS,
        pricing=policy.as_dict(),
    )


def _execute_one_cell(
    spec: Any,
    *,
    client: Any,
    directory: Path,
    policy: LifecyclePricingPolicy,
    cap: Decimal,
    wire: WireCaptureTransport,
    started_at: float,
) -> dict[str, Any]:
    """Execute one prepared cell, write its evidence, and audit what it wrote.

    One episode. There is no iteration here and no parameter that would make one.
    As defence in depth, this private implementation core rechecks the pinned
    controls, specification identity/version, directory mode and fixed pricing.
    The supported outer paths own selecting ``FIXTURE``, reserving every output
    name, and constructing the client over the same wire capture passed here for
    evidence recording; this helper is not a supported authorisation boundary
    for arbitrary in-process callers.
    """
    check_pinned_controls()
    check_spec_identity(spec)
    check_output_directory(directory)
    check_fixed_canary_pricing_envelope(policy, cap=cap)
    check_request_profile(GPT_5_6_LUNA_PROFILE, model=CANARY_MODEL)
    check_client_endpoint(client)
    if client.max_retries != SDK_MAX_RETRIES:
        raise CanaryRefusal(
            "sdk_client_retries_changed", f"max_retries={client.max_retries}"
        )
    ledger_path = directory / LEDGER_NAME
    guard = LifecycleCostGuard(
        policy=policy,
        cap_usd=cap,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        token_hard_cap=TOKEN_HARD_CAP,
    )
    transport = OpenAIResponsesTransport(
        model=CANARY_MODEL,
        client=client,
        deadline_seconds=TURN_DEADLINE_SECONDS,
        cost_guard=guard,
    )
    recorder = ProviderEvidenceRecorder(
        path=ledger_path,
        identity=EvidenceRunIdentity(
            operation_id=spec.operation_id,
            spec_digest_sha256=spec.spec_digest_sha256,
            scenario_id=CANARY_SCENARIO_ID,
            agent_id=CANARY_AGENT_ID,
        ),
        provider=provider_identity_for(transport),
        controls=controls_for(policy, cap=cap),
        wire=wire,
        guard=guard,
    )
    transport.attach_recorder(recorder)
    agent = DeadlinedModelAgent(
        transport,
        model=CANARY_MODEL,
        agent_id=CANARY_AGENT_ID,
        recorder=recorder,
        max_transport_calls=MAX_PROVIDER_CALLS,
        deadline_at=wall_clock_deadline_at(started_at),
    )
    try:
        run = run_episode(
            spec,
            CANARY_SCENARIO_ID,
            CANARY_AGENT_ID,
            agent_factory=lambda: agent,
            agent_kind="model",
            evidence_recorder=recorder,
        )
    except ProviderFailure as failure:
        raise CanaryFailure(
            "provider_boundary_fault", f"fault {failure.fault}"
        ) from failure
    elapsed = time.monotonic() - started_at
    return _audit_what_was_written(
        spec, run, directory=directory, ledger_path=ledger_path, elapsed=elapsed
    )


def _audit_what_was_written(
    spec: Any,
    run: Any,
    *,
    directory: Path,
    ledger_path: Path,
    elapsed: float,
) -> dict[str, Any]:
    """Ledger, artefact, bundle, playback — and refuse if any of the four fails."""
    ledger = read_execution_ledger(ledger_path, require_complete=True)
    if ledger.status != TERMINAL_SCORED:
        raise CanaryFailure("ledger_not_scored", f"{ledger.status!r}")
    if ledger.totals.fault_counts:
        raise CanaryFailure("ledger_recorded_faults", f"{ledger.totals.fault_counts}")
    if ledger.totals.provider_calls > MAX_PROVIDER_CALLS:
        raise CanaryFailure(
            "call_bound_exceeded",
            f"{ledger.totals.provider_calls} against {MAX_PROVIDER_CALLS}",
        )

    artifact_path = write_artifact(run, directory / ARTIFACT_NAME)
    record = read_artifact(artifact_path)
    if record["artifact_version"] != ARTIFACT_VERSION:
        raise CanaryFailure(
            "artifact_version_mismatch",
            f"{record['artifact_version']} not {ARTIFACT_VERSION}",
        )
    bundle = audit_execution_bundle_files(artifact_path, ledger_path)

    before = ledger.totals.provider_calls
    report = replay_artifact(spec, record)
    if not report.ok:
        raise CanaryFailure("artifact_did_not_replay", f"{list(report.differences)}")
    after = read_execution_ledger(ledger_path, require_complete=True)
    if after.totals.provider_calls != before:
        raise CanaryFailure("replay_reached_provider")
    _scan_what_was_written(directory)
    return {
        "ledger": ledger_path.name,
        "artifact": artifact_path.name,
        "artifact_version": record["artifact_version"],
        "status": run.outcome.status,
        "reliable": run.reliable,
        "provider_calls": ledger.totals.provider_calls,
        "attempts": ledger.totals.attempts,
        "measured_cost_usd": ledger.totals.measured_cost_usd,
        "forfeited_reservation_usd": ledger.totals.forfeited_reservation_usd,
        "wall_clock_seconds": round(elapsed, 3),
        "bundle_audit": bundle.summary(),
        "replay": {
            "ok": report.ok,
            "reproduction": report.reproduction,
            "carried_forward": list(report.carried_forward),
            "provider_calls": after.totals.provider_calls - before,
        },
        "ledger_audit": ledger.summary(),
    }


def _scan_what_was_written(directory: Path) -> None:
    """Read back everything this run filed and refuse if any of it looks secret.

    Defence in depth over the writers rather than a substitute for them: no
    credential and no provider prose is put into either document by
    construction. This is the check that notices if that ever stops being true,
    and it names the shape it matched rather than quoting the match.
    """
    for name in OUTPUT_NAMES:
        path = directory / name
        if not path.exists():
            continue
        tripped = scan_for_secrets(path.read_text(encoding="utf-8", errors="replace"))
        if tripped:
            raise CanaryFailure("evidence_failed_secret_scan", f"{name}: {tripped}")


# --------------------------------------------------------------- the two modes


def run_offline_preflight(
    *,
    directory: Path,
    policy: LifecyclePricingPolicy,
    cap: Decimal,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    """The default path: the whole cell over a mock, with nothing to leak."""
    refused = refuse_live_environment(environ)
    check_pinned_controls()
    check_output_directory(directory)
    check_fixed_canary_pricing_envelope(policy, cap=cap)
    budget = check_worst_case(policy, cap=cap)
    spec = load_spec(FIXTURE)
    check_spec_identity(spec)

    with reserve_output_names(directory):
        wire = WireCaptureTransport()
        provider = ScriptedProvider(max_calls=MAX_PROVIDER_CALLS)
        wire.attach(httpx.MockTransport(provider))
        client = openai.OpenAI(
            api_key=PLACEHOLDER_KEY,
            base_url=OPENAI_BASE_URL,
            max_retries=SDK_MAX_RETRIES,
            http_client=httpx.Client(transport=wire),
        )
        result = _execute_one_cell(
            spec,
            client=client,
            directory=directory,
            policy=policy,
            cap=cap,
            wire=wire,
            started_at=time.monotonic(),
        )
    return {
        **result,
        "mode": MODE_OFFLINE,
        "live_provider_call": False,
        "credential_read": False,
        "network_access": False,
        "transport": "httpx.MockTransport through the real OpenAI SDK",
        "refused_environment_variables": refused,
        "budget": budget,
    }


def run_live_cell(
    *,
    directory: Path,
    policy: LifecyclePricingPolicy,
    cap: Decimal,
) -> dict[str, Any]:
    """The gated path, wrapped in the cleanup that outlives every branch of it.

    The credential's whole lifetime is inside :func:`_run_live_cell`, and it is
    short: read after the last gate, removed from the environment before the
    client is built, dropped in ``finally``. It is never returned, never stored
    on an object and never rendered.

    The environment is cleared here rather than there so that a gate which
    refuses *before* the credential is read clears it too. A refusal is still an
    attempt, and an authorisation left standing after one is an authorisation
    the next thing this process does can use.
    """
    try:
        return _run_live_cell(directory=directory, policy=policy, cap=cap)
    finally:
        # Every path out of a live attempt, including the gates that refuse
        # before the credential is ever read. An authorisation and a key that
        # outlive the attempt they were granted for are available to whatever
        # this process does next, and "the run refused" is not a reason to leave
        # them lying there.
        with suppress(KeyError):
            del os.environ[CREDENTIAL_VARIABLE]
        with suppress(KeyError):
            del os.environ[LIVE_AUTHORIZATION_VARIABLE]


def _run_live_cell(
    *,
    directory: Path,
    policy: LifecyclePricingPolicy,
    cap: Decimal,
) -> dict[str, Any]:
    """The gated path itself. Every check runs before the credential is read."""
    check_pinned_controls()
    check_live_authorization(os.environ)
    check_output_directory(directory)
    check_fixed_canary_pricing_envelope(policy, cap=cap)
    budget = check_worst_case(policy, cap=cap)
    spec = load_spec(FIXTURE)
    check_spec_identity(spec)

    # Taken before the credential is read: an output name this run cannot own is
    # a reason to refuse, and a refusal that has already read a key has already
    # done the thing the gate exists to defer.
    with reserve_output_names(directory):
        api_key: str | None = None
        try:
            # Read once, and immediately taken out of the environment along with
            # the authorisation that permitted it. Anything this process spawns,
            # dumps or re-reads from here on sees neither.
            api_key = os.environ.pop(CREDENTIAL_VARIABLE, None)
            os.environ.pop(LIVE_AUTHORIZATION_VARIABLE, None)
            if not api_key:
                raise CanaryRefusal(
                    "credential_absent", f"{CREDENTIAL_VARIABLE} is unset"
                )
            # The capture is installed *under* the client rather than swapped in
            # afterwards, so there is no moment at which a client exists whose
            # requests this run is not observing.
            wire = WireCaptureTransport()
            wire.attach(open_live_transport())
            client = open_live_client(api_key, transport=wire)
            api_key = None
            result = _execute_one_cell(
                spec,
                client=client,
                directory=directory,
                policy=policy,
                cap=cap,
                wire=wire,
                started_at=time.monotonic(),
            )
        finally:
            # Both halves, unconditionally, and inside the reservation so that
            # neither cleanup can be skipped by the other failing. The local
            # reference goes first because it is the one an exception traceback
            # could otherwise carry, and the environment is cleared again in
            # case anything below re-set it. Relying on the process exiting
            # would be relying on the one thing a failure is most likely to take
            # away.
            api_key = None
            del api_key
            with suppress(KeyError):
                del os.environ[CREDENTIAL_VARIABLE]
            with suppress(KeyError):
                del os.environ[LIVE_AUTHORIZATION_VARIABLE]
    return {
        **result,
        "mode": MODE_LIVE,
        "live_provider_call": True,
        "credential_read": True,
        "credential_retained": False,
        "budget": budget,
    }


# ----------------------------------------------------------------- the command


def build_parser() -> argparse.ArgumentParser:
    """The whole command line. No cell selector appears here, by construction."""
    parser = argparse.ArgumentParser(
        description=(
            "The one-cell Lifecycle execution-integrity canary: openai / Responses "
            f"/ {CANARY_MODEL} / scenario {CANARY_SCENARIO_ID} / one episode. There "
            "is no argument that selects a provider, a model, a scenario or a "
            "number of episodes."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--offline-preflight",
        action="store_true",
        help=(
            "run the whole cell over an in-process mock transport. The default, "
            "and the only mode that needs no authorisation"
        ),
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help=(
            "attempt the one live cell. Refused unless "
            f"{LIVE_AUTHORIZATION_VARIABLE} states exactly "
            f"{LIVE_AUTHORIZATION_VALUE!r}, {CREDENTIAL_VARIABLE} is present, no "
            "endpoint or tenancy override is set, the operator states the exact "
            "fixed canary envelope (USD 0.20 input and USD 1.20 output per million "
            "tokens with a USD 0.52 cap), and the output directory exists at mode "
            "0700"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "an existing directory, mode 0700, that the ledger and the episode "
            "artefact are written to. It is not created here"
        ),
    )
    parser.add_argument(
        "--cost-cap-usd",
        required=True,
        help=(
            "required exact decimal USD amount. Must equal USD 0.52 numerically; "
            "equivalent exact-decimal spellings are accepted"
        ),
    )
    parser.add_argument(
        "--input-usd-per-mtok",
        required=True,
        help=(
            "required operator-stated exact decimal rate per million tokens. Must "
            "equal USD 0.20 numerically; equivalent exact-decimal spellings are "
            "accepted, without claiming this is any vendor's current price"
        ),
    )
    parser.add_argument(
        "--output-usd-per-mtok",
        required=True,
        help=(
            "required operator-stated exact decimal rate per million tokens. Must "
            "equal USD 1.20 numerically; equivalent exact-decimal spellings are "
            "accepted, on the same provenance terms"
        ),
    )
    return parser


def _failure(mode: str, exc: BaseException) -> dict[str, Any]:
    """What an operator is told when a run does not pass.

    A class, a code and the fixed sentence that code means — and nothing the
    exception said. An exception raised by a client, an SDK or a provider is
    prose written by somebody else: it is entitled to quote the request header
    it failed on, the response body it got back, or the credential it was
    rejected for, and this build has no way to know which. Rendering it puts all
    three into whatever collects this tool's stdout. The exception itself is not
    lost — it stays in memory, on the traceback, for whoever is holding the
    debugger — but it is never printed and never filed.

    A code this build does not know is reported as unclassified rather than
    passed through, because "render it if it looks safe" is the same decision
    that leaks the first time an unfamiliar exception arrives.
    """
    if isinstance(exc, CanaryRefusal):
        failure_class, code = FAILURE_CLASS_REFUSAL, exc.code
    elif isinstance(exc, CanaryFailure):
        failure_class, code = FAILURE_CLASS_CELL, exc.code
    else:
        failure_class, code = FAILURE_CLASS_INTERNAL, CODE_UNEXPECTED_INTERNAL_ERROR
    if code not in FAILURE_DETAILS:
        code = CODE_UNCLASSIFIED
    return {
        "canary": "lifecycle_v1_execution_integrity",
        "ok": False,
        "mode": mode,
        "n": CANARY_EPISODES,
        "claim": CANARY_CLAIM,
        "live_provider_call": mode == MODE_LIVE,
        "failure": {
            "class": failure_class,
            "code": code,
            "detail": FAILURE_DETAILS[code],
        },
    }


def _emit(payload: Mapping[str, Any], *, mode: str) -> int:
    """Print one report, or withhold it whole if it is shaped like a secret.

    The last thing between this process and its stdout. Redacting the match and
    printing the rest would be a guess about which part was the secret; a report
    that trips this is withheld entirely and replaced by the code saying so.
    """
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    tripped = scan_for_secrets(rendered)
    if tripped:
        # Built from fixed terms only — no claim text, no summary, nothing this
        # run computed — because the report that replaces a suspect one cannot
        # be assembled out of the same material.
        withheld = {
            "canary": "lifecycle_v1_execution_integrity",
            "ok": False,
            "mode": mode if mode in (MODE_OFFLINE, MODE_LIVE) else CODE_UNCLASSIFIED,
            "live_provider_call": mode == MODE_LIVE,
            "failure": {
                "class": FAILURE_CLASS_INTERNAL,
                "code": CODE_OUTPUT_WITHHELD_BY_SECRET_SCANNER,
                "detail": FAILURE_DETAILS[CODE_OUTPUT_WITHHELD_BY_SECRET_SCANNER],
            },
        }
        print(json.dumps(withheld, indent=2, sort_keys=True))
        return 1
    print(rendered)
    return 0 if payload.get("ok") else 1


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    mode = MODE_LIVE if arguments.live else MODE_OFFLINE
    try:
        cap = Decimal(arguments.cost_cap_usd)
        rates = (
            Decimal(arguments.input_usd_per_mtok),
            Decimal(arguments.output_usd_per_mtok),
        )
    except InvalidOperation:
        print("every rate and cap is an exact decimal string", file=sys.stderr)
        return 2
    try:
        policy = LifecyclePricingPolicy(
            policy_id=FIXED_CANARY_POLICY_ID,
            input_usd_per_mtok=rates[0],
            output_usd_per_mtok=rates[1],
            rate_source=RATE_SOURCE_OPERATOR,
        )
        if arguments.live:
            summary = run_live_cell(
                directory=arguments.output_dir, policy=policy, cap=cap
            )
        else:
            summary = run_offline_preflight(
                directory=arguments.output_dir,
                policy=policy,
                cap=cap,
                environ=os.environ,
            )
    except Exception as exc:
        return _emit(_failure(mode, exc), mode=mode)
    return _emit(
        {
            "canary": "lifecycle_v1_execution_integrity",
            "ok": True,
            "n": CANARY_EPISODES,
            "episodes": CANARY_EPISODES,
            "claim": CANARY_CLAIM,
            "provider": CANARY_PROVIDER,
            "api": CANARY_API,
            "model": CANARY_MODEL,
            "request_profile": CANARY_PROFILE_ID,
            "scenario_id": CANARY_SCENARIO_ID,
            **summary,
        },
        mode=mode,
    )


if __name__ == "__main__":
    raise SystemExit(main())
