"""Historical validator for the quarantined Maintenance matched-arm Alpha.

The obsolete fixed 3x4 / two-of-three operator is retained only so sealed material
can still be parsed and audited.  It is not a current methodology implementation:
all new execution is quarantined, no provider or credential path is reachable, and
the sealed Engine 0.7 Alpha remains permanently ``UNINTERPRETABLE``.  It must not
be rerun, regraded, or used to create a public research summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import time
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, cast

import httpx
import openai

from boundarybench.jsonsafe import canonical_json_bytes
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
from operatebench.agents.playback import (
    RecordedOutcomeAgent,
    RecordingAgent,
    tape_from_records,
)
from operatebench.agents.pricing import (
    RATE_SOURCE_OPERATOR,
    LifecycleCostGuard,
    LifecyclePricingPolicy,
)
from operatebench.agents.transport import content_digest
from operatebench.artifact import (
    ARTIFACT_VERSION,
    read_artifact,
    replay_artifact,
    write_artifact,
)
from operatebench.core.instance import new_operation_instance_id
from operatebench.core.retrieval import RetrieveBatch
from operatebench.domains.lettings.maintenance.semantic_arms_v1 import (
    ADMISSIBLE,
    ARM_BOUNDARY,
    ARM_FULL,
    ARM_NAMES,
    ARM_ORDERED,
    ARM_STATIC,
    ERROR,
    NOT_ESTABLISHED,
    UNREACHED,
    CompiledArm,
    CompiledPoint,
    compile_maintenance_v1,
    maintenance_v1_semantic_scenario,
    map_full_outcome,
    run_boundary,
    run_static,
)
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.execution_bundle import audit_execution_bundle_files
from operatebench.execution_ledger import (
    COST_CAP_POLICY_REFUSE,
    LedgerControls,
    read_execution_ledger,
)
from operatebench.providers.cost import usd_text
from operatebench.providers.openai_responses import (
    GPT_5_6_LUNA_MODEL,
    GPT_5_6_LUNA_PROFILE,
    OPENAI_BASE_URL,
    PARALLEL_TOOL_CALLS,
    STORE,
    TOOL_CHOICE,
)
from operatebench.providers.telemetry import TurnDeadline
from operatebench.runner import run_episode
from operatebench.version import OPERATEBENCH_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "examples" / "operatebench" / "maintenance_v0_1.yaml"
# Digest this quarantined source after replacing only the next literal with zeroes.
OPERATOR_SOURCE_DIGEST_SHA256 = "e8e215303c20f5ef0cdec22355f8ad5bb8cef4d4cf09db2f4a79256b7a0c7eab"  # fmt: skip  # noqa: E501
# Validation remains compatible with evidence emitted by the pre-quarantine operator.
HISTORICAL_OPERATOR_SOURCE_DIGEST_SHA256 = (
    "e10a8020d532ad6461c4a3b3dbf7f5b0eceb7e3714c8037cf19f9e3855e2fbbf"
)
ENGINE_VERSION = "0.9.0"
OPERATION_VERSION = "0.6.0"
OPERATION_ID = "lettings_maintenance_synthetic_v1"
SPEC_DIGEST_SHA256 = "9e99b409b153e03aeb240c6402e14ff21c8bf0df3b9a1df2016c8fa64138e44d"
SCENARIO_ID = "V1"
SEMANTIC_SCENARIO_ID = "maintenance_recurring_leak_synthetic_v1"
SEMANTIC_SCENARIO_DIGEST_SHA256 = (
    "593b82e0cd8ce9cb0b25bafefc5c2f61e9cf405b6f66b7e8ec8c5b15a40e8b89"
)
COMPILER_ID = "operatebench.maintenance.semantic_arms_v1.compile_maintenance_v1"
SCAFFOLD_ID = "lifecycle_openai_responses_v7_maintenance_v1"
PROVIDER = "openai"
API = "responses"
MODEL = "gpt-5.6-luna"
PROFILE_ID = "gpt56luna_reasoning_none_sampling_omitted_v2"
POLICY_ID = "matched_arm_alpha_operator_pinned_v1"
RATE_SOURCE = "operator_supplied_pinned_rates"
TRIALS_PER_ARM = 3
POINT_DENOMINATOR = 5
MAX_POINT_CALLS = 8
FULL_EXPECTED_CALLS = 46
FULL_MAX_CALLS = 50
EXPECTED_CALLS = 228
HARD_CALL_CAP = 510
LARGEST_REQUEST_TOKEN_BOUND = 23_409
MAX_INPUT_TOKENS = 11_938_590
MAX_OUTPUT_TOKENS_TOTAL = 2_088_960
COMBINED_TOKEN_CAP = 14_027_550
INPUT_USD_PER_MTOK = Decimal("0.20")
OUTPUT_USD_PER_MTOK = Decimal("1.20")
WORST_CASE_USD = Decimal("4.894470")
CELL_CAP_USD = Decimal("5.00")
TURN_DEADLINE_SECONDS = 30.0
WALL_CLOCK_DEADLINE_SECONDS = 18_000.0
MAX_ATTEMPTS_PER_CALL = 1
SDK_MAX_RETRIES = 0
OPENAI_SDK_VERSION = "2.53.0"

RESULT_SCHEMA = "operatebench.matched_arm_alpha_result.v2"
MANIFEST_SCHEMA = "operatebench.matched_arm_alpha_evidence_manifest.v1"
TAPE_SCHEMA = "operatebench.matched_arm_alpha_control_tape.v1"
MODE_OFFLINE = "offline_preflight"
MODE_LIVE = "live"
RESULT_NAME = "alpha-result.json"
MANIFEST_NAME = "alpha-evidence-manifest.json"
PUBLIC_SUMMARY_NAME = "alpha-public-summary.json"
CREDENTIAL_VARIABLE = "OPENAI_API_KEY"
LIVE_AUTHORIZATION_VARIABLE = "OPERATEBENCH_MATCHED_ALPHA_AUTHORIZATION"
EXPECTED_SOURCE_REVISION_VARIABLE = "OPERATEBENCH_MATCHED_ALPHA_EXPECTED_SOURCE_SHA"
PLACEHOLDER_KEY = "matched-alpha-mock-transport-not-a-credential"
RESERVATION_SUFFIX = ".alpha-reservation-not-evidence"

SUPPORTED_LONGITUDINAL_DIFFERENTIAL = "SUPPORTED_LONGITUDINAL_DIFFERENTIAL"
NO_SUPPORTED_LONGITUDINAL_DIFFERENTIAL = "NO_SUPPORTED_LONGITUDINAL_DIFFERENTIAL"
UNINTERPRETABLE = "UNINTERPRETABLE"

AUTHORED_SCHEDULE: tuple[tuple[int, str], ...] = (
    (1, ARM_STATIC),
    (1, ARM_BOUNDARY),
    (1, ARM_ORDERED),
    (1, ARM_FULL),
    (2, ARM_BOUNDARY),
    (2, ARM_ORDERED),
    (2, ARM_FULL),
    (2, ARM_STATIC),
    (3, ARM_ORDERED),
    (3, ARM_FULL),
    (3, ARM_STATIC),
    (3, ARM_BOUNDARY),
)
POINT_IDS = (
    "initial_report",
    "approval_granted",
    "work_verified",
    "invoice_validated",
    "warranty_reopened",
)


def _control_names() -> tuple[str, ...]:
    names: list[str] = []
    for trial, arm in AUTHORED_SCHEDULE:
        if arm != ARM_FULL:
            for point in POINT_IDS:
                prefix = f"control-t{trial}-{arm}-{point}"
                names.extend((f"{prefix}.ledger.ndjson", f"{prefix}.tape.json"))
    return tuple(names)


OUTPUT_NAMES = (
    *_control_names(),
    *(
        name
        for trial in range(1, 4)
        for name in (f"full-t{trial}.ledger.ndjson", f"full-t{trial}.artifact.json")
    ),
    MANIFEST_NAME,
    RESULT_NAME,
    PUBLIC_SUMMARY_NAME,
)


class AlphaRefusal(Exception):
    """Safe fixed-code refusal; runtime/provider prose is never emitted."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class AlphaFailure(AlphaRefusal):
    pass


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def normalized_operator_source_bytes(source: bytes | None = None) -> bytes:
    data = Path(__file__).read_bytes() if source is None else source
    pattern = rb'OPERATOR_SOURCE_DIGEST_SHA256 = "[0-9a-f]{64}"'
    replacement = b'OPERATOR_SOURCE_DIGEST_SHA256 = "' + b"0" * 64 + b'"'
    normalized, count = re.subn(pattern, replacement, data)
    if count != 1:
        raise AlphaRefusal("operator_source_digest_literal_invalid")
    return normalized


def normalized_operator_source_digest() -> str:
    return _sha_bytes(normalized_operator_source_bytes())


def plan_payload() -> dict[str, Any]:
    return {
        "experiment": "maintenance_v1_matched_arm_alpha_v2",
        "operatebench_version": ENGINE_VERSION,
        "operation_id": OPERATION_ID,
        "operation_version": OPERATION_VERSION,
        "spec_digest_sha256": SPEC_DIGEST_SHA256,
        "scenario_id": SCENARIO_ID,
        "semantic_scenario_id": SEMANTIC_SCENARIO_ID,
        "semantic_scenario_digest_sha256": SEMANTIC_SCENARIO_DIGEST_SHA256,
        "compiler_id": COMPILER_ID,
        "provider": PROVIDER,
        "api": API,
        "model": MODEL,
        "profile_id": PROFILE_ID,
        "scaffold_id": SCAFFOLD_ID,
        "protocol_version": "operatebench.model.v4",
        "request_mapping": "lifecycle_openai_responses_model_request_v7",
        "reasoning": {"effort": "none"},
        "temperature": "omitted",
        "top_p": "omitted",
        "store": False,
        "parallel_tool_calls": False,
        "tool_choice": "required",
        "max_output_tokens": 4096,
        "trials_per_arm": 3,
        "point_denominator": 5,
        "max_point_calls": 8,
        "full_max_calls": 50,
        "attempts_per_call": 1,
        "sdk_retries": 0,
        "openai_sdk_version": OPENAI_SDK_VERSION,
        "largest_request_token_bound": 23409,
        "turn_deadline_seconds": 30,
        "wall_clock_deadline_seconds": 18000,
        "expected_calls": EXPECTED_CALLS,
        "full_expected_calls_per_trial": 46,
        "hard_call_cap": 510,
        "max_input_tokens": 11938590,
        "max_output_tokens_total": 2088960,
        "combined_token_cap": 14027550,
        "policy_id": POLICY_ID,
        "rate_source": RATE_SOURCE,
        "input_usd_per_mtok": "0.20",
        "output_usd_per_mtok": "1.20",
        "worst_case_usd": "4.894470",
        "cost_cap_usd": "5.00",
        "schedule": [[trial, arm] for trial, arm in AUTHORED_SCHEDULE],
        "point_ids": list(POINT_IDS),
    }


def compute_plan_digest(payload: Mapping[str, Any]) -> str:
    return _sha_bytes(canonical_json_bytes(dict(payload), "matched Alpha plan"))


PLAN_DIGEST_SHA256 = "6884b82a3d299c8b03eb4e48b9fa48898d544e1de58095f58cc4483c52daaa20"


def _authorization_value_for_operator_digest(
    source_revision: str, operator_source_digest: str
) -> str:
    return (
        f"matched-alpha-v2:{source_revision}:{operator_source_digest}:"
        f"{PLAN_DIGEST_SHA256}:max-calls-510"
    )


def authorization_value(source_revision: str) -> str:
    return _authorization_value_for_operator_digest(
        source_revision, OPERATOR_SOURCE_DIGEST_SHA256
    )


def _policy() -> LifecyclePricingPolicy:
    return LifecyclePricingPolicy(
        policy_id=POLICY_ID,
        input_usd_per_mtok=INPUT_USD_PER_MTOK,
        output_usd_per_mtok=OUTPUT_USD_PER_MTOK,
        rate_source=RATE_SOURCE,
    )


def _controls(max_calls: int, expected: int) -> LedgerControls:
    return LedgerControls(
        max_provider_calls=max_calls,
        max_attempts_per_call=MAX_ATTEMPTS_PER_CALL,
        expected_provider_calls=expected,
        token_hard_cap=COMBINED_TOKEN_CAP,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        cost_cap_usd=usd_text(CELL_CAP_USD),
        cost_cap_policy=COST_CAP_POLICY_REFUSE,
        turn_deadline_seconds=TURN_DEADLINE_SECONDS,
        wall_clock_deadline_seconds=WALL_CLOCK_DEADLINE_SECONDS,
        pricing=_policy().as_dict(),
    )


def check_pinned_build() -> None:
    if (
        OPERATEBENCH_VERSION != ENGINE_VERSION
        or ENGINE_VERSION != "0.9.0"
        or OPERATION_ID != "lettings_maintenance_synthetic_v1"
        or OPERATION_VERSION != "0.6.0"
        or SPEC_DIGEST_SHA256
        != "9e99b409b153e03aeb240c6402e14ff21c8bf0df3b9a1df2016c8fa64138e44d"
        or COMPILER_ID
        != "operatebench.maintenance.semantic_arms_v1.compile_maintenance_v1"
        or SCAFFOLD_ID != "lifecycle_openai_responses_v7_maintenance_v1"
        or MODEL != "gpt-5.6-luna"
        or PROFILE_ID != "gpt56luna_reasoning_none_sampling_omitted_v2"
        or POLICY_ID != "matched_arm_alpha_operator_pinned_v1"
        or RATE_SOURCE != "operator_supplied_pinned_rates"
        or Decimal("0.20") != INPUT_USD_PER_MTOK
        or Decimal("1.20") != OUTPUT_USD_PER_MTOK
        or Decimal("5.00") != CELL_CAP_USD
        or HARD_CALL_CAP != 510
        or EXPECTED_CALLS != 228
        or FULL_EXPECTED_CALLS != 46
        or MAX_POINT_CALLS != 8
        or MAX_ATTEMPTS_PER_CALL != 1
        or SDK_MAX_RETRIES != 0
        or OPENAI_SDK_VERSION != "2.53.0"
        or TURN_DEADLINE_SECONDS != 30.0
        or WALL_CLOCK_DEADLINE_SECONDS != 18_000.0
        or LARGEST_REQUEST_TOKEN_BOUND != 23_409
        or COMBINED_TOKEN_CAP != 14_027_550
        or MAX_INPUT_TOKENS != 11_938_590
        or MAX_OUTPUT_TOKENS_TOTAL != 2_088_960
        or Decimal("4.894470") != WORST_CASE_USD
    ):
        raise AlphaRefusal("pinned_build_control_changed")
    if openai.__version__ != OPENAI_SDK_VERSION:
        raise AlphaRefusal("openai_sdk_version_mismatch")
    if normalized_operator_source_digest() != OPERATOR_SOURCE_DIGEST_SHA256:
        raise AlphaRefusal("operator_source_digest_mismatch")
    if (
        GPT_5_6_LUNA_MODEL != MODEL
        or GPT_5_6_LUNA_PROFILE.profile_id != PROFILE_ID
        or GPT_5_6_LUNA_PROFILE.reasoning != {"effort": "none"}
        or GPT_5_6_LUNA_PROFILE.omitted_fields() != ("temperature", "top_p")
        or STORE is not False
        or PARALLEL_TOOL_CALLS is not False
        or TOOL_CHOICE != "required"
        or MODEL_PROTOCOL_VERSION != "operatebench.model.v4"
        or LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION
        != "lifecycle_openai_responses_model_request_v7"
        or LIFECYCLE_RETRY_POLICY.max_attempts != 1
        or MAX_OUTPUT_TOKENS != 4096
        or RATE_SOURCE_OPERATOR != RATE_SOURCE
        or compute_plan_digest(plan_payload()) != PLAN_DIGEST_SHA256
    ):
        raise AlphaRefusal("pinned_runtime_surface_changed")
    if (
        _policy().worst_case_usd(
            calls=HARD_CALL_CAP,
            input_token_bound=LARGEST_REQUEST_TOKEN_BOUND,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        != WORST_CASE_USD
    ):
        raise AlphaRefusal("pinned_cost_envelope_changed")


def compile_pinned_arms() -> tuple[Any, Any, Mapping[str, CompiledArm]]:
    spec = load_spec(FIXTURE)
    spec.verify_identity()
    scenario = maintenance_v1_semantic_scenario(spec)
    if (spec.operation_id, spec.operation_version, spec.spec_digest_sha256) != (
        OPERATION_ID,
        OPERATION_VERSION,
        SPEC_DIGEST_SHA256,
    ):
        raise AlphaRefusal("fixture_identity_mismatch")
    if (scenario.scenario_id, scenario.digest_sha256) != (
        SEMANTIC_SCENARIO_ID,
        SEMANTIC_SCENARIO_DIGEST_SHA256,
    ):
        raise AlphaRefusal("semantic_scenario_identity_mismatch")
    arms = compile_maintenance_v1(spec, scenario)
    if tuple(arms) != ARM_NAMES or any(
        arm.point_ids != POINT_IDS for arm in arms.values()
    ):
        raise AlphaRefusal("compiler_surface_mismatch")
    return spec, scenario, arms


def _open_output_directory(directory: Path) -> int:
    try:
        fd = os.open(
            directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
    except OSError as exc:
        raise AlphaRefusal("output_directory_unopenable") from exc
    info = os.fstat(fd)
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
        os.close(fd)
        raise AlphaRefusal("output_directory_not_owned_mode_0700")
    return fd


@contextmanager
def reserve_output_names(directory: Path) -> Iterator[None]:
    fd = _open_output_directory(directory)
    held: list[tuple[str, int]] = []
    try:
        for name in OUTPUT_NAMES:
            try:
                os.lstat(name, dir_fd=fd)
            except FileNotFoundError:
                pass
            else:
                raise AlphaRefusal("output_target_exists")
            try:
                os.lstat(name + RESERVATION_SUFFIX, dir_fd=fd)
            except FileNotFoundError:
                pass
            else:
                raise AlphaRefusal("output_reserved")
        for name in OUTPUT_NAMES:
            reservation = name + RESERVATION_SUFFIX
            try:
                rfd = os.open(
                    reservation,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                    0o600,
                    dir_fd=fd,
                )
            except OSError as exc:
                raise AlphaRefusal("output_not_reservable") from exc
            os.write(rfd, b"matched Alpha reservation; not evidence\n")
            os.fsync(rfd)
            held.append((reservation, rfd))
        yield
    finally:
        for name, rfd in held:
            with suppress(OSError):
                os.close(rfd)
            with suppress(OSError):
                os.unlink(name, dir_fd=fd)
        os.close(fd)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    data = _json_output_bytes(value)
    fd = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600
    )
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def _json_output_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(dict(value), "matched Alpha output") + b"\n"


def open_live_transport() -> httpx.BaseTransport:
    return httpx.HTTPTransport(retries=0)


def open_live_client(api_key: str, *, transport: httpx.BaseTransport) -> openai.OpenAI:
    return openai.OpenAI(
        api_key=api_key,
        base_url=OPENAI_BASE_URL,
        max_retries=0,
        http_client=httpx.Client(transport=transport, timeout=TURN_DEADLINE_SECONDS),
    )


class DeadlinedAgent(EvidenceRecordingModelAgent):
    def __init__(
        self, *args: Any, deadline_at: float, wire: WireCaptureTransport, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._deadline_at = deadline_at
        self._wire = wire

    def decide(self, observation: Any) -> Any:
        if time.monotonic() >= self._deadline_at:
            raise AlphaFailure("wall_deadline_exceeded")
        if self._wire.calls >= HARD_CALL_CAP:
            raise AlphaFailure("hard_call_cap_reached")
        outcome = super().decide(observation)
        if time.monotonic() >= self._deadline_at:
            raise AlphaFailure("wall_deadline_exceeded")
        return outcome


class AlphaOpenAIResponsesTransport(OpenAIResponsesTransport):
    """Bound every SDK attempt by both the turn and global wall deadline."""

    def __init__(
        self,
        *args: Any,
        deadline_at: float,
        clock: Any = time.monotonic,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, clock=clock, **kwargs)
        self._alpha_deadline_at = deadline_at
        self._alpha_clock = clock

    def _deadline(self, started: float) -> TurnDeadline:
        budget = min(self._deadline_seconds, self._alpha_deadline_at - started)
        if budget <= 0:
            raise AlphaFailure("wall_deadline_exceeded")
        clock = self._alpha_clock
        bound = min(started + budget, self._alpha_deadline_at)
        return TurnDeadline(
            remaining_seconds=budget,
            cancelled=lambda: clock() >= bound,
            started_at=started,
        )

    def send(self, request: Any) -> Any:
        answer = super().send(request)
        if self._alpha_clock() >= self._alpha_deadline_at:
            raise AlphaFailure("wall_deadline_exceeded")
        return answer


def _expected_provider_identity() -> dict[str, Any]:
    """Derive the exact ledger provider header from the canonical live profile."""
    client = openai.OpenAI(
        api_key=PLACEHOLDER_KEY,
        base_url=OPENAI_BASE_URL,
        max_retries=SDK_MAX_RETRIES,
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(500))
        ),
    )
    try:
        transport = AlphaOpenAIResponsesTransport(
            model=GPT_5_6_LUNA_MODEL,
            client=client,
            deadline_seconds=TURN_DEADLINE_SECONDS,
            cost_guard=LifecycleCostGuard(
                policy=_policy(),
                cap_usd=CELL_CAP_USD,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                token_hard_cap=COMBINED_TOKEN_CAP,
            ),
            profile=GPT_5_6_LUNA_PROFILE,
            deadline_at=float("inf"),
        )
        return provider_identity_for(transport).as_dict()
    finally:
        client.close()


def _new_agent(
    spec: Any,
    *,
    client: Any,
    wire: WireCaptureTransport,
    ledger: Path,
    identity_label: str,
    operation_instance_id: str,
    max_calls: int,
    expected: int,
    guard: LifecycleCostGuard,
    deadline_at: float,
) -> tuple[RecordingAgent, ProviderEvidenceRecorder]:
    transport = AlphaOpenAIResponsesTransport(
        model=MODEL,
        client=client,
        deadline_seconds=TURN_DEADLINE_SECONDS,
        cost_guard=guard,
        profile=GPT_5_6_LUNA_PROFILE,
        deadline_at=deadline_at,
    )
    recorder = ProviderEvidenceRecorder(
        path=ledger,
        identity=EvidenceRunIdentity(
            operation_id=spec.operation_id,
            spec_digest_sha256=spec.spec_digest_sha256,
            scenario_id=SCENARIO_ID,
            agent_id=identity_label,
        ),
        provider=provider_identity_for(transport),
        controls=_controls(max_calls, expected),
        wire=wire,
        guard=guard,
    )
    transport.attach_recorder(recorder)
    inner = DeadlinedAgent(
        transport,
        model=MODEL,
        agent_id=identity_label,
        recorder=recorder,
        max_transport_calls=max_calls,
        deadline_at=deadline_at,
        wire=wire,
    )
    recording = RecordingAgent(inner)
    identity = {
        "operation_id": spec.operation_id,
        "operation_instance_id": operation_instance_id,
        "scenario_id": SCENARIO_ID,
        "agent_id": identity_label,
        "spec_digest_sha256": spec.spec_digest_sha256,
    }
    recording.begin_episode(identity)
    return recording, recorder


def _run_control(
    spec: Any,
    point: CompiledPoint,
    arm: str,
    trial: int,
    *,
    client: Any,
    wire: WireCaptureTransport,
    directory: Path,
    guard: LifecycleCostGuard,
    deadline_at: float,
) -> tuple[str, dict[str, Any]]:
    prefix = f"control-t{trial}-{arm}-{point.point_id}"
    ledger_path = directory / f"{prefix}.ledger.ndjson"
    tape_path = directory / f"{prefix}.tape.json"
    opaque = new_operation_instance_id()
    agent, recorder = _new_agent(
        spec,
        client=client,
        wire=wire,
        ledger=ledger_path,
        identity_label=f"alpha-control-{secrets.token_hex(16)}",
        operation_instance_id=opaque,
        max_calls=MAX_POINT_CALLS,
        expected=2,
        guard=guard,
        deadline_at=deadline_at,
    )
    recorder.begin(operation_instance_id=opaque)
    decisions: list[Any] = []
    observations: list[Any] = []
    session = point.new_retrieval_session()
    try:
        while len(decisions) < MAX_POINT_CALLS:
            if time.monotonic() >= deadline_at:
                raise AlphaFailure("wall_deadline_exceeded")
            observation = replace(
                session.current_observation(), operation_instance_id=opaque
            )
            observations.append(observation)
            decision = agent.decide(observation)
            decisions.append(decision)
            if isinstance(decision, RetrieveBatch):
                session.submit(decision)
                continue
            break
    except BaseException:
        recorder.finalize_aborted()
        raise
    if time.monotonic() >= deadline_at:
        recorder.finalize_aborted()
        raise AlphaFailure("wall_deadline_exceeded")
    recorder.finalize_scored()
    status = (
        run_boundary(spec, point, decisions).vector_status
        if arm == ARM_BOUNDARY
        else run_static(point, decisions).vector_status
    )
    tape = agent.tape()
    playback = RecordedOutcomeAgent(tape, agent_id="alpha-control-playback")
    playback.begin_episode({})
    replayed = [playback.decide(observation) for observation in observations]
    replay_status = (
        run_boundary(spec, point, replayed).vector_status
        if arm == ARM_BOUNDARY
        else run_static(point, replayed).vector_status
    )
    if replay_status != status or playback.cursor != len(tape):
        raise AlphaFailure("control_replay_mismatch")
    tape_doc = {
        "schema": TAPE_SCHEMA,
        "trial": trial,
        "arm": arm,
        "point_id": point.point_id,
        "decisions": tape.as_list(),
        "local_replay_status": replay_status,
        "provider_calls_during_replay": 0,
    }
    _write_json(tape_path, tape_doc)
    ledger = read_execution_ledger(ledger_path, require_complete=True)
    return status, {
        "kind": "control",
        "trial": trial,
        "arm": arm,
        "point_id": point.point_id,
        "ledger": ledger_path.name,
        "ledger_digest_sha256": _sha_file(ledger_path),
        "tape": tape_path.name,
        "tape_digest_sha256": _sha_file(tape_path),
        "request_digests_sha256": [call.request_digest_sha256 for call in ledger.calls],
        "model": ledger.header.provider.model,
        "profile_id": PROFILE_ID,
        "settings_digest_sha256": ledger.header.provider.settings_digest_sha256,
        "provider_calls": ledger.totals.provider_calls,
        "provider_attempts": ledger.totals.attempts,
        "input_tokens": ledger.totals.input_tokens,
        "output_tokens": ledger.totals.output_tokens,
        "retries": ledger.totals.attempts - ledger.totals.provider_calls,
        "measured_cost_usd": ledger.totals.measured_cost_usd,
        "status": status,
        "replay_ok": True,
    }


def _run_full(
    spec: Any,
    arm: CompiledArm,
    trial: int,
    *,
    client: Any,
    wire: WireCaptureTransport,
    directory: Path,
    guard: LifecycleCostGuard,
    deadline_at: float,
) -> tuple[list[str], dict[str, Any]]:
    ledger_path = directory / f"full-t{trial}.ledger.ndjson"
    artifact_path = directory / f"full-t{trial}.artifact.json"
    opaque_agent = f"alpha-full-{secrets.token_hex(16)}"
    transport = AlphaOpenAIResponsesTransport(
        model=MODEL,
        client=client,
        deadline_seconds=TURN_DEADLINE_SECONDS,
        cost_guard=guard,
        profile=GPT_5_6_LUNA_PROFILE,
        deadline_at=deadline_at,
    )
    recorder = ProviderEvidenceRecorder(
        path=ledger_path,
        identity=EvidenceRunIdentity(
            operation_id=spec.operation_id,
            spec_digest_sha256=spec.spec_digest_sha256,
            scenario_id=SCENARIO_ID,
            agent_id=opaque_agent,
        ),
        provider=provider_identity_for(transport),
        controls=_controls(FULL_MAX_CALLS, FULL_EXPECTED_CALLS),
        wire=wire,
        guard=guard,
    )
    transport.attach_recorder(recorder)
    agent = DeadlinedAgent(
        transport,
        model=MODEL,
        agent_id=opaque_agent,
        recorder=recorder,
        max_transport_calls=FULL_MAX_CALLS,
        deadline_at=deadline_at,
        wire=wire,
    )
    if time.monotonic() >= deadline_at:
        raise AlphaFailure("wall_deadline_exceeded")
    run = run_episode(
        spec,
        SCENARIO_ID,
        opaque_agent,
        agent_factory=lambda: agent,
        agent_kind="model",
        evidence_recorder=recorder,
    )
    if time.monotonic() >= deadline_at:
        raise AlphaFailure("wall_deadline_exceeded")
    vector = list(map_full_outcome(arm.points, run.outcome))
    write_artifact(run, artifact_path)
    record = read_artifact(artifact_path)
    bundle = audit_execution_bundle_files(artifact_path, ledger_path)
    replay = replay_artifact(spec, record)
    if not replay.ok:
        raise AlphaFailure("full_replay_mismatch")
    ledger = read_execution_ledger(ledger_path, require_complete=True)
    return vector, {
        "kind": "full",
        "trial": trial,
        "arm": ARM_FULL,
        "point_id": None,
        "ledger": ledger_path.name,
        "ledger_digest_sha256": _sha_file(ledger_path),
        "artifact": artifact_path.name,
        "artifact_digest_sha256": _sha_file(artifact_path),
        "request_digests_sha256": [call.request_digest_sha256 for call in ledger.calls],
        "model": ledger.header.provider.model,
        "profile_id": PROFILE_ID,
        "settings_digest_sha256": ledger.header.provider.settings_digest_sha256,
        "artifact_version": ARTIFACT_VERSION,
        "provider_calls": ledger.totals.provider_calls,
        "provider_attempts": ledger.totals.attempts,
        "input_tokens": ledger.totals.input_tokens,
        "output_tokens": ledger.totals.output_tokens,
        "retries": ledger.totals.attempts - ledger.totals.provider_calls,
        "measured_cost_usd": ledger.totals.measured_cost_usd,
        "status": run.outcome.status,
        "vector": vector,
        "bundle_audit_ok": bool(bundle.ok),
        "replay_ok": replay.ok,
        "replay_provider_calls": 0,
    }


def _trial_pass(vector: Sequence[str]) -> bool:
    return len(vector) == 5 and all(value == ADMISSIBLE for value in vector)


def _historical_differential_verdict(
    vectors: Mapping[str, Sequence[Sequence[str]]],
) -> str:
    """Reconstruct an obsolete verdict solely when validating historical artifacts."""

    if any(
        value == ERROR for rows in vectors.values() for vector in rows for value in vector
    ):
        return UNINTERPRETABLE
    counts = {
        arm: sum(_trial_pass(vector) for vector in vectors.get(arm, ()))
        for arm in ARM_NAMES
    }
    if (
        all(counts[arm] >= 2 for arm in (ARM_STATIC, ARM_BOUNDARY, ARM_ORDERED))
        and counts[ARM_FULL] <= 1
    ):
        return SUPPORTED_LONGITUDINAL_DIFFERENTIAL
    return NO_SUPPORTED_LONGITUDINAL_DIFFERENTIAL


def differential_verdict(vectors: Mapping[str, Sequence[Sequence[str]]]) -> str:
    """Refuse obsolete public scientific verdict generation."""

    raise AlphaRefusal("alpha_operator_quarantined")


def _execute(
    *,
    directory: Path,
    client: Any,
    wire: WireCaptureTransport,
    mode: str,
    external_source_revision: str,
) -> dict[str, Any]:
    """Retained legacy implementation; no supported entry point can call it."""

    spec, scenario, arms = compile_pinned_arms()
    started = time.monotonic()
    deadline_at = started + WALL_CLOCK_DEADLINE_SECONDS
    guard = LifecycleCostGuard(
        policy=_policy(),
        cap_usd=CELL_CAP_USD,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        token_hard_cap=COMBINED_TOKEN_CAP,
    )
    manifest: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    vectors: dict[str, list[list[str]]] = {arm: [] for arm in ARM_NAMES}
    for trial, arm_name in AUTHORED_SCHEDULE:
        if time.monotonic() >= deadline_at:
            raise AlphaFailure("wall_deadline_exceeded")
        if arm_name == ARM_FULL:
            vector, evidence = _run_full(
                spec,
                arms[arm_name],
                trial,
                client=client,
                wire=wire,
                directory=directory,
                guard=guard,
                deadline_at=deadline_at,
            )
        else:
            vector = []
            for point in arms[arm_name].points:
                status, evidence = _run_control(
                    spec,
                    point,
                    arm_name,
                    trial,
                    client=client,
                    wire=wire,
                    directory=directory,
                    guard=guard,
                    deadline_at=deadline_at,
                )
                vector.append(status)
                manifest.append(evidence)
            evidence = None
        if evidence is not None:
            manifest.append(evidence)
        vectors[arm_name].append(vector)
        rows.append(
            {
                "trial": trial,
                "arm": arm_name,
                "point_ids": list(POINT_IDS),
                "vector": vector,
                "pass": _trial_pass(vector),
                "calls": sum(
                    item["provider_calls"]
                    for item in manifest
                    if item["trial"] == trial and item["arm"] == arm_name
                ),
                "faults": [],
                "evidence_digests": [
                    item.get("tape_digest_sha256", item.get("artifact_digest_sha256"))
                    for item in manifest
                    if item["trial"] == trial and item["arm"] == arm_name
                ],
            }
        )
    totals = {
        key: sum(cast(int, item[key]) for item in manifest)
        for key in (
            "provider_calls",
            "provider_attempts",
            "input_tokens",
            "output_tokens",
            "retries",
        )
    }
    measured = sum((Decimal(item["measured_cost_usd"]) for item in manifest), Decimal(0))
    pass_counts = {
        arm: sum(_trial_pass(vector) for vector in vectors[arm]) for arm in ARM_NAMES
    }
    manifest_doc = {"schema": MANIFEST_SCHEMA, "entries": manifest}
    if time.monotonic() >= deadline_at:
        raise AlphaFailure("wall_deadline_exceeded")
    _write_json(directory / MANIFEST_NAME, manifest_doc)
    external_provider_calls = 0 if mode == MODE_OFFLINE else totals["provider_calls"]
    result = {
        "schema": RESULT_SCHEMA,
        "mode": mode,
        "external_provider_calls": external_provider_calls,
        "external_source_revision": external_source_revision,
        "operator_source_digest_sha256": OPERATOR_SOURCE_DIGEST_SHA256,
        "plan": plan_payload(),
        "plan_digest_sha256": PLAN_DIGEST_SHA256,
        "preregistration": {
            "authorization_digest_sha256": _sha_bytes(
                authorization_value(external_source_revision).encode()
            ),
            "expected_source_revision": external_source_revision,
        },
        "scenario": {
            "scenario_id": SCENARIO_ID,
            "semantic_scenario_id": scenario.scenario_id,
            "semantic_scenario_digest_sha256": scenario.digest_sha256,
            "spec_digest_sha256": spec.spec_digest_sha256,
        },
        "compiler": {"identity": COMPILER_ID},
        "provider": {
            "provider": PROVIDER,
            "api": API,
            "model": MODEL,
            "profile_id": PROFILE_ID,
            "protocol_version": MODEL_PROTOCOL_VERSION,
            "request_mapping": LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION,
        },
        "scaffold": {"identity": SCAFFOLD_ID, "ordered_is_memory_evidence": False},
        "schedule": [[trial, arm] for trial, arm in AUTHORED_SCHEDULE],
        "trials": rows,
        "pass_counts": pass_counts,
        "evidence_manifest_name": MANIFEST_NAME,
        "evidence_manifest_digest_sha256": _sha_file(directory / MANIFEST_NAME),
        "evidence_manifest": manifest,
        "public_summary_name": PUBLIC_SUMMARY_NAME,
        "public_summary_digest_sha256": "",
        "accounting": {
            "subject_calls": totals["provider_calls"],
            **totals,
            "measured_cost_usd": usd_text(measured),
            "hard_call_cap": HARD_CALL_CAP,
            "combined_token_cap": COMBINED_TOKEN_CAP,
            "cost_cap_usd": usd_text(CELL_CAP_USD),
        },
        "execution_integrity": {
            "complete": True,
            "no_replacement": True,
            "control_replays_ok": True,
            "full_bundle_audits_ok": True,
            "full_replays_ok": True,
        },
        "differential_verdict": _historical_differential_verdict(vectors),
    }
    summary = _historical_public_summary(result)
    result["public_summary_digest_sha256"] = _sha_bytes(_json_output_bytes(summary))
    if time.monotonic() >= deadline_at:
        raise AlphaFailure("wall_deadline_exceeded")
    _write_json(directory / PUBLIC_SUMMARY_NAME, summary)
    validate_result(result, directory=directory)
    if time.monotonic() >= deadline_at:
        raise AlphaFailure("wall_deadline_exceeded")
    _write_json(directory / RESULT_NAME, result)
    validate_result(
        json.loads((directory / RESULT_NAME).read_text()), directory=directory
    )
    _scan_outputs(directory)
    return result


def _scan_outputs(directory: Path) -> None:
    patterns = (
        re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"),
        re.compile(r"(?i)\bbearer\s+"),
        re.compile(r"(?i)\bapi[_-]?key\b\s*[:=]"),
    )
    for name in OUTPUT_NAMES:
        text = (directory / name).read_text(encoding="utf-8", errors="replace")
        if any(pattern.search(text) for pattern in patterns):
            raise AlphaFailure("secret_scan_failed")


def run_offline_preflight(
    *, directory: Path, environ: Mapping[str, str]
) -> dict[str, Any]:
    """Refuse obsolete synthetic execution; arguments are intentionally unread."""

    raise AlphaRefusal("alpha_operator_quarantined")


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AlphaRefusal("source_revision_unavailable") from exc


def _git_revision_and_clean() -> str:
    head = _git_head()
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AlphaRefusal("source_revision_unavailable") from exc
    if status:
        raise AlphaRefusal("tracked_source_not_clean")
    try:
        tracked = subprocess.run(
            [
                "git",
                "ls-files",
                "--error-unmatch",
                str(Path(__file__).relative_to(REPO_ROOT)),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
    except ValueError as exc:
        raise AlphaRefusal("operator_source_outside_repository") from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AlphaRefusal("source_revision_unavailable") from exc
    if tracked.returncode != 0:
        raise AlphaRefusal("operator_source_not_tracked")
    return head


def _run_live(*, directory: Path, environ: MutableMapping[str, str]) -> dict[str, Any]:
    """Refuse obsolete live execution before reading either argument."""

    raise AlphaRefusal("alpha_operator_quarantined")


def run_live(*, directory: Path) -> dict[str, Any]:
    """Refuse obsolete live execution without consulting process credentials."""

    raise AlphaRefusal("alpha_operator_quarantined")


_TOP_FIELDS = {
    "schema",
    "mode",
    "external_provider_calls",
    "external_source_revision",
    "operator_source_digest_sha256",
    "plan",
    "plan_digest_sha256",
    "preregistration",
    "scenario",
    "compiler",
    "provider",
    "scaffold",
    "schedule",
    "trials",
    "pass_counts",
    "evidence_manifest_name",
    "evidence_manifest_digest_sha256",
    "evidence_manifest",
    "public_summary_name",
    "public_summary_digest_sha256",
    "accounting",
    "execution_integrity",
    "differential_verdict",
}
_TRIAL_FIELDS = {
    "trial",
    "arm",
    "point_ids",
    "vector",
    "pass",
    "calls",
    "faults",
    "evidence_digests",
}
_ACCOUNT_FIELDS = {
    "subject_calls",
    "provider_calls",
    "provider_attempts",
    "input_tokens",
    "output_tokens",
    "retries",
    "measured_cost_usd",
    "hard_call_cap",
    "combined_token_cap",
    "cost_cap_usd",
}
_COMMON_EVIDENCE_FIELDS = {
    "kind",
    "trial",
    "arm",
    "point_id",
    "ledger",
    "ledger_digest_sha256",
    "request_digests_sha256",
    "model",
    "profile_id",
    "settings_digest_sha256",
    "provider_calls",
    "provider_attempts",
    "input_tokens",
    "output_tokens",
    "retries",
    "measured_cost_usd",
    "status",
    "replay_ok",
}
_CONTROL_EVIDENCE_FIELDS = _COMMON_EVIDENCE_FIELDS | {"tape", "tape_digest_sha256"}
_FULL_EVIDENCE_FIELDS = _COMMON_EVIDENCE_FIELDS | {
    "artifact",
    "artifact_digest_sha256",
    "artifact_version",
    "vector",
    "bundle_audit_ok",
    "replay_provider_calls",
}


def _exact_fields(value: Any, fields: set[str], code: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise AlphaRefusal(code)
    return value


def _nat(value: Any, code: str) -> int:
    if type(value) is not int or value < 0:
        raise AlphaRefusal(code)
    return value


def _money(value: Any, code: str) -> Decimal:
    if not isinstance(value, str):
        raise AlphaRefusal(code)
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise AlphaRefusal(code) from exc
    if not amount.is_finite() or amount < 0:
        raise AlphaRefusal(code)
    return amount


def validate_result(
    raw: Mapping[str, Any], *, directory: Path | None, verify_files: bool = True
) -> dict[str, Any]:
    value = cast(dict[str, Any], json.loads(json.dumps(raw)))
    _exact_fields(value, _TOP_FIELDS, "result_schema_not_closed")
    if value["schema"] != RESULT_SCHEMA or value["mode"] not in (MODE_OFFLINE, MODE_LIVE):
        raise AlphaRefusal("result_identity_mismatch")
    external_calls = _nat(
        value["external_provider_calls"], "external_provider_calls_invalid"
    )
    if (
        not isinstance(value["external_source_revision"], str)
        or re.fullmatch(r"[0-9a-f]{40}", value["external_source_revision"]) is None
        or value["operator_source_digest_sha256"]
        not in {
            OPERATOR_SOURCE_DIGEST_SHA256,
            HISTORICAL_OPERATOR_SOURCE_DIGEST_SHA256,
        }
    ):
        raise AlphaRefusal("result_source_binding_mismatch")
    expected_plan = plan_payload()
    stated_plan = value["plan"]
    if not isinstance(stated_plan, dict) or set(stated_plan) != set(expected_plan):
        raise AlphaRefusal("result_plan_schema_mismatch")
    if stated_plan != expected_plan:
        raise AlphaRefusal("result_plan_identity_mismatch")
    if (
        compute_plan_digest(stated_plan) != value["plan_digest_sha256"]
        or value["plan_digest_sha256"] != PLAN_DIGEST_SHA256
    ):
        raise AlphaRefusal("result_plan_digest_mismatch")
    prereg = value["preregistration"]
    if prereg != {
        "authorization_digest_sha256": _sha_bytes(
            _authorization_value_for_operator_digest(
                value["external_source_revision"],
                value["operator_source_digest_sha256"],
            ).encode()
        ),
        "expected_source_revision": value["external_source_revision"],
    }:
        raise AlphaRefusal("result_preregistration_mismatch")
    if value["scenario"] != {
        "scenario_id": SCENARIO_ID,
        "semantic_scenario_id": SEMANTIC_SCENARIO_ID,
        "semantic_scenario_digest_sha256": SEMANTIC_SCENARIO_DIGEST_SHA256,
        "spec_digest_sha256": SPEC_DIGEST_SHA256,
    }:
        raise AlphaRefusal("result_scenario_identity_mismatch")
    if value["compiler"] != {"identity": COMPILER_ID}:
        raise AlphaRefusal("result_compiler_identity_mismatch")
    if value["provider"] != {
        "provider": PROVIDER,
        "api": API,
        "model": MODEL,
        "profile_id": PROFILE_ID,
        "protocol_version": MODEL_PROTOCOL_VERSION,
        "request_mapping": LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION,
    }:
        raise AlphaRefusal("result_provider_identity_mismatch")
    if value["scaffold"] != {
        "identity": SCAFFOLD_ID,
        "ordered_is_memory_evidence": False,
    }:
        raise AlphaRefusal("result_scaffold_identity_mismatch")
    if value["schedule"] != [list(item) for item in AUTHORED_SCHEDULE]:
        raise AlphaRefusal("result_schedule_mismatch")
    rows = value["trials"]
    if not isinstance(rows, list) or len(rows) != 12:
        raise AlphaRefusal("result_trial_count")
    grouped: dict[str, list[list[str]]] = {arm: [] for arm in ARM_NAMES}
    allowed = {ADMISSIBLE, NOT_ESTABLISHED, UNREACHED, ERROR}
    for expected, row in zip(AUTHORED_SCHEDULE, rows, strict=True):
        row = _exact_fields(row, _TRIAL_FIELDS, "trial_schema_not_closed")
        if (row["trial"], row["arm"]) != expected or row["point_ids"] != list(POINT_IDS):
            raise AlphaRefusal("trial_identity_mismatch")
        vector = row["vector"]
        if (
            not isinstance(vector, list)
            or len(vector) != 5
            or any(type(item) is not str or item not in allowed for item in vector)
        ):
            raise AlphaRefusal("trial_vector_invalid")
        if type(row["pass"]) is not bool or row["pass"] != _trial_pass(vector):
            raise AlphaRefusal("trial_pass_mismatch")
        _nat(row["calls"], "trial_calls_invalid")
        if not isinstance(row["faults"], list) or not isinstance(
            row["evidence_digests"], list
        ):
            raise AlphaRefusal("trial_evidence_invalid")
        grouped[row["arm"]].append(vector)
    pass_counts = {
        arm: sum(_trial_pass(vector) for vector in grouped[arm]) for arm in ARM_NAMES
    }
    if value["pass_counts"] != pass_counts or value[
        "differential_verdict"
    ] != _historical_differential_verdict(grouped):
        raise AlphaRefusal("result_derived_science_mismatch")
    manifest = value["evidence_manifest"]
    if not isinstance(manifest, list) or len(manifest) != 48:
        raise AlphaRefusal("evidence_manifest_count")
    expected_provider = _expected_provider_identity()
    expected_manifest = [
        identity
        for trial, arm in AUTHORED_SCHEDULE
        for identity in (
            [(trial, arm, None)]
            if arm == ARM_FULL
            else [(trial, arm, point) for point in POINT_IDS]
        )
    ]
    observed_identities: list[tuple[int, str, Any]] = []
    for expected_identity, item in zip(expected_manifest, manifest, strict=True):
        if not isinstance(item, dict):
            raise AlphaRefusal("evidence_entry_not_object")
        kind = item.get("kind")
        fields = (
            _CONTROL_EVIDENCE_FIELDS
            if kind == "control"
            else _FULL_EVIDENCE_FIELDS
            if kind == "full"
            else set()
        )
        _exact_fields(item, fields, "evidence_entry_schema_not_closed")
        if (
            item["model"] != expected_provider["model"]
            or item["profile_id"] != expected_provider["settings"]["request_profile"]
            or item["settings_digest_sha256"]
            != expected_provider["settings_digest_sha256"]
        ):
            raise AlphaRefusal("evidence_provider_identity_mismatch")
        if type(item["replay_ok"]) is not bool or item["replay_ok"] is not True:
            raise AlphaRefusal("evidence_replay_invalid")
        if not isinstance(item["settings_digest_sha256"], str) or not isinstance(
            item["request_digests_sha256"], list
        ):
            raise AlphaRefusal("evidence_request_binding_invalid")
        evidence_digest = (
            item["tape_digest_sha256"]
            if kind == "control"
            else item["artifact_digest_sha256"]
        )
        for digest in [
            item["ledger_digest_sha256"],
            item["settings_digest_sha256"],
            evidence_digest,
            *item["request_digests_sha256"],
        ]:
            if (
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise AlphaRefusal("evidence_digest_invalid")
        observed_identities.append((item["trial"], item["arm"], item["point_id"]))
        trial, arm, point_id = expected_identity
        prefix = (
            f"full-t{trial}" if arm == ARM_FULL else f"control-t{trial}-{arm}-{point_id}"
        )
        expected_names = (
            (f"{prefix}.ledger.ndjson", f"{prefix}.artifact.json")
            if arm == ARM_FULL
            else (f"{prefix}.ledger.ndjson", f"{prefix}.tape.json")
        )
        if (
            item["ledger"] != expected_names[0]
            or (item["artifact"] if kind == "full" else item["tape"]) != expected_names[1]
        ):
            raise AlphaRefusal("evidence_filename_mismatch")
        if kind == "control" and item["status"] not in {
            ADMISSIBLE,
            NOT_ESTABLISHED,
            ERROR,
        }:
            raise AlphaRefusal("control_status_invalid")
        if kind == "full" and (
            item["vector"]
            != next(
                row["vector"]
                for row in rows
                if row["trial"] == item["trial"] and row["arm"] == ARM_FULL
            )
            or item["bundle_audit_ok"] is not True
            or item["replay_provider_calls"] != 0
            or item["artifact_version"] != ARTIFACT_VERSION
        ):
            raise AlphaRefusal("full_evidence_integrity_mismatch")
    if observed_identities != expected_manifest:
        raise AlphaRefusal("evidence_identity_order_mismatch")
    ledger_digests = [item["ledger_digest_sha256"] for item in manifest]
    artifact_digests = [
        item["artifact_digest_sha256"] for item in manifest if item["kind"] == "full"
    ]
    if len(set(ledger_digests)) != 48 or len(set(artifact_digests)) != 3:
        raise AlphaRefusal("duplicate_evidence_digest")
    for row in rows:
        cell = [
            item
            for item in manifest
            if item["trial"] == row["trial"] and item["arm"] == row["arm"]
        ]
        expected_digests = [
            item["tape_digest_sha256"]
            if item["kind"] == "control"
            else item["artifact_digest_sha256"]
            for item in cell
        ]
        if (
            row["calls"] != sum(item["provider_calls"] for item in cell)
            or row["evidence_digests"] != expected_digests
            or row["faults"] != []
        ):
            raise AlphaRefusal("trial_evidence_binding_mismatch")
    accounting = _exact_fields(
        value["accounting"], _ACCOUNT_FIELDS, "accounting_schema_not_closed"
    )
    for key in (
        "subject_calls",
        "provider_calls",
        "provider_attempts",
        "input_tokens",
        "output_tokens",
        "retries",
        "hard_call_cap",
        "combined_token_cap",
    ):
        _nat(accounting[key], "accounting_integer_invalid")
    cost = _money(accounting["measured_cost_usd"], "accounting_cost_invalid")
    _money(accounting["cost_cap_usd"], "accounting_cap_invalid")
    if (
        accounting["hard_call_cap"] != HARD_CALL_CAP
        or accounting["combined_token_cap"] != COMBINED_TOKEN_CAP
        or accounting["cost_cap_usd"] != usd_text(CELL_CAP_USD)
    ):
        raise AlphaRefusal("accounting_control_mismatch")
    sums = {
        key: sum(_nat(item.get(key), "manifest_accounting_invalid") for item in manifest)
        for key in (
            "provider_calls",
            "provider_attempts",
            "input_tokens",
            "output_tokens",
            "retries",
        )
    }
    if (
        accounting["provider_calls"] != sums["provider_calls"]
        or accounting["subject_calls"] != sums["provider_calls"]
    ):
        raise AlphaRefusal("accounting_calls_mismatch")
    if (value["mode"] == MODE_OFFLINE and external_calls != 0) or (
        value["mode"] == MODE_LIVE and external_calls != sums["provider_calls"]
    ):
        raise AlphaRefusal("external_provider_calls_mismatch")
    if any(
        accounting[key] != sums[key]
        for key in ("provider_attempts", "input_tokens", "output_tokens", "retries")
    ):
        raise AlphaRefusal("accounting_totals_mismatch")
    measured = sum(
        (
            _money(item.get("measured_cost_usd"), "manifest_cost_invalid")
            for item in manifest
        ),
        Decimal(0),
    )
    if (
        cost != measured
        or accounting["provider_calls"] > HARD_CALL_CAP
        or accounting["input_tokens"] > MAX_INPUT_TOKENS
        or accounting["output_tokens"] > MAX_OUTPUT_TOKENS_TOTAL
        or cost > CELL_CAP_USD
    ):
        raise AlphaRefusal("accounting_envelope_mismatch")
    integrity = value["execution_integrity"]
    if integrity != {
        "complete": True,
        "no_replacement": True,
        "control_replays_ok": True,
        "full_bundle_audits_ok": True,
        "full_replays_ok": True,
    }:
        raise AlphaRefusal("execution_integrity_mismatch")
    if verify_files:
        if directory is None:
            raise AlphaRefusal("evidence_directory_required")
        manifest_path = directory / MANIFEST_NAME
        try:
            standalone_manifest = json.loads(manifest_path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AlphaRefusal("manifest_file_binding_mismatch") from exc
        if (
            value["evidence_manifest_name"] != MANIFEST_NAME
            or standalone_manifest != {"schema": MANIFEST_SCHEMA, "entries": manifest}
            or _sha_file(manifest_path) != value["evidence_manifest_digest_sha256"]
        ):
            raise AlphaRefusal("manifest_file_binding_mismatch")
        spec, _, arms = compile_pinned_arms()
        execution_run_ids: set[str] = set()
        operation_instance_ids: set[str] = set()
        full_artifact_operation_ids: set[str] = set()
        full_artifact_agent_ids: set[str] = set()
        for item in manifest:
            ledger = directory / item["ledger"]
            if _sha_file(ledger) != item["ledger_digest_sha256"]:
                raise AlphaRefusal("ledger_digest_mismatch")
            ledger_record = read_execution_ledger(ledger, require_complete=True)
            if (
                ledger_record.header.execution_run_id in execution_run_ids
                or ledger_record.header.operation_instance_id in operation_instance_ids
            ):
                raise AlphaRefusal("duplicate_evidence_identity")
            execution_run_ids.add(ledger_record.header.execution_run_id)
            operation_instance_ids.add(ledger_record.header.operation_instance_id)
            expected_controls = _controls(
                FULL_MAX_CALLS if item["kind"] == "full" else MAX_POINT_CALLS,
                FULL_EXPECTED_CALLS if item["kind"] == "full" else 2,
            ).as_dict()
            if (
                ledger_record.header.provider.as_dict() != expected_provider
                or ledger_record.header.controls.as_dict() != expected_controls
            ):
                raise AlphaRefusal("ledger_header_binding_mismatch")
            if (
                ledger_record.header.provider.model != item["model"]
                or ledger_record.header.provider.settings_digest_sha256
                != item["settings_digest_sha256"]
                or [call.request_digest_sha256 for call in ledger_record.calls]
                != item["request_digests_sha256"]
                or ledger_record.totals.provider_calls != item["provider_calls"]
                or ledger_record.totals.attempts != item["provider_attempts"]
                or ledger_record.totals.input_tokens != item["input_tokens"]
                or ledger_record.totals.output_tokens != item["output_tokens"]
                or ledger_record.totals.attempts - ledger_record.totals.provider_calls
                != item["retries"]
                or ledger_record.totals.measured_cost_usd != item["measured_cost_usd"]
            ):
                raise AlphaRefusal("ledger_accounting_binding_mismatch")
            evidence_name = item.get("tape") or item.get("artifact")
            evidence_digest = item.get("tape_digest_sha256") or item.get(
                "artifact_digest_sha256"
            )
            if (
                not isinstance(evidence_name, str)
                or _sha_file(directory / evidence_name) != evidence_digest
            ):
                raise AlphaRefusal("evidence_digest_mismatch")
            row = next(
                row
                for row in rows
                if row["trial"] == item["trial"] and row["arm"] == item["arm"]
            )
            try:
                if item["kind"] == "control":
                    tape_doc = json.loads((directory / item["tape"]).read_text())
                    if set(tape_doc) != {
                        "schema",
                        "trial",
                        "arm",
                        "point_id",
                        "decisions",
                        "local_replay_status",
                        "provider_calls_during_replay",
                    } or (
                        tape_doc["schema"],
                        tape_doc["trial"],
                        tape_doc["arm"],
                        tape_doc["point_id"],
                    ) != (TAPE_SCHEMA, item["trial"], item["arm"], item["point_id"]):
                        raise AlphaRefusal("evidence_causal_replay_mismatch")
                    tape = tape_from_records(tape_doc["decisions"])
                    if len(tape) != len(ledger_record.calls):
                        raise AlphaRefusal("evidence_causal_replay_mismatch")
                    for recorded, call in zip(
                        tape.decisions, ledger_record.calls, strict=True
                    ):
                        if (
                            recorded.invocation_index != call.invocation_index
                            or recorded.turn_index != call.turn_index
                            or recorded.observation_digest_sha256
                            != call.observation_digest_sha256
                            or call.decision is None
                            or content_digest(dict(recorded.outcome))
                            != call.decision.decision_digest_sha256
                        ):
                            raise AlphaRefusal("evidence_causal_replay_mismatch")
                    point = next(
                        point
                        for point in arms[item["arm"]].points
                        if point.point_id == item["point_id"]
                    )
                    playback = RecordedOutcomeAgent(tape, agent_id="alpha-audit-playback")
                    playback.begin_episode({})
                    decisions = []
                    session = point.new_retrieval_session()
                    while playback.cursor < len(tape):
                        observation = replace(
                            session.current_observation(),
                            operation_instance_id=ledger_record.header.operation_instance_id,
                        )
                        decision = playback.decide(observation)
                        decisions.append(decision)
                        if isinstance(decision, RetrieveBatch):
                            session.submit(decision)
                    playback.check_exhausted()
                    derived = (
                        run_boundary(spec, point, decisions).vector_status
                        if item["arm"] == ARM_BOUNDARY
                        else run_static(point, decisions).vector_status
                    )
                    point_index = POINT_IDS.index(item["point_id"])
                    if (
                        derived != tape_doc["local_replay_status"]
                        or derived != item["status"]
                        or derived != row["vector"][point_index]
                        or tape_doc["provider_calls_during_replay"] != 0
                    ):
                        raise AlphaRefusal("evidence_causal_replay_mismatch")
                else:
                    artifact_path = directory / item["artifact"]
                    record = read_artifact(artifact_path)
                    artifact_operation_id = record["operation_instance_id"]
                    artifact_agent_id = record["agent_id"]
                    if (
                        artifact_operation_id in full_artifact_operation_ids
                        or artifact_agent_id in full_artifact_agent_ids
                        or artifact_operation_id
                        != ledger_record.header.operation_instance_id
                        or artifact_agent_id != ledger_record.header.agent_id
                    ):
                        raise AlphaRefusal("duplicate_evidence_identity")
                    full_artifact_operation_ids.add(artifact_operation_id)
                    full_artifact_agent_ids.add(artifact_agent_id)
                    bundle = audit_execution_bundle_files(artifact_path, ledger)
                    replay = replay_artifact(spec, record)
                    derived_vector = list(
                        map_full_outcome(
                            arms[ARM_FULL].points, replay.validated_episode_outcome
                        )
                    )
                    if (
                        not bundle.ok
                        or not replay.ok
                        or replay.reproduction != "playback"
                        or item["replay_provider_calls"] != 0
                        or record["status"] != item["status"]
                        or derived_vector != item["vector"]
                        or derived_vector != row["vector"]
                    ):
                        raise AlphaRefusal("evidence_causal_replay_mismatch")
            except AlphaRefusal:
                raise
            except Exception as exc:
                raise AlphaRefusal("evidence_causal_replay_mismatch") from exc
        summary_path = directory / PUBLIC_SUMMARY_NAME
        try:
            summary_info = summary_path.lstat()
            summary_doc = json.loads(summary_path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AlphaRefusal("public_summary_file_binding_mismatch") from exc
        if (
            value["public_summary_name"] != PUBLIC_SUMMARY_NAME
            or not stat.S_ISREG(summary_info.st_mode)
            or stat.S_IMODE(summary_info.st_mode) != 0o600
            or summary_doc != _historical_public_summary(value)
            or _sha_file(summary_path) != value["public_summary_digest_sha256"]
        ):
            raise AlphaRefusal("public_summary_file_binding_mismatch")
    return value


def _historical_public_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "operatebench.matched_arm_alpha_public_summary.v1",
        "experiment": result["plan"]["experiment"],
        "external_source_revision": result["external_source_revision"],
        "operator_source_digest_sha256": result["operator_source_digest_sha256"],
        "plan_digest_sha256": result["plan_digest_sha256"],
        "schedule": result["schedule"],
        "vectors": [
            {"trial": row["trial"], "arm": row["arm"], "vector": row["vector"]}
            for row in result["trials"]
        ],
        "pass_counts": result["pass_counts"],
        "accounting": result["accounting"],
        "execution_integrity": result["execution_integrity"],
        "differential_verdict": result["differential_verdict"],
    }


def public_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse obsolete public scientific summary generation."""

    raise AlphaRefusal("alpha_operator_quarantined")


def _offline_test_result_fixture() -> dict[str, Any]:
    synthetic_source_revision = _sha_bytes(b"matched Alpha synthetic fixture source")[:40]
    expected_settings_digest = _expected_provider_identity()["settings_digest_sha256"]
    entries: list[dict[str, Any]] = []
    rows = []
    for trial, arm in AUTHORED_SCHEDULE:
        count = 1 if arm == ARM_FULL else 5
        cell_entries = []
        for index in range(count):
            point_id = None if arm == ARM_FULL else POINT_IDS[index]
            prefix = (
                f"full-t{trial}"
                if arm == ARM_FULL
                else f"control-t{trial}-{arm}-{point_id}"
            )
            item = {
                "kind": "full" if arm == ARM_FULL else "control",
                "trial": trial,
                "arm": arm,
                "point_id": point_id,
                "ledger": f"{prefix}.ledger.ndjson",
                "ledger_digest_sha256": _sha_bytes(
                    f"synthetic fixture ledger {prefix}".encode()
                ),
                "request_digests_sha256": ["0" * 64],
                "model": MODEL,
                "profile_id": PROFILE_ID,
                "settings_digest_sha256": expected_settings_digest,
                "provider_calls": 1,
                "provider_attempts": 1,
                "input_tokens": 1,
                "output_tokens": 1,
                "retries": 0,
                "measured_cost_usd": "0.000002",
                "status": ADMISSIBLE,
                "replay_ok": True,
            }
            if arm == ARM_FULL:
                item.update(
                    {
                        "artifact": f"{prefix}.artifact.json",
                        "artifact_digest_sha256": _sha_bytes(
                            f"synthetic fixture artifact {prefix}".encode()
                        ),
                        "artifact_version": ARTIFACT_VERSION,
                        "vector": [ADMISSIBLE] * 5,
                        "bundle_audit_ok": True,
                        "replay_provider_calls": 0,
                    }
                )
            else:
                item.update(
                    {"tape": f"{prefix}.tape.json", "tape_digest_sha256": "0" * 64}
                )
            entries.append(item)
            cell_entries.append(item)
        rows.append(
            {
                "trial": trial,
                "arm": arm,
                "point_ids": list(POINT_IDS),
                "vector": [ADMISSIBLE] * 5,
                "pass": True,
                "calls": count,
                "faults": [],
                "evidence_digests": [
                    item.get("artifact_digest_sha256", item.get("tape_digest_sha256"))
                    for item in cell_entries
                ],
            }
        )
    plan = plan_payload()
    result = {
        "schema": RESULT_SCHEMA,
        "mode": MODE_OFFLINE,
        "external_provider_calls": 0,
        "external_source_revision": synthetic_source_revision,
        "operator_source_digest_sha256": OPERATOR_SOURCE_DIGEST_SHA256,
        "plan": plan,
        "plan_digest_sha256": compute_plan_digest(plan),
        "preregistration": {
            "authorization_digest_sha256": _sha_bytes(
                authorization_value(synthetic_source_revision).encode()
            ),
            "expected_source_revision": synthetic_source_revision,
        },
        "scenario": {
            "scenario_id": SCENARIO_ID,
            "semantic_scenario_id": SEMANTIC_SCENARIO_ID,
            "semantic_scenario_digest_sha256": SEMANTIC_SCENARIO_DIGEST_SHA256,
            "spec_digest_sha256": SPEC_DIGEST_SHA256,
        },
        "compiler": {"identity": COMPILER_ID},
        "provider": {
            "provider": PROVIDER,
            "api": API,
            "model": MODEL,
            "profile_id": PROFILE_ID,
            "protocol_version": MODEL_PROTOCOL_VERSION,
            "request_mapping": LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION,
        },
        "scaffold": {
            "identity": SCAFFOLD_ID,
            "ordered_is_memory_evidence": False,
        },
        "schedule": [list(item) for item in AUTHORED_SCHEDULE],
        "trials": rows,
        "pass_counts": dict.fromkeys(ARM_NAMES, 3),
        "evidence_manifest_name": MANIFEST_NAME,
        "evidence_manifest_digest_sha256": "0" * 64,
        "evidence_manifest": entries,
        "public_summary_name": PUBLIC_SUMMARY_NAME,
        "public_summary_digest_sha256": "",
        "accounting": {
            "subject_calls": 48,
            "provider_calls": 48,
            "provider_attempts": 48,
            "input_tokens": 48,
            "output_tokens": 48,
            "retries": 0,
            "measured_cost_usd": "0.000096",
            "hard_call_cap": 510,
            "combined_token_cap": 14027550,
            "cost_cap_usd": "5",
        },
        "execution_integrity": {
            "complete": True,
            "no_replacement": True,
            "control_replays_ok": True,
            "full_bundle_audits_ok": True,
            "full_replays_ok": True,
        },
        "differential_verdict": NO_SUPPORTED_LONGITUDINAL_DIFFERENTIAL,
    }
    result["public_summary_digest_sha256"] = _sha_bytes(
        _json_output_bytes(_historical_public_summary(result))
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Historical diagnostic interface for the quarantined matched-arm Alpha; "
            "the sealed Alpha is permanently UNINTERPRETABLE, with no rerun or regrade."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--offline-preflight",
        action="store_true",
        help="obsolete historical selector; execution remains quarantined",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="obsolete historical selector; live execution is quarantined",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="legacy argument; the quarantined command never reads or writes this path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    summary = {
        "ok": False,
        "mode": "historical_quarantine",
        "failure": {"code": "alpha_operator_quarantined"},
    }
    print(json.dumps(summary, sort_keys=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
