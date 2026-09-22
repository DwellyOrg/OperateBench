"""Provider methodology invariants.

Four distinctions are exercised through durable records:

* configuration identity versus execution identity;
* provider attempt and measurement evidence;
* infrastructure failures versus completed semantic samples;
* output-token truncation versus malformed protocol.

Every provider path here runs through the real Anthropic SDK over an in-process
``httpx.MockTransport``. Nothing in this file opens a socket or needs a
credential.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest

from boundarybench.adapter import ADAPTER_CONTRACT_VERSION
from boundarybench.runmanifest import (
    MANIFEST_FILENAME,
    RunLimits,
    RunManifestError,
    RunManifestFormatError,
    RunManifestMismatchError,
    build_run_manifest,
    load_run_manifest,
    open_run_session,
)
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.suite import validate_suite
from tests.conftest import SUITE_MANIFEST


def at(second: int) -> datetime:
    """A stated instant, so a record's timestamp is an assertion not an artefact."""
    return datetime(2026, 8, 8, 12, 0, second, 500000, tzinfo=UTC)


def execution_id(tag: str) -> str:
    """A canonical version-4 UUID whose random field is chosen by the test."""
    return f"1f0d7a3c-{tag}-4a1b-9c2d-000000000001"


def build_manifest(**overrides: object):
    kwargs: dict[str, object] = {
        "suite": validate_suite(SUITE_MANIFEST),
        "scaffold": load_scaffold(STANDARD_SCAFFOLD),
        "provider": "test-double",
        "model": "scripted-test-double",
        "implementation": "scripted_test_double",
        "adapter_version": ADAPTER_CONTRACT_VERSION,
        "adapter_settings": {"temperature": 0},
        "trials": 1,
        "limits": RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0),
    }
    kwargs.update(overrides)
    return build_run_manifest(**kwargs)  # type: ignore[arg-type]


# -- configuration identity is not execution identity -----------------------


def test_configuration_id_is_deterministic_and_execution_id_is_not() -> None:
    """The same plan is the same configuration and never the same execution.

    Both halves matter. A deterministic ``configuration_id`` is what makes a
    resume checkable — the directory can prove the request has not changed —
    and a unique ``execution_id`` is what stops two independent invocations of
    the same plan being recorded as one run with no way to tell their evidence
    apart.
    """
    first = build_manifest()
    second = build_manifest()

    assert first.configuration_id == second.configuration_id
    assert len(first.configuration_id) == 64
    assert first.execution_id != second.execution_id


def test_configuration_id_excludes_the_wall_clock() -> None:
    """When a run started is execution evidence, never configuration identity."""
    payload = build_manifest().configuration_payload()
    rendered = repr(payload)

    assert "created_at" not in rendered
    assert "execution_id" not in rendered


def test_resume_reuses_the_stored_execution_and_discards_the_candidate(
    tmp_path: Path,
) -> None:
    """A fresh candidate id must never make an exact resume impossible.

    The second invocation mints its own execution id and timestamp before it can
    know whether a run directory already exists, because the manifest is built
    first. If that candidate were kept, resuming would either be refused — the
    stored and requested manifests differ — or would append rows under a second
    execution id into one execution's ledger. It is discarded instead: the run
    directory's stored execution is the execution, and this invocation joins it.
    """
    first = build_manifest(
        now=lambda: at(1), execution_id_factory=lambda: execution_id("aaaa")
    )
    with open_run_session(tmp_path / "run", first) as session:
        assert session.resumed is False
        assert session.manifest.execution_id == execution_id("aaaa")

    second = build_manifest(
        now=lambda: at(9), execution_id_factory=lambda: execution_id("bbbb")
    )
    with open_run_session(tmp_path / "run", second) as session:
        assert session.resumed is True
        assert session.manifest.execution_id == execution_id("aaaa")
        assert session.manifest.created_at_utc == "2026-08-08T12:00:01.500000Z"
        assert session.manifest.configuration_id == second.configuration_id


def test_a_second_directory_is_the_same_configuration_and_a_new_execution(
    tmp_path: Path,
) -> None:
    """Same plan, two directories: one configuration id, two execution ids."""
    first = build_manifest(
        now=lambda: at(1), execution_id_factory=lambda: execution_id("aaaa")
    )
    second = build_manifest(
        now=lambda: at(2), execution_id_factory=lambda: execution_id("bbbb")
    )
    with (
        open_run_session(tmp_path / "one", first) as left,
        open_run_session(tmp_path / "two", second) as right,
    ):
        assert left.manifest.configuration_id == right.manifest.configuration_id
        assert left.manifest.execution_id != right.manifest.execution_id


def _stored(root: Path) -> dict[str, Any]:
    return json.loads((root / MANIFEST_FILENAME).read_text(encoding="utf-8"))


def _rewrite(root: Path, payload: dict[str, Any]) -> None:
    (root / MANIFEST_FILENAME).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("execution_id", "1f0d7a3c-cccc-4a1b-9c2d-000000000001"),
        ("created_at_utc", "2020-01-01T00:00:00.000000Z"),
        ("configuration_id", "0" * 64),
    ],
)
def test_edited_execution_fields_are_refused(
    tmp_path: Path, field: str, value: str
) -> None:
    """Move one execution fact and the binding that ties it to the others breaks."""
    root = tmp_path / "run"
    manifest = build_manifest(
        now=lambda: at(1), execution_id_factory=lambda: execution_id("aaaa")
    )
    with open_run_session(root, manifest):
        pass

    payload = _stored(root)
    payload["execution"][field] = value
    _rewrite(root, payload)

    with pytest.raises(RunManifestError):
        load_run_manifest(root / MANIFEST_FILENAME)


def test_a_recomputed_execution_digest_still_cannot_relabel_the_configuration(
    tmp_path: Path,
) -> None:
    """The execution block names the configuration the rest of the file states.

    Recomputing the execution digest after an edit is trivial — it is unkeyed
    and its inputs are all in the file. What that cannot do is make the
    execution block agree with the configuration id the manifest actually
    hashes to, which is why the two are cross-checked rather than each merely
    checked against itself.
    """
    from boundarybench.runmanifest import execution_digest_of

    root = tmp_path / "run"
    manifest = build_manifest(
        now=lambda: at(1), execution_id_factory=lambda: execution_id("aaaa")
    )
    with open_run_session(root, manifest):
        pass

    payload = _stored(root)
    forged = "1" * 64
    payload["execution"]["configuration_id"] = forged
    payload["execution"]["execution_digest"] = execution_digest_of(
        configuration_id=forged,
        execution_id=payload["execution"]["execution_id"],
        created_at_utc=payload["execution"]["created_at_utc"],
    )
    _rewrite(root, payload)

    with pytest.raises(RunManifestMismatchError):
        load_run_manifest(root / MANIFEST_FILENAME)


def test_a_naive_creation_timestamp_is_refused() -> None:
    """ "Assume UTC" is right on a server and silently wrong on a laptop."""
    with pytest.raises(RunManifestFormatError):
        build_manifest(now=lambda: datetime(2026, 8, 8, 12, 0, 0))


# -- an episode records when it ran and how long it took ---------------------


def stepping_clock(*values: float):
    """A monotonic clock that reads a stated sequence, then holds its last value."""
    remaining = list(values)
    last = [values[-1]]

    def read() -> float:
        if remaining:
            last[0] = remaining.pop(0)
        return last[0]

    return read


def test_an_episode_records_its_own_start_end_and_measured_duration(cube) -> None:
    """When an episode ran, and how long it took, measured here rather than inferred.

    The two clocks answer different questions and neither can answer the other's.
    The wall clock says *when*, which is what makes a resume that spans a
    provider date boundary readable afterwards; it can also jump backwards. The
    monotonic clock says *how long*, which is the only duration a local build can
    honestly measure, and it cannot be turned into a date.

    Only the *start* is a wall-clock observation. The end is derived from it and
    the measured duration, which is why the second moment this fixture offers is
    never read; see ``tests/test_provider_methodology_integrity.py`` for that separation.
    """
    variant = cube.cell_variant("S0_P0")
    moments = iter([at(1), at(4)])

    result = run_scripted_episode(
        variant, wall_clock=lambda: next(moments), clock=stepping_clock(0.0, 2.5)
    )

    assert result.started_at_utc == "2026-08-08T12:00:01.500000Z"
    assert result.completed_at_utc == "2026-08-08T12:00:04.000000Z"
    assert result.elapsed_seconds == pytest.approx(2.5)


# -- what the provider was actually asked, and what came back ----------------


PINNED_MODEL = "claude-test-20990101"
#: The output ceiling this build pins, restated rather than imported: a test
#: that read the constant would agree with any value it was moved to.
MAX_OUTPUT_TOKENS_PINNED = 1024


def anthropic_adapter(client, **kwargs):
    from boundarybench.providers.anthropic_messages import AnthropicMessagesAdapter

    kwargs.setdefault("sleep", lambda seconds: None)
    return AnthropicMessagesAdapter(model=PINNED_MODEL, client=client, **kwargs)


def turn_request(cube):
    from boundarybench.adapter import build_turn_request
    from boundarybench.environment import Environment

    return build_turn_request(
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        environment=Environment(cube.cell_variant("S0_P0")),
        turns_remaining=12,
    )


def deadline(seconds: float = 30.0):
    from boundarybench.adapter import TurnDeadline

    return TurnDeadline(remaining_seconds=seconds, cancelled=lambda: False)


def ticking(step: float = 0.25):
    now = [0.0]

    def read() -> float:
        value = now[0]
        now[0] += step
        return value

    return read


def test_a_success_after_retries_preserves_every_preceding_transport_fault(cube) -> None:
    """A successful row preserves the transport faults before the response.

    A turn that is rate limited twice and then answered is not the same
    evidence as a turn that answered first time: it cost three requests, two
    backoffs and a stretch of the episode's wall-clock budget, and an operator
    reading a clean success would have no way to see the provider was throttling
    the run. The attempts are therefore accumulated across the whole turn and
    survive the success.
    """
    from tests.anthropic_transport import (
        error_body,
        message_body,
        scripted_client,
        tool_use_block,
    )

    _transport, client = scripted_client(
        (429, error_body()),
        (503, error_body()),
        message_body([tool_use_block("read_records", {})], model=PINNED_MODEL),
    )
    from boundarybench.providers.anthropic_messages import AnthropicRetryPolicy

    adapter = anthropic_adapter(
        client,
        retry=AnthropicRetryPolicy(max_attempts=3, initial_backoff_seconds=0.0),
        clock=ticking(),
    )

    adapter.next_call(turn_request(cube), deadline())
    telemetry = adapter.last_telemetry()

    assert telemetry is not None
    assert telemetry.attempt_count == 3
    assert [attempt.fault for attempt in telemetry.attempts] == [
        "provider_rate_limited",
        "provider_server_error",
        None,
    ]
    assert [attempt.http_status for attempt in telemetry.attempts] == [429, 503, None]
    assert [attempt.response_received for attempt in telemetry.attempts] == [
        True,
        True,
        True,
    ]
    assert telemetry.usage_reported is True
    assert telemetry.turn_latency_seconds is not None


def test_exhausted_status_responses_still_measure_what_the_turn_spent(cube) -> None:
    """Error bodies arrived, but no accepted answer supplied token measurements.

    Reporting nothing at all for a fully failed turn made the two indistinguishable
    from a turn that never happened. The attempts, the faults and the locally
    measured latency are all real observations that survive the failure; the
    token counts stay null, because null is "not measured" and this build never
    renders it as zero.
    """
    from boundarybench.adapter import AdapterProviderError
    from boundarybench.providers.anthropic_messages import AnthropicRetryPolicy
    from tests.anthropic_transport import error_body, scripted_client

    _transport, client = scripted_client(
        (503, error_body()), (503, error_body()), (503, error_body())
    )
    adapter = anthropic_adapter(
        client,
        retry=AnthropicRetryPolicy(max_attempts=3, initial_backoff_seconds=0.0),
        clock=ticking(),
    )

    with pytest.raises(AdapterProviderError):
        adapter.next_call(turn_request(cube), deadline())

    telemetry = adapter.last_telemetry()
    assert telemetry is not None
    assert telemetry.attempt_count == 3
    assert telemetry.response_received is True
    assert telemetry.usage_reported is False
    assert telemetry.turn_latency_seconds is not None
    assert adapter.last_usage().input_tokens is None


def test_telemetry_is_cleared_at_the_top_of_every_turn(cube) -> None:
    """A measurement that outlives the call it measured is stale evidence."""
    from boundarybench.adapter import AdapterProviderError
    from boundarybench.providers.anthropic_messages import AnthropicRetryPolicy
    from tests.anthropic_transport import (
        error_body,
        message_body,
        scripted_client,
        tool_use_block,
    )

    _transport, client = scripted_client(
        (429, error_body()),
        message_body([tool_use_block("read_records", {})], model=PINNED_MODEL),
        (401, error_body()),
    )
    adapter = anthropic_adapter(
        client,
        retry=AnthropicRetryPolicy(max_attempts=3, initial_backoff_seconds=0.0),
        clock=ticking(),
    )
    request = turn_request(cube)

    adapter.next_call(request, deadline())
    assert adapter.last_telemetry().attempt_count == 2

    with pytest.raises(AdapterProviderError):
        adapter.next_call(request, deadline())

    second = adapter.last_telemetry()
    assert second.attempt_count == 1
    assert second.attempts[0].fault == "provider_authentication"


def test_provider_telemetry_never_carries_provider_text(cube) -> None:
    """A closed structure, so nothing from the wire can reach a durable row."""
    from boundarybench.adapter import AdapterProviderError
    from tests.anthropic_transport import error_body, scripted_client

    secret = "sk-ant-should-never-appear"
    _transport, client = scripted_client((401, error_body(message=secret)))
    adapter = anthropic_adapter(client, clock=ticking())

    with pytest.raises(AdapterProviderError):
        adapter.next_call(turn_request(cube), deadline())

    rendered = json.dumps(adapter.last_telemetry().as_dict())
    assert secret not in rendered
    assert "sk-ant" not in rendered


# -- truncation is a budget limit, not a malformed answer --------------------


def test_max_tokens_is_an_output_limit_and_not_a_protocol_failure(cube) -> None:
    """A cut-off answer is not a model that cannot follow a schema.

    ``stop_reason='max_tokens'`` says the model was still speaking when this
    build's own pinned output ceiling stopped it. Filing that as a protocol
    failure blamed the model for a limit the run chose, and buried a
    configuration signal — raise the ceiling — inside the bucket that is
    supposed to mean "the model answered badly".
    """
    from boundarybench.adapter import AdapterOutputLimitError, AdapterProtocolError
    from tests.anthropic_transport import message_body, scripted_client, tool_use_block

    _transport, client = scripted_client(
        message_body(
            [tool_use_block("read_records", {})],
            stop_reason="max_tokens",
            model=PINNED_MODEL,
        )
    )
    adapter = anthropic_adapter(client, clock=ticking())

    with pytest.raises(AdapterOutputLimitError) as raised:
        adapter.next_call(turn_request(cube), deadline())

    assert not isinstance(raised.value, AdapterProtocolError)
    assert "max_tokens" not in str(raised.value)
    assert str(MAX_OUTPUT_TOKENS_PINNED) in str(raised.value)


def test_a_truncated_answer_with_no_tool_call_is_still_an_output_limit(cube) -> None:
    """Truncation is decided before the call count, because it explains it."""
    from boundarybench.adapter import AdapterOutputLimitError
    from tests.anthropic_transport import message_body, scripted_client, text_block

    _transport, client = scripted_client(
        message_body(
            [text_block("I was still thinking when")],
            stop_reason="max_tokens",
            model=PINNED_MODEL,
        )
    )
    adapter = anthropic_adapter(client, clock=ticking())

    with pytest.raises(AdapterOutputLimitError):
        adapter.next_call(turn_request(cube), deadline())


def test_an_unknown_stop_reason_stays_a_redacted_protocol_failure(cube) -> None:
    """Only the one named reason moves; everything else keeps its old, safe home."""
    from boundarybench.adapter import AdapterOutputLimitError, AdapterProtocolError
    from tests.anthropic_transport import message_body, scripted_client, tool_use_block

    _transport, client = scripted_client(
        message_body(
            [tool_use_block("read_records", {})],
            stop_reason="refusal",
            model=PINNED_MODEL,
        )
    )
    adapter = anthropic_adapter(client, clock=ticking())

    with pytest.raises(AdapterProtocolError) as raised:
        adapter.next_call(turn_request(cube), deadline())

    assert not isinstance(raised.value, AdapterOutputLimitError)
    assert "refusal" not in str(raised.value)


def test_a_truncated_answer_is_still_paid_for(cube) -> None:
    """The provider produced those tokens; the ceiling is this build's choice."""
    from boundarybench.adapter import AdapterOutputLimitError
    from tests.anthropic_transport import message_body, scripted_client, tool_use_block

    _transport, client = scripted_client(
        message_body(
            [tool_use_block("read_records", {})],
            stop_reason="max_tokens",
            input_tokens=91,
            output_tokens=1024,
            model=PINNED_MODEL,
        )
    )
    adapter = anthropic_adapter(client, clock=ticking())

    with pytest.raises(AdapterOutputLimitError):
        adapter.next_call(turn_request(cube), deadline())

    assert adapter.last_usage().output_tokens == 1024
    assert adapter.last_telemetry().usage_reported is True


def run_scripted_episode(variant, script=None, **kwargs):
    from boundarybench.adapter import (
        ScriptedTestAdapter,
        identity_for_test_double,
        policy_following_script,
    )
    from boundarybench.runner import run_episode

    adapter = kwargs.pop(
        "adapter",
        ScriptedTestAdapter(
            script=script or policy_following_script,
            identity=identity_for_test_double(),
            settings={},
        ),
    )
    return run_episode(
        variant=variant,
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        adapter=adapter,
        limits=kwargs.pop(
            "limits",
            RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0),
        ),
        **kwargs,
    )


# -- whole runs, through the real SDK over an in-process transport -----------


def anthropic_manifest(retry=None, **overrides):
    """A manifest that pins exactly the settings the adapter under test reports.

    The retry policy is inside adapter settings and so inside configuration
    identity, which is why it has to be threaded through here rather than
    defaulted: a manifest pinning one policy while the adapter runs another is a
    provenance mismatch, and the runner refuses it.
    """
    from boundarybench.providers.anthropic_messages import (
        ANTHROPIC_ADAPTER_VERSION,
        ANTHROPIC_IMPLEMENTATION,
        ANTHROPIC_PROVIDER,
        AnthropicRetryPolicy,
        anthropic_settings,
    )

    return build_manifest(
        provider=ANTHROPIC_PROVIDER,
        model=PINNED_MODEL,
        implementation=ANTHROPIC_IMPLEMENTATION,
        adapter_version=ANTHROPIC_ADAPTER_VERSION,
        adapter_settings=anthropic_settings(
            retry or AnthropicRetryPolicy(), model=PINNED_MODEL
        ),
        **overrides,
    )


def instant_retry():
    """The shipped policy with its waits removed, so tests do not sleep."""
    from boundarybench.providers.anthropic_messages import AnthropicRetryPolicy

    return AnthropicRetryPolicy(max_attempts=3, initial_backoff_seconds=0.0)


def execute(root: Path, manifest, adapter, **kwargs):
    from boundarybench.runner import execute_run

    with open_run_session(root, manifest) as session:
        return execute_run(
            suite=validate_suite(SUITE_MANIFEST),
            scaffold=load_scaffold(STANDARD_SCAFFOLD),
            adapter=adapter,
            session=session,
            **kwargs,
        )


def ledger_rows(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (root / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]


# -- a row says which execution produced it, and when ------------------------


def test_every_row_carries_its_execution_and_the_instant_the_episode_ran(
    tmp_path: Path,
) -> None:
    """Configuration on the row is not enough to tell two executions apart."""
    from boundarybench.adapter import PolicyFollowingFakeAdapter, fake_identity

    identity = fake_identity()
    manifest = build_manifest(
        provider=identity.provider,
        model=identity.model,
        implementation=identity.implementation,
        adapter_version=identity.version,
        adapter_settings={},
        now=lambda: at(1),
        execution_id_factory=lambda: execution_id("aaaa"),
    )
    root = tmp_path / "run"
    report = execute(root, manifest, PolicyFollowingFakeAdapter(settings={}))

    assert report.completed_count == 12
    for row in ledger_rows(root):
        assert row["configuration_id"] == manifest.configuration_id
        assert row["execution_id"] == execution_id("aaaa")
        assert row["started_at_utc"] <= row["completed_at_utc"]
        assert row["elapsed_seconds"] >= 0
        assert "run_id" not in row


# -- an infrastructure failure is not a completed sample ---------------------


def test_a_refused_credential_records_one_ineligible_attempt_and_stops(
    tmp_path: Path,
) -> None:
    """Twelve identical auth failures are one fact recorded twelve times.

    A refused credential is a property of the run's configuration, not of the
    episode that met it, so the thirteenth attempt fails exactly as the first
    did. The first is recorded durably — that evidence is the point — and the
    run stops rather than spending the plan and eleven more round-trips to learn
    nothing.
    """
    from boundarybench.ledger import (
        OUTCOME_PROVIDER_AUTH_FAILURE,
        SAMPLE_INFRASTRUCTURE_INELIGIBLE,
    )
    from tests.anthropic_transport import RecordingTransport, error_body

    transport = RecordingTransport([(401, error_body())] * 12)
    adapter = anthropic_adapter(transport.client(), clock=ticking())
    root = tmp_path / "run"

    report = execute(root, anthropic_manifest(), adapter)

    assert transport.calls == 1
    assert report.completed_count == 1
    assert report.planned_count == 12
    (row,) = ledger_rows(root)
    assert row["outcome"] == OUTCOME_PROVIDER_AUTH_FAILURE
    assert row["sample_status"] == SAMPLE_INFRASTRUCTURE_INELIGIBLE
    assert report.sample_counts["completed_semantic_sample"] == 0
    assert report.run_terminal_failure is not None


def test_a_terminated_run_refuses_to_continue_in_the_same_execution(
    tmp_path: Path,
) -> None:
    """Fixing the credential does not change the configuration, so nothing else can.

    Whatever an operator changes to make the run work — a credential, an
    environment — is deliberately outside configuration identity, so the
    directory would happily resume and the resulting ledger would hold one
    execution whose first row says the credential was refused and whose next
    eleven say it was not. That is not one execution's evidence.
    """
    from boundarybench.runner import RunTerminatedError
    from tests.anthropic_transport import RecordingTransport, error_body

    root = tmp_path / "run"
    manifest = anthropic_manifest()
    first = RecordingTransport([(401, error_body())] * 12)
    execute(root, manifest, anthropic_adapter(first.client(), clock=ticking()))

    second = RecordingTransport([(401, error_body())] * 12)
    with pytest.raises(RunTerminatedError) as raised:
        execute(root, manifest, anthropic_adapter(second.client(), clock=ticking()))

    assert second.calls == 0
    assert "output directory" in str(raised.value)
    # The evidence that was already recorded is untouched.
    assert len(ledger_rows(root)) == 1


def test_a_transient_provider_failure_is_durable_ineligible_and_not_rerun(
    tmp_path: Path,
) -> None:
    """A spent slot stays spent, and never counts as a completed sample.

    Silently re-running it would turn one outage into a second sample of the
    same episode, which is the thing this design cannot measure. Scheduling an
    explicit replacement is a later feature; until it exists the honest record
    is a plan slot consumed by an ineligible attempt, and the report says so.
    """
    from boundarybench.ledger import (
        OUTCOME_PROVIDER_FAILURE,
        SAMPLE_INFRASTRUCTURE_INELIGIBLE,
    )
    from tests.anthropic_transport import (
        RecordingTransport,
        error_body,
        message_body,
        tool_use_block,
    )

    root = tmp_path / "run"
    manifest = anthropic_manifest(retry=instant_retry())
    transport = RecordingTransport(
        [(503, error_body()), (503, error_body()), (503, error_body())],
        default=message_body([tool_use_block("read_records", {})], model=PINNED_MODEL),
    )
    report = execute(
        root,
        manifest,
        anthropic_adapter(transport.client(), retry=instant_retry(), clock=ticking()),
    )

    first = ledger_rows(root)[0]
    assert first["outcome"] == OUTCOME_PROVIDER_FAILURE
    assert first["sample_status"] == SAMPLE_INFRASTRUCTURE_INELIGIBLE
    assert first["provider_telemetry"]["attempt_count"] == 3
    assert first["provider_telemetry"]["fault_counts"] == {"provider_server_error": 3}
    assert first["usage"]["input_tokens"] is None
    assert first["measurement"]["status"] == "full"
    assert report.completed_count == 12

    # Re-running the same configuration in the same directory is a no-op: the
    # failed episode has a row, and a row means done.
    replay = RecordingTransport(
        [],
        default=message_body([tool_use_block("read_records", {})], model=PINNED_MODEL),
    )
    again = execute(
        root,
        manifest,
        anthropic_adapter(replay.client(), retry=instant_retry(), clock=ticking()),
    )
    assert replay.calls == 0
    assert again.executed_count == 0
    assert ledger_rows(root)[0] == first


def test_a_success_after_retries_is_a_completed_sample_that_keeps_its_faults(
    tmp_path: Path,
) -> None:
    """A completed row retains the rate limits that preceded its response."""
    from boundarybench.ledger import OUTCOME_SUCCESS, SAMPLE_COMPLETED
    from tests.anthropic_transport import (
        RecordingTransport,
        error_body,
        message_body,
        tool_use_block,
    )

    root = tmp_path / "run"
    ok = message_body([tool_use_block("read_records", {})], model=PINNED_MODEL)
    transport = RecordingTransport([(429, error_body()), (429, error_body())], default=ok)
    report = execute(
        root,
        anthropic_manifest(retry=instant_retry()),
        anthropic_adapter(transport.client(), retry=instant_retry(), clock=ticking()),
    )

    first = ledger_rows(root)[0]
    telemetry = first["provider_telemetry"]
    assert telemetry["attempt_count"] > 3
    assert telemetry["fault_counts"] == {"provider_rate_limited": 2}
    assert telemetry["turns"][0]["attempt_count"] == 3
    assert [a["fault"] for a in telemetry["turns"][0]["attempts"]] == [
        "provider_rate_limited",
        "provider_rate_limited",
        None,
    ]
    # It never reaches a decision — the scripted provider only ever reads — so
    # it is a behavioural attempt rather than a completed sample. What matters
    # here is that the faults survived whatever the outcome turned out to be.
    assert first["outcome"] != OUTCOME_SUCCESS
    assert report.provider_fault_counts == {"provider_rate_limited": 2}
    assert report.sample_counts[SAMPLE_COMPLETED] == 0


def test_an_output_limit_row_is_a_behavioural_attempt_and_not_a_protocol_failure(
    tmp_path: Path,
) -> None:
    """Output truncation has its own outcome and eligibility end to end."""
    from boundarybench.ledger import (
        OUTCOME_MODEL_OUTPUT_LIMIT,
        SAMPLE_BEHAVIORAL_ATTEMPT,
    )
    from tests.anthropic_transport import (
        RecordingTransport,
        message_body,
        tool_use_block,
    )

    root = tmp_path / "run"
    transport = RecordingTransport(
        [],
        default=message_body(
            [tool_use_block("read_records", {})],
            stop_reason="max_tokens",
            model=PINNED_MODEL,
        ),
    )
    report = execute(
        root, anthropic_manifest(), anthropic_adapter(transport.client(), clock=ticking())
    )

    first = ledger_rows(root)[0]
    assert first["outcome"] == OUTCOME_MODEL_OUTPUT_LIMIT
    assert first["sample_status"] == SAMPLE_BEHAVIORAL_ATTEMPT
    assert first["failure_event"]["kind"] == "output_token_limit"
    # The truncated answer was produced and paid for.
    assert first["usage"]["output_tokens"] == 29
    assert report.outcome_counts[OUTCOME_MODEL_OUTPUT_LIMIT] == 12
    assert report.sample_counts[SAMPLE_BEHAVIORAL_ATTEMPT] == 12


def test_a_broken_telemetry_accessor_never_replaces_the_real_failure(cube) -> None:
    """Precedence is one-directional: the fault that ended the turn always wins."""
    from boundarybench.adapter import (
        AdapterCall,
        AdapterProtocolError,
        AdapterUsage,
        identity_for_test_double,
    )
    from boundarybench.ledger import OUTCOME_MODEL_PROTOCOL_FAILURE

    class BrokenTelemetry:
        identity = identity_for_test_double()
        settings: ClassVar[dict[str, Any]] = {}

        def next_call(self, request, deadline) -> AdapterCall:
            raise AdapterProtocolError("the answer was not one call")

        def last_usage(self) -> AdapterUsage:
            return AdapterUsage()

        def last_telemetry(self):
            raise RuntimeError("the telemetry accessor is broken")

    result = run_scripted_episode(cube.cell_variant("S0_P0"), adapter=BrokenTelemetry())

    assert result.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert result.provider_telemetry is None


def test_the_report_distinguishes_full_partial_and_unmeasured_coverage(
    tmp_path: Path,
) -> None:
    """A fake measures nothing, and "nothing" is never rendered as zero."""
    from boundarybench.adapter import PolicyFollowingFakeAdapter, fake_identity

    identity = fake_identity()
    manifest = build_manifest(
        provider=identity.provider,
        model=identity.model,
        implementation=identity.implementation,
        adapter_version=identity.version,
        adapter_settings={},
    )
    report = execute(tmp_path / "run", manifest, PolicyFollowingFakeAdapter(settings={}))
    payload = report.as_dict()

    coverage = payload["measurement_coverage"]
    assert coverage["episodes"] == {"full": 0, "partial": 0, "unmeasured": 12}
    assert coverage["turns"]["measured"] == 0
    assert coverage["attempts"]["total"] is None
    assert payload["provider_attempts"]["attempts"] is None
    assert payload["provider_fault_counts"] == {}
    assert payload["sample_counts"]["completed_semantic_sample"] == 12
    assert payload["sample_eligibility_note"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("execution_id", "1f0d7a3c-cccc-4a1b-9c2d-000000000001"),
        ("started_at_utc", "not-a-timestamp"),
        ("completed_at_utc", "2000-01-01T00:00:00.000000Z"),
        ("elapsed_seconds", -1.0),
    ],
)
def test_a_tampered_row_timestamp_or_execution_id_fails_closed(
    tmp_path: Path, field: str, value: object
) -> None:
    """Evidence about when a run happened is checked, not merely stored."""
    from boundarybench.adapter import PolicyFollowingFakeAdapter, fake_identity
    from boundarybench.ledger import LedgerError, read_ledger
    from boundarybench.runner import compiled_variants

    identity = fake_identity()
    manifest = build_manifest(
        provider=identity.provider,
        model=identity.model,
        implementation=identity.implementation,
        adapter_version=identity.version,
        adapter_settings={},
        now=lambda: at(1),
        execution_id_factory=lambda: execution_id("aaaa"),
    )
    root = tmp_path / "run"
    execute(root, manifest, PolicyFollowingFakeAdapter(settings={}))

    rows = ledger_rows(root)
    rows[0][field] = value
    (root / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(LedgerError):
        read_ledger(
            root / "episodes.jsonl",
            manifest,
            compiled_variants(validate_suite(SUITE_MANIFEST)),
        )
