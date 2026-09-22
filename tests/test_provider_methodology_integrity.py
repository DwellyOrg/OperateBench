"""Provider methodology integrity invariants.

Two distinctions are exercised with adversarial inputs:

* completion is derived from the recorded start and monotonic duration, even if
  a wall clock steps backwards;
* request rejection ends an episode, while configuration failure can stop a run.

Every provider path here runs through the real installed Anthropic SDK over an
in-process ``httpx.MockTransport``. Nothing in this file opens a socket or reads
a credential.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from boundarybench.adapter import ADAPTER_CONTRACT_VERSION
from boundarybench.runmanifest import RunLimits, build_run_manifest, open_run_session
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.suite import validate_suite
from tests.conftest import SUITE_MANIFEST

PINNED_MODEL = "claude-test-20990101"


def at(second: int, *, minute: int = 0) -> datetime:
    """A stated instant, so a record's timestamp is an assertion not an artefact."""
    return datetime(2026, 8, 8, 12, minute, second, 500000, tzinfo=UTC)


def parse(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%f%z")


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


def anthropic_manifest(retry=None, **overrides: object):
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


def anthropic_adapter(client, **kwargs: Any):
    from boundarybench.providers.anthropic_messages import AnthropicMessagesAdapter

    kwargs.setdefault("sleep", lambda seconds: None)
    kwargs.setdefault("clock", ticking())
    return AnthropicMessagesAdapter(model=PINNED_MODEL, client=client, **kwargs)


def ticking(step: float = 0.25):
    """A monotonic clock that advances a fixed step on every reading."""
    now = [0.0]

    def read() -> float:
        value = now[0]
        now[0] += step
        return value

    return read


def stepping_clock(*values: float):
    """A monotonic clock that reads the given values, then repeats the last."""
    remaining = list(values)
    last = [values[-1]]

    def read() -> float:
        if remaining:
            last[0] = remaining.pop(0)
        return last[0]

    return read


def execute(root: Path, manifest, adapter, **kwargs: Any):
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


def run_scripted_episode(variant, **kwargs: Any):
    from boundarybench.adapter import (
        ScriptedTestAdapter,
        identity_for_test_double,
        policy_following_script,
    )
    from boundarybench.runner import run_episode

    adapter = kwargs.pop(
        "adapter",
        ScriptedTestAdapter(
            script=policy_following_script,
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


# -- completion is derived, never a second wall-clock reading ----------------


def test_completion_is_the_recorded_start_plus_the_monotonic_elapsed(cube) -> None:
    """One wall-clock reading per episode, and the end is arithmetic from it.

    ``completed_at_utc`` is not an observation: it is ``started_at_utc`` plus the
    monotonic duration the episode measured. A second wall-clock reading would be
    a second observation of a quantity that can move between the two, and the
    record would then be a subtraction of two different clocks.
    """
    variant = cube.cell_variant("S0_P0")
    reads: list[int] = []

    def wall_clock() -> datetime:
        reads.append(len(reads))
        return at(1)

    result = run_scripted_episode(
        variant, wall_clock=wall_clock, clock=stepping_clock(0.0, 2.5)
    )

    assert reads == [0], "the wall clock is read once, at the start, and never again"
    assert result.started_at_utc == "2026-08-08T12:00:01.500000Z"
    assert result.completed_at_utc == "2026-08-08T12:00:04.000000Z"
    assert result.elapsed_seconds == pytest.approx(2.5)


def test_a_wall_clock_stepped_backwards_cannot_invert_one_episode(cube) -> None:
    """The correction lands on the *next* episode's start, never on this one's end."""
    variant = cube.cell_variant("S0_P0")
    moments = iter([at(30), at(2), at(1)])

    result = run_scripted_episode(
        variant, wall_clock=lambda: next(moments), clock=stepping_clock(0.0, 1.25)
    )

    assert result.started_at_utc == "2026-08-08T12:00:30.500000Z"
    assert result.completed_at_utc == "2026-08-08T12:00:31.750000Z"
    assert parse(result.completed_at_utc) - parse(result.started_at_utc) == timedelta(
        seconds=result.elapsed_seconds
    )


def test_a_zero_duration_episode_completes_at_the_instant_it_started(cube) -> None:
    """A duration of zero is a real measurement, and stays a readable row."""
    variant = cube.cell_variant("S0_P0")

    result = run_scripted_episode(
        variant, wall_clock=lambda: at(7), clock=stepping_clock(0.0)
    )

    assert result.elapsed_seconds == 0.0
    assert result.completed_at_utc == result.started_at_utc


def test_a_backwards_wall_clock_run_appends_reads_and_resumes(tmp_path: Path) -> None:
    """End to end: NTP steps the clock back mid-run and the evidence still reads.

    The whole point of deriving the end from the start is that the ledger's own
    ordering check — an episode does not finish before it starts — stays true of
    every row under a clock correction, so the run appends, the reader accepts,
    and the directory resumes. The duration is not hidden while that happens: it
    is the monotonic measurement, and the derived end is exactly that far after
    the recorded start.
    """
    from boundarybench.adapter import PolicyFollowingFakeAdapter, fake_identity

    # A clock that jumps thirty seconds backwards after the first episode and
    # then runs forwards again, which is what a correction actually looks like.
    moments = iter(
        [at(30), at(0), *(at(second) for second in range(1, 11)), at(11), at(12)]
    )
    identity = fake_identity()
    root = tmp_path / "run"
    manifest = build_manifest(
        provider=identity.provider,
        model=identity.model,
        implementation=identity.implementation,
        adapter_version=identity.version,
        adapter_settings={},
    )

    report = execute(
        root,
        manifest,
        PolicyFollowingFakeAdapter(settings={}),
        wall_clock=lambda: next(moments),
        clock=ticking(),
    )

    assert report.completed_count == 12
    assert report.succeeded_count == 12
    rows = ledger_rows(root)
    assert [row["started_at_utc"] for row in rows][1] < rows[0]["started_at_utc"], (
        "the fixture must actually step the wall clock backwards mid-run"
    )
    for row in rows:
        assert row["elapsed_seconds"] > 0
        assert row["completed_at_utc"] >= row["started_at_utc"]
        assert parse(row["completed_at_utc"]) - parse(row["started_at_utc"]) == timedelta(
            seconds=row["elapsed_seconds"]
        )

    # The directory reads back and resumes: twelve rows, nothing re-executed.
    again = execute(
        root,
        manifest,
        PolicyFollowingFakeAdapter(settings={}),
        wall_clock=lambda: at(59),
        clock=ticking(),
    )
    assert again.executed_count == 0
    assert again.resumed_count == 12
    assert ledger_rows(root) == rows


# -- a refused payload is not a condemned configuration ---------------------


def test_only_authentication_configuration_and_a_spent_budget_stop_the_run() -> None:
    """The closed set, stated: a request rejection is an episode's fact.

    The third member is the run's own authorised cost budget being spent, which
    meets the same test as the other two: what is left of a budget does not
    grow, so the arithmetic that refused this request refuses every remaining
    one. It is the only one of the three that is not fixed by an operator action
    outside configuration identity — a larger cap *is* a different run.

    The fourth is that budget being *passed*: a response measuring more than the
    reservation that authorised it. It meets the same test from the other side —
    the run is already beyond what it authorised, and every remaining episode
    would be spending past a cap that has been reached rather than up to one.
    """
    from boundarybench.ledger import (
        OUTCOME_COST_CAP_REACHED,
        OUTCOME_COST_RESERVATION_BREACHED,
        OUTCOME_PROVIDER_AUTH_FAILURE,
        OUTCOME_PROVIDER_CONFIGURATION_FAILURE,
        OUTCOME_PROVIDER_REQUEST_FAILURE,
        RUN_TERMINAL_OUTCOMES,
    )

    assert set(RUN_TERMINAL_OUTCOMES) == {
        OUTCOME_PROVIDER_AUTH_FAILURE,
        OUTCOME_PROVIDER_CONFIGURATION_FAILURE,
        OUTCOME_COST_CAP_REACHED,
        OUTCOME_COST_RESERVATION_BREACHED,
    }
    assert OUTCOME_PROVIDER_REQUEST_FAILURE not in RUN_TERMINAL_OUTCOMES


def test_a_configuration_fault_is_an_ineligible_sample_and_its_own_outcome() -> None:
    """A missing model is not an outage, and neither is a completed sample."""
    from boundarybench.adapter import PROVIDER_FAULT_CONFIGURATION, PROVIDER_FAULTS
    from boundarybench.ledger import (
        OUTCOME_PROVIDER_CONFIGURATION_FAILURE,
        SAMPLE_INFRASTRUCTURE_INELIGIBLE,
        classify_failure,
        sample_status_for,
    )

    assert PROVIDER_FAULT_CONFIGURATION in PROVIDER_FAULTS
    outcome, error_class, _detail, event = classify_failure(
        phase="adapter_call", kind=PROVIDER_FAULT_CONFIGURATION, detail="fixed sentence"
    )

    assert outcome == OUTCOME_PROVIDER_CONFIGURATION_FAILURE
    assert error_class == PROVIDER_FAULT_CONFIGURATION
    assert event["kind"] == PROVIDER_FAULT_CONFIGURATION
    assert sample_status_for(outcome) == SAMPLE_INFRASTRUCTURE_INELIGIBLE


@pytest.mark.parametrize("status", [400, 413, 422])
def test_a_request_specific_refusal_is_not_a_configuration_fault(status: int) -> None:
    """The payload this turn sent is not the run's pinned model."""
    from boundarybench.adapter import (
        PROVIDER_FAULT_REQUEST_REJECTED,
        AdapterProviderError,
    )
    from tests.anthropic_transport import error_body, scripted_client

    _transport, client = scripted_client((status, error_body()))
    adapter = anthropic_adapter(client)

    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_turn_request(), _deadline())

    assert raised.value.fault == PROVIDER_FAULT_REQUEST_REJECTED


def test_a_missing_model_or_resource_is_the_configuration_fault() -> None:
    """404 is the status that proves the run's own pinning is wrong."""
    from boundarybench.adapter import PROVIDER_FAULT_CONFIGURATION, AdapterProviderError
    from tests.anthropic_transport import error_body, scripted_client

    _transport, client = scripted_client((404, error_body("not_found_error", "no model")))
    adapter = anthropic_adapter(client)

    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_turn_request(), _deadline())

    assert raised.value.fault == PROVIDER_FAULT_CONFIGURATION
    detail = str(raised.value)
    assert "404" in detail
    assert "no model" not in detail
    assert "not_found_error" not in detail


def _turn_request():
    from boundarybench.adapter import build_turn_request
    from boundarybench.environment import Environment
    from boundarybench.runner import compiled_variants

    variants = compiled_variants(validate_suite(SUITE_MANIFEST))
    return build_turn_request(
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        environment=Environment(next(iter(variants.values()))),
        turns_remaining=12,
    )


def _deadline(seconds: float = 30.0):
    from boundarybench.adapter import TurnDeadline

    return TurnDeadline(remaining_seconds=seconds, cancelled=lambda: False)


def _read_records_default():
    from tests.anthropic_transport import message_body, tool_use_block

    return message_body([tool_use_block("read_records", {})], model=PINNED_MODEL)


def test_an_oversized_payload_fails_its_episode_and_the_plan_continues(
    tmp_path: Path,
) -> None:
    """413 is about the request this episode sent, so the other eleven still run.

    Treating it as run-terminal spent the whole plan on one episode's payload:
    eleven planned episodes were never attempted, the directory could never
    continue, and the evidence recorded a condemned configuration that was not
    condemned.
    """
    from boundarybench.ledger import (
        OUTCOME_PROVIDER_REQUEST_FAILURE,
        SAMPLE_INFRASTRUCTURE_INELIGIBLE,
    )
    from tests.anthropic_transport import RecordingTransport, error_body

    transport = RecordingTransport(
        [(413, error_body("request_too_large", "payload too big"))],
        default=_read_records_default(),
    )
    root = tmp_path / "run"
    manifest = anthropic_manifest()

    report = execute(root, manifest, anthropic_adapter(transport.client()))

    rows = ledger_rows(root)
    assert len(rows) == 12
    assert rows[0]["outcome"] == OUTCOME_PROVIDER_REQUEST_FAILURE
    assert rows[0]["sample_status"] == SAMPLE_INFRASTRUCTURE_INELIGIBLE
    assert rows[0]["error_class"] == "provider_request_rejected"
    assert "413" in rows[0]["error_detail"]
    assert "payload too big" not in rows[0]["error_detail"]
    assert "request_too_large" not in rows[0]["error_detail"]
    assert {row["outcome"] for row in rows[1:]} == {"limit_exhausted"}
    assert report.run_terminal_failure is None
    assert report.as_dict()["run_terminated"] is None
    assert transport.calls > 1

    # The spent slot stays spent, and the directory resumes rather than refusing.
    replay = RecordingTransport([], default=_read_records_default())
    again = execute(root, manifest, anthropic_adapter(replay.client()))
    assert replay.calls == 0
    assert again.executed_count == 0
    assert again.resumed_count == 12


def test_a_refused_credential_records_one_row_and_refuses_continuation(
    tmp_path: Path,
) -> None:
    """Authentication stays run-terminal: the thirteenth key is the first key."""
    from boundarybench.ledger import OUTCOME_PROVIDER_AUTH_FAILURE
    from boundarybench.runner import RunTerminatedError
    from tests.anthropic_transport import RecordingTransport, error_body

    transport = RecordingTransport([(401, error_body())] * 12)
    root = tmp_path / "run"
    manifest = anthropic_manifest()

    report = execute(root, manifest, anthropic_adapter(transport.client()))

    assert transport.calls == 1
    (row,) = ledger_rows(root)
    assert row["outcome"] == OUTCOME_PROVIDER_AUTH_FAILURE
    assert report.as_dict()["run_terminated"]["outcome"] == OUTCOME_PROVIDER_AUTH_FAILURE

    second = RecordingTransport([], default=_read_records_default())
    with pytest.raises(RunTerminatedError):
        execute(root, manifest, anthropic_adapter(second.client()))
    assert second.calls == 0
    assert len(ledger_rows(root)) == 1


def test_a_configuration_fault_records_one_row_and_refuses_continuation(
    tmp_path: Path,
) -> None:
    """A model the account cannot reach condemns every remaining episode too.

    This is the one request rejection that is proven to be about the run rather
    than the payload: the model id is pinned in configuration identity and sent
    identically on every turn, so a provider that says the resource does not
    exist has answered for the whole plan.
    """
    from boundarybench.ledger import (
        OUTCOME_PROVIDER_CONFIGURATION_FAILURE,
        SAMPLE_INFRASTRUCTURE_INELIGIBLE,
    )
    from boundarybench.runner import RunTerminatedError
    from tests.anthropic_transport import RecordingTransport, error_body

    transport = RecordingTransport(
        [(404, error_body("not_found_error", "model claude-test-20990101 not found"))]
        * 12
    )
    root = tmp_path / "run"
    manifest = anthropic_manifest()

    report = execute(root, manifest, anthropic_adapter(transport.client()))

    assert transport.calls == 1
    (row,) = ledger_rows(root)
    assert row["outcome"] == OUTCOME_PROVIDER_CONFIGURATION_FAILURE
    assert row["sample_status"] == SAMPLE_INFRASTRUCTURE_INELIGIBLE
    assert row["error_class"] == "provider_configuration"
    assert row["failure_event"]["kind"] == "provider_configuration"
    assert "404" in row["error_detail"]
    assert "not found" not in row["error_detail"]
    assert "not_found_error" not in row["error_detail"]
    assert report.sample_counts["completed_semantic_sample"] == 0
    terminated = report.as_dict()["run_terminated"]
    assert terminated["outcome"] == OUTCOME_PROVIDER_CONFIGURATION_FAILURE
    assert terminated["episode_id"] == row["episode_id"]

    second = RecordingTransport([], default=_read_records_default())
    with pytest.raises(RunTerminatedError) as raised:
        execute(root, manifest, anthropic_adapter(second.client()))
    assert second.calls == 0
    assert "output directory" in str(raised.value)
    assert len(ledger_rows(root)) == 1


def test_the_ledger_refuses_a_configuration_row_relabelled_as_a_rejection(
    tmp_path: Path,
) -> None:
    """The outcome follows from the recorded fault, on every read."""
    from boundarybench.ledger import (
        OUTCOME_PROVIDER_REQUEST_FAILURE,
        LedgerError,
        read_ledger,
    )
    from boundarybench.runner import compiled_variants
    from tests.anthropic_transport import RecordingTransport, error_body

    transport = RecordingTransport([(404, error_body())] * 12)
    root = tmp_path / "run"
    manifest = anthropic_manifest()
    execute(root, manifest, anthropic_adapter(transport.client()))

    ledger = root / "episodes.jsonl"
    (row,) = ledger_rows(root)
    row["outcome"] = OUTCOME_PROVIDER_REQUEST_FAILURE
    ledger.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(LedgerError) as raised:
        read_ledger(ledger, manifest, compiled_variants(validate_suite(SUITE_MANIFEST)))

    assert "provider_configuration" in str(raised.value)
