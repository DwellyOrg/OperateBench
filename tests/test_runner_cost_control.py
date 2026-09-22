"""A whole capped run: measurement, exposure, stopping and resume.

Every run here is executed by the real runner, through the real Anthropic
adapter, over the in-process ``MockTransport``. No socket is opened.

The helpers at the top are shared with :mod:`tests.test_run_limit_identity`, which
tests the same run directories from the manifest and ledger side.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from boundarybench.adapter import build_turn_request
from boundarybench.budget import CostControls, RunCostGuard
from boundarybench.environment import Environment
from boundarybench.ledger import (
    OUTCOME_COST_CAP_REACHED,
    OUTCOME_MODEL_PROTOCOL_FAILURE,
)
from boundarybench.pricing import price_for
from boundarybench.providers.anthropic_messages import (
    ANTHROPIC_ADAPTER_VERSION,
    ANTHROPIC_IMPLEMENTATION,
    ANTHROPIC_PROVIDER,
    MAX_OUTPUT_TOKENS,
    AnthropicMessagesAdapter,
    AnthropicRetryPolicy,
    build_messages_request,
    request_input_token_bound,
)
from boundarybench.runmanifest import RunLimits, build_run_manifest, open_run_session
from boundarybench.runner import RunReport, execute_run
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.suite import validate_suite
from tests.anthropic_transport import (
    RecordingTransport,
    error_body,
    message_body,
    text_block,
    tool_use_block,
)
from tests.conftest import SUITE_MANIFEST

SONNET = "claude-sonnet-5"
MAX_TURNS = 4

STOP_CALL = tool_use_block(
    "complete_case",
    {
        "disposition": "STOP",
        "primary_reason_code": "NO_APPLICABLE_RULE",
        "secondary_reason_codes": [],
        "evidence_refs": [],
    },
)


def _controls(cap: str | None, episodes: int | None) -> CostControls:
    return CostControls(
        max_cost_usd=None if cap is None else Decimal(cap),
        max_episodes=episodes,
        price=None if cap is None else price_for(provider="anthropic", model=SONNET),
    )


def first_request_bound() -> int:
    """The conservative input bound of the run's very first request.

    Computed the same way the adapter computes it, from the same scaffold and
    the same first variant, so a test can state a cap in terms of a request that
    is about to be made rather than in terms of a magic number.
    """
    suite = validate_suite(SUITE_MANIFEST)
    variant = suite.cubes[0].cube.variants[0]
    request = build_turn_request(
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        environment=Environment(variant),
        turns_remaining=MAX_TURNS,
    )
    return request_input_token_bound(build_messages_request(request, model=SONNET))


def first_request_reservation() -> Decimal:
    return price_for(provider="anthropic", model=SONNET).cost(
        input_tokens=first_request_bound(), output_tokens=MAX_OUTPUT_TOKENS
    )


def capped_run(
    root: Path,
    *,
    cap: str = "100",
    episodes: int | None = 12,
    steps: Any = (),
    default: Any = None,
    guard: RunCostGuard | None = None,
    no_guard: bool = False,
    tighten_episodes: int | None = None,
) -> tuple[RunReport, RecordingTransport, RunCostGuard | None]:
    """One capped Anthropic run against the in-process transport.

    ``tighten_episodes`` rewrites the built manifest's episode ceiling below its
    own plan and rehashes it, which ``build_run_manifest`` refuses to produce.
    It exists to reach the runner's own execution ceiling — the backstop behind
    the plan check — and nothing else uses it.

    ``no_guard`` executes a capped manifest with no guard at all, which is the
    other half of the binding this build refuses; nothing but that probe uses it.
    """
    import dataclasses

    from boundarybench.runmanifest import configuration_digest, execution_digest_of

    suite = validate_suite(SUITE_MANIFEST)
    scaffold = load_scaffold(STANDARD_SCAFFOLD)
    controls = _controls(cap, episodes)
    active = guard
    if active is None and controls.enforces_cost and not no_guard:
        # Built from the controls the manifest will actually record, tightened
        # ceiling included: the runner binds the guard to the manifest's own
        # block before it dispatches anything, so a guard enforcing the untightened
        # limits would be refused before this probe reached what it is probing.
        active = RunCostGuard(
            controls=(
                controls if tighten_episodes is None else _controls(cap, tighten_episodes)
            ),
            max_output_tokens=MAX_OUTPUT_TOKENS,
            max_attempts_per_turn=AnthropicRetryPolicy().max_attempts,
        )
    transport = RecordingTransport(
        list(steps),
        default=default
        if default is not None
        else message_body([STOP_CALL], model=SONNET),
    )
    adapter = AnthropicMessagesAdapter(
        model=SONNET,
        client=transport.client(),
        cost_guard=active,
        sleep=lambda seconds: None,
    )
    manifest = build_run_manifest(
        suite=suite,
        scaffold=scaffold,
        provider=ANTHROPIC_PROVIDER,
        model=SONNET,
        implementation=ANTHROPIC_IMPLEMENTATION,
        adapter_version=ANTHROPIC_ADAPTER_VERSION,
        adapter_settings=adapter.settings,
        trials=1,
        limits=RunLimits(
            max_turns=MAX_TURNS, max_messages=16, episode_timeout_seconds=30.0
        ),
        cost_controls=controls,
    )
    if tighten_episodes is not None:
        tightened = dataclasses.replace(
            manifest,
            cost_controls=_controls(cap, tighten_episodes),
            configuration_id="",
        )
        identity = configuration_digest(tightened.configuration_payload())
        manifest = dataclasses.replace(
            tightened,
            configuration_id=identity,
            execution_digest=execution_digest_of(
                configuration_id=identity,
                execution_id=tightened.execution_id,
                created_at_utc=tightened.created_at_utc,
            ),
        )
    with open_run_session(root, manifest) as session:
        report = execute_run(
            suite=suite,
            scaffold=scaffold,
            adapter=adapter,
            session=session,
            cost_guard=active,
        )
    return report, transport, active


def run_one_capped_run(root: Path, *, episodes: int | None = 12) -> RunReport:
    report, _transport, _guard = capped_run(root, episodes=episodes)
    return report


def run_until_cap_exhausted(root: Path) -> RunReport:
    """A run whose first request is unaffordable, so nothing is ever sent."""
    report, _transport, _guard = capped_run(root, cap="0.000001")
    return report


# -- measured cost -----------------------------------------------------------


def test_a_capped_run_prices_every_episode_it_measured(tmp_path: Path) -> None:
    report = run_one_capped_run(tmp_path / "run")
    payload = report.as_dict()

    assert report.completed_count == 12
    # 137 input and 29 output tokens per episode at USD 2 / USD 10 per million.
    assert payload["cost_accounting"]["measured_usd"] == "0.006768"
    assert payload["cost_accounting"]["exposure_usd"] == "0"
    assert payload["cost_accounting"]["max_cost_usd"] == "100"
    assert payload["usage_measured_episodes"]["cost_usd"] == 12


def test_each_row_carries_its_own_measured_cost_and_zero_exposure(
    tmp_path: Path,
) -> None:
    """Zero exposure is a measurement; null exposure would be its absence."""
    root = tmp_path / "run"
    run_one_capped_run(root)
    rows = [
        json.loads(line)
        for line in (root / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert all(row["usage"]["cost_usd"] == pytest.approx(0.000564) for row in rows)
    assert all(row["cost_exposure_usd"] == 0 for row in rows)


def test_a_run_without_a_cap_records_null_cost_and_null_exposure(
    tmp_path: Path,
) -> None:
    """Absence, not zero: this run pinned no price and measured no cost."""
    root = tmp_path / "run"
    report, _transport, guard = capped_run(root, cap=None, episodes=None)
    assert guard is None

    rows = [
        json.loads(line)
        for line in (root / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["usage"]["cost_usd"] is None for row in rows)
    assert all(row["cost_exposure_usd"] is None for row in rows)
    assert report.as_dict()["cost_accounting"] is None


def test_a_protocol_invalid_answer_still_records_what_it_consumed(
    tmp_path: Path,
) -> None:
    """The tokens were spent whether or not this build could read the answer."""
    root = tmp_path / "run"
    report, _transport, guard = capped_run(
        root,
        default=message_body(
            [text_block("prose, not a tool call")],
            model=SONNET,
            input_tokens=91,
            output_tokens=92,
        ),
    )

    payload = report.as_dict()
    assert payload["outcome_counts"] == {OUTCOME_MODEL_PROTOCOL_FAILURE: 12}
    assert payload["cost_accounting"]["measured_usd"] == "0.013224"
    assert guard.exposure_usd == Decimal(0)


# -- fail closed -------------------------------------------------------------


def test_a_cap_that_cannot_pay_for_the_first_request_sends_nothing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    report, transport, _guard = capped_run(root, cap="0.000001")

    assert transport.calls == 0
    assert report.completed_count == 1
    assert report.run_terminal_failure is not None
    assert report.run_terminal_failure.outcome == OUTCOME_COST_CAP_REACHED
    # And the plan stops rather than recording eleven identical refusals.
    assert report.executed_count == 1


def test_a_cap_that_exactly_covers_one_request_permits_exactly_one(
    tmp_path: Path,
) -> None:
    """Boundary equality at run level: the authorised request is made."""
    root = tmp_path / "run"
    report, transport, _guard = capped_run(root, cap=str(first_request_reservation()))

    assert transport.calls == 1
    assert report.run_terminal_failure is not None
    assert report.run_terminal_failure.outcome == OUTCOME_COST_CAP_REACHED


def test_a_cap_one_unit_below_that_permits_none(tmp_path: Path) -> None:
    root = tmp_path / "run"
    report, transport, _guard = capped_run(
        root, cap=str(first_request_reservation() - Decimal("0.000001"))
    )

    assert transport.calls == 0
    assert report.completed_count == 1


def test_exposure_from_retried_faults_stops_the_run_before_the_plan_ends(
    tmp_path: Path,
) -> None:
    """Retries are charged against the budget as conservative exposure."""
    root = tmp_path / "run"
    report, transport, guard = capped_run(
        root,
        cap="0.15",
        steps=[(500, error_body())] * 40,
    )

    assert 0 < report.executed_count < 12
    assert report.run_terminal_failure is not None
    assert report.run_terminal_failure.outcome == OUTCOME_COST_CAP_REACHED
    assert guard.exposure_usd > 0
    assert guard.measured_usd == Decimal(0)
    assert transport.calls < 12 * AnthropicRetryPolicy().max_attempts


def test_the_stopping_row_carries_no_provider_text(tmp_path: Path) -> None:
    root = tmp_path / "run"
    capped_run(root, cap="0.000001")
    row = json.loads(
        (root / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )

    assert row["error_class"] == "cost_cap_exhausted"
    assert "USD" in row["error_detail"]
    assert "test-key-not-a-credential" not in row["error_detail"]
    assert "provider.invalid" not in row["error_detail"]


def test_the_executed_episode_ceiling_stops_a_plan_that_exceeds_it(
    tmp_path: Path,
) -> None:
    """The backstop behind the plan check, reached through a rehashed manifest.

    ``build_run_manifest`` refuses a plan larger than the ceiling, so this
    rewrites the ceiling underneath a plan that was already fixed. The two
    checks answer different questions — what was authorised, and what actually
    ran — and only the second one is a fact about execution.
    """
    root = tmp_path / "run"
    report, transport, _guard = capped_run(root, tighten_episodes=3)

    assert report.executed_count == 3
    assert report.planned_count == 12
    assert transport.calls == 3
    assert report.as_dict()["limits_stop"]["reason"] == "episode_limit"


def test_a_stored_plan_larger_than_its_ceiling_is_refused_on_the_next_read(
    tmp_path: Path,
) -> None:
    """Tamper behaviour: the ceiling is re-derived against the plan on load.

    The run above executed under a rehashed manifest whose plan exceeds its own
    ceiling, and the manifest is now on disk. Reading it back refuses it rather
    than continuing under a ceiling the plan contradicts — the digest agrees
    with the file, so the check is against the two stored fields themselves.
    """
    from boundarybench.runmanifest import RunManifestError, load_run_manifest

    root = tmp_path / "run"
    capped_run(root, tighten_episodes=3)

    with pytest.raises(RunManifestError) as refused:
        load_run_manifest(root / "run_manifest.json")
    assert "authorised for at most 3" in str(refused.value)


# -- resume ------------------------------------------------------------------


def test_a_resumed_run_cannot_forget_the_spend_its_ledger_records(
    tmp_path: Path,
) -> None:
    """The whole point of durable evidence: the second invocation starts owing."""
    root = tmp_path / "run"
    run_one_capped_run(root)

    fresh = RunCostGuard(
        controls=_controls("100", 12),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_attempts_per_turn=AnthropicRetryPolicy().max_attempts,
    )
    report, transport, _guard = capped_run(root, guard=fresh)

    assert transport.calls == 0
    assert report.resumed_count == 12
    assert fresh.measured_usd == Decimal("0.000564") * 12
    assert fresh.remaining_usd == Decimal("100") - Decimal("0.000564") * 12


def test_a_resumed_run_cannot_forget_the_exposure_its_ledger_records(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    # Every episode's turn fails three times, so every row carries exposure and
    # no row carries a measurement.
    first, _transport, before = capped_run(
        root, cap="100", steps=[(500, error_body())] * 200
    )
    assert first.completed_count == 12

    fresh = RunCostGuard(
        controls=_controls("100", 12),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_attempts_per_turn=AnthropicRetryPolicy().max_attempts,
    )
    capped_run(root, guard=fresh, steps=[(500, error_body())] * 200)

    assert before.exposure_usd > 0
    assert fresh.exposure_usd == before.exposure_usd
    assert fresh.measured_usd == Decimal(0)


def test_the_reported_totals_are_the_ledgers_and_not_the_guards(
    tmp_path: Path,
) -> None:
    """A resume that executed nothing still reports what the run has spent."""
    root = tmp_path / "run"
    run_one_capped_run(root)
    report = run_one_capped_run(root)

    assert report.executed_count == 0
    assert report.as_dict()["cost_accounting"]["measured_usd"] == "0.006768"
