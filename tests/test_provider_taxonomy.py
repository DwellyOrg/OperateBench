"""Provider faults, as recorded outcomes rather than as "the adapter raised".

The initial adapter contract had one bucket for everything a provider integration
could do wrong. That was sufficient while the only adapter was an in-process fake
that could not fail in any of these ways. Once a service sits behind the boundary,
a refused credential, a misconfigured model id, a rate limit and an outage are four
different operator actions, and a run whose ledger cannot tell them apart is a
run nobody can act on.

The tests here are about the taxonomy, not about Anthropic: they drive the
faults through the contract's :class:`AdapterProviderError`, so they hold for
every provider adapter that arrives later.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from boundarybench.adapter import (
    ADAPTER_CONTRACT_VERSION,
    PROVIDER_FAULT_AUTHENTICATION,
    PROVIDER_FAULT_NETWORK_ERROR,
    PROVIDER_FAULT_RATE_LIMITED,
    PROVIDER_FAULT_REQUEST_REJECTED,
    PROVIDER_FAULT_RESPONSE_INVALID,
    PROVIDER_FAULT_SERVER_ERROR,
    PROVIDER_FAULT_TIMEOUT,
    AdapterCall,
    AdapterProviderError,
    ScriptedTestAdapter,
    TurnRequest,
    identity_for_test_double,
)
from boundarybench.ledger import (
    OUTCOME_PROVIDER_AUTH_FAILURE,
    OUTCOME_PROVIDER_FAILURE,
    OUTCOME_PROVIDER_RATE_LIMITED,
    OUTCOME_PROVIDER_REQUEST_FAILURE,
    LedgerError,
    append_episode,
    read_ledger,
    sample_status_for,
)
from boundarybench.runmanifest import RunLimits, build_run_manifest, open_run_session
from boundarybench.runner import (
    build_episode_record,
    compiled_variants,
    run_episode,
)
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.suite import validate_suite
from tests.conftest import SUITE_MANIFEST

STABLE_DETAIL = (
    "the provider answered HTTP 429 (provider_rate_limited) after 3 attempt(s)"
)


def _manifest(trials: int = 1):
    return build_run_manifest(
        suite=validate_suite(SUITE_MANIFEST),
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        provider="test-double",
        model="scripted-test-double",
        implementation="scripted_test_double",
        adapter_version=ADAPTER_CONTRACT_VERSION,
        adapter_settings={},
        trials=trials,
        limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0),
    )


def _raising_adapter(fault: str, detail: str = STABLE_DETAIL) -> ScriptedTestAdapter:
    def script(request: TurnRequest) -> AdapterCall:
        raise AdapterProviderError(fault, detail)

    return ScriptedTestAdapter(
        script=script, identity=identity_for_test_double(), settings={}
    )


def _episode(fault: str, detail: str = STABLE_DETAIL):
    suite = validate_suite(SUITE_MANIFEST)
    variant = next(iter(compiled_variants(suite).values()))
    return (
        run_episode(
            variant=variant,
            scaffold=load_scaffold(STANDARD_SCAFFOLD),
            adapter=_raising_adapter(fault, detail),
            limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0),
        ),
        variant,
    )


@pytest.mark.parametrize(
    ("fault", "outcome"),
    [
        (PROVIDER_FAULT_AUTHENTICATION, OUTCOME_PROVIDER_AUTH_FAILURE),
        (PROVIDER_FAULT_REQUEST_REJECTED, OUTCOME_PROVIDER_REQUEST_FAILURE),
        (PROVIDER_FAULT_RATE_LIMITED, OUTCOME_PROVIDER_RATE_LIMITED),
        (PROVIDER_FAULT_SERVER_ERROR, OUTCOME_PROVIDER_FAILURE),
        (PROVIDER_FAULT_NETWORK_ERROR, OUTCOME_PROVIDER_FAILURE),
        (PROVIDER_FAULT_TIMEOUT, OUTCOME_PROVIDER_FAILURE),
        (PROVIDER_FAULT_RESPONSE_INVALID, OUTCOME_PROVIDER_FAILURE),
    ],
)
def test_each_provider_fault_is_its_own_recorded_outcome(
    fault: str, outcome: str
) -> None:
    """A wrong key and an outage are not the same operator action."""
    result, _variant = _episode(fault)

    assert result.outcome == outcome
    # The class stays the fault itself, so the four that share an outcome are
    # still told apart in a run report's error-class counts.
    assert result.error_class == fault
    assert result.failure_event is not None
    assert result.failure_event["phase"] == "adapter_call"
    assert result.failure_event["kind"] == fault
    assert result.failure_event["detail"] == STABLE_DETAIL
    assert result.evaluation is None


def test_a_provider_fault_records_no_exception_type_or_attempted_call() -> None:
    """The row carries the classified fault, not the integration's internals.

    ``AdapterProviderError`` is this build's own class, so recording its name
    would say nothing an operator can use, and the call was never made, so there
    is no attempted call to record either.
    """
    result, _variant = _episode(PROVIDER_FAULT_TIMEOUT)

    assert result.failure_event is not None
    assert result.failure_event["exception_type"] is None
    assert result.failure_event["attempted_call"] is None
    assert result.failure_event["response_type"] is None


def test_a_provider_fault_leaves_one_unanswered_request(tmp_path: Path) -> None:
    """It is raised out of the call, so the request was never answered."""
    result, _variant = _episode(PROVIDER_FAULT_SERVER_ERROR)

    assert result.turns_used == 1
    assert result.messages_used == 1
    assert result.trajectory.steps == ()


def _completed(tmp_path: Path, fault: str) -> tuple[Any, ...]:
    manifest = _manifest()
    suite = validate_suite(SUITE_MANIFEST)
    variants = compiled_variants(suite)
    entry = manifest.episode_plan[0]
    variant = variants[entry.variant_id]
    result = run_episode(
        variant=variant,
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        adapter=_raising_adapter(fault),
        limits=manifest.limits,
    )
    record = build_episode_record(manifest, entry, variant, result)
    with open_run_session(tmp_path / "run", manifest) as session:
        append_episode(session.paths.ledger_path, record)
        rows = read_ledger(session.paths.ledger_path, manifest, variants)
    return rows


def test_a_provider_fault_row_survives_the_ledgers_own_validation(
    tmp_path: Path,
) -> None:
    """The new shapes are readable evidence, not just writable ones."""
    rows = _completed(tmp_path, PROVIDER_FAULT_RATE_LIMITED)

    assert len(rows) == 1
    assert rows[0].outcome == OUTCOME_PROVIDER_RATE_LIMITED
    assert rows[0].error_class == PROVIDER_FAULT_RATE_LIMITED
    assert rows[0].error_detail == STABLE_DETAIL


def _rewrite(path: Path, edit: Mapping[str, Any]) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0].update(edit)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
        )
    )


def test_a_provider_outcome_that_does_not_follow_from_its_fault_is_refused(
    tmp_path: Path,
) -> None:
    """The taxonomy is fail-closed for the new outcomes too."""
    manifest = _manifest()
    variants = compiled_variants(validate_suite(SUITE_MANIFEST))
    entry = manifest.episode_plan[0]
    result = run_episode(
        variant=variants[entry.variant_id],
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        adapter=_raising_adapter(PROVIDER_FAULT_RATE_LIMITED),
        limits=manifest.limits,
    )
    record = build_episode_record(manifest, entry, variants[entry.variant_id], result)
    with open_run_session(tmp_path / "run", manifest) as session:
        append_episode(session.paths.ledger_path, record)
        ledger = session.paths.ledger_path
        # A rate limit relabelled as an authentication failure: the same row,
        # with an outcome its own recorded cause does not imply.
        _rewrite(ledger, {"outcome": OUTCOME_PROVIDER_AUTH_FAILURE})
        with pytest.raises(LedgerError) as raised:
            read_ledger(ledger, manifest, variants)

    assert "does not follow from the cause" in str(raised.value)


def test_a_provider_fault_cannot_be_claimed_for_a_completed_episode(
    tmp_path: Path,
) -> None:
    """The loop stops at the terminal dispatch and never calls the provider again."""
    from tests.test_runtime_execution_integrity import _completed_run, _rewrite_first_row

    root = tmp_path / "run"
    manifest, variants = _completed_run(root)

    def relabel(row: dict[str, Any]) -> None:
        row.update(
            outcome=OUTCOME_PROVIDER_FAILURE,
            # Carried through with the outcome: eligibility is derived from it,
            # so leaving it behind would refuse the row on that derivation
            # before reaching the terminality check this test is about.
            sample_status=sample_status_for(OUTCOME_PROVIDER_FAILURE),
            error_class=PROVIDER_FAULT_SERVER_ERROR,
            error_detail=STABLE_DETAIL,
            evaluation=None,
            failure_event={
                "phase": "adapter_call",
                "kind": PROVIDER_FAULT_SERVER_ERROR,
                "attempted_call": None,
                "response_type": None,
                "exception_type": None,
                "detail": STABLE_DETAIL,
            },
        )

    _rewrite_first_row(root / "episodes.jsonl", relabel)

    with pytest.raises(LedgerError) as raised:
        read_ledger(root / "episodes.jsonl", manifest, variants)  # type: ignore[arg-type]

    assert "never reach this failure afterwards" in str(raised.value)


def test_a_fault_the_contract_does_not_define_cannot_be_reported() -> None:
    """The set is closed, so an integration cannot invent an outcome."""
    with pytest.raises(ValueError) as raised:
        AdapterProviderError("provider_had_a_bad_day", "detail")

    assert "not a provider fault" in str(raised.value)


def test_an_attempt_outcome_the_contract_does_not_define_cannot_be_recorded() -> None:
    """A provider attempt is one of three things, and none of them is free text.

    The third is the request this integration made and could not classify —
    neither a response it could read nor one of the named faults. It is recorded
    rather than dropped because the request was still sent, still may have been
    billed, and the reservation it forfeited has to be accounted for somewhere.
    """
    from boundarybench.adapter import ATTEMPT_OUTCOMES, ProviderAttempt

    with pytest.raises(ValueError) as raised:
        ProviderAttempt(index=1, outcome="went-fine")

    assert "not a provider attempt outcome" in str(raised.value)
    assert list(ATTEMPT_OUTCOMES) == ["fault", "response", "unclassified_error"]


def test_an_attempt_cannot_name_a_fault_the_contract_does_not_define() -> None:
    """The same closed set as the raised error, checked where it is recorded.

    An attempt is built by the adapter and travels to a durable row without
    passing through :class:`AdapterProviderError`, so the two need their own
    guard: a fault name that only exists on the telemetry path would reach a
    ledger unchallenged.
    """
    from boundarybench.adapter import ProviderAttempt

    with pytest.raises(ValueError) as raised:
        ProviderAttempt(index=1, outcome="fault", fault="provider_had_a_bad_day")

    assert "not a provider fault" in str(raised.value)


def test_an_unclassified_adapter_exception_is_only_recorded_at_a_named_boundary() -> None:
    """The redacted sentence names which of the two calls broke, or is not written.

    The boundary is part of the durable detail, so an unnamed one would produce a
    row that says an adapter failed somewhere. There are exactly two places an
    adapter is called from, and a third would be a contract change rather than a
    string.
    """
    from boundarybench.adapter import (
        ADAPTER_BOUNDARIES,
        ADAPTER_BOUNDARY_LAST_USAGE,
        redacted_adapter_detail,
    )

    detail = redacted_adapter_detail(
        RuntimeError("the provider said sk-ant-secret"),
        boundary=ADAPTER_BOUNDARY_LAST_USAGE,
    )
    assert "RuntimeError" in detail
    assert ADAPTER_BOUNDARY_LAST_USAGE in detail
    assert "sk-ant-secret" not in detail

    with pytest.raises(ValueError) as raised:
        redacted_adapter_detail(RuntimeError("boom"), boundary="somewhere_else")

    assert "not an adapter boundary" in str(raised.value)
    assert list(ADAPTER_BOUNDARIES) == ["last_usage", "next_call"]
