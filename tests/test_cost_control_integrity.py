"""Cost-control invariants exercised with adversarial stored and live inputs.

Every probe here runs the real runner over the real Anthropic SDK against the
in-process ``MockTransport``, or the real ledger/manifest readers over bytes a
real run wrote. Nothing is asserted through a helper that also produced the
number being checked, and no socket is opened.

The checks cover:

1. A cost guard whose cap differs from the manifest is refused before dispatch.
2. A response one input token above the authorised reservation was settled as
   measured cost, committing above the declared cap.
3. A ledger row whose measured cost or whose retry exposure had been zeroed was
   accepted on resume, so the run forgot what it had spent.
4. A stored pricing rate retyped from the string ``"2"`` to the JSON number
   ``999`` was accepted and silently projected back to the shipped ``"2"``.
"""

from __future__ import annotations

import dataclasses
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from boundarybench.budget import (
    BudgetError,
    CostControls,
    CostReservationBreachedError,
    RunCostGuard,
    usd_text,
)
from boundarybench.ledger import (
    OUTCOME_COST_RESERVATION_BREACHED,
    OUTCOME_PROVIDER_RATE_LIMITED,
    LedgerError,
    read_ledger,
)
from boundarybench.pricing import UnknownModelPriceError, price_for
from boundarybench.providers.anthropic_messages import (
    MAX_OUTPUT_TOKENS,
    AnthropicRetryPolicy,
)
from boundarybench.runmanifest import (
    RunManifestFormatError,
    load_run_manifest,
    open_run_session,
)
from boundarybench.runner import RunProvenanceError, compiled_variants, execute_run
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.suite import validate_suite
from tests.anthropic_transport import error_body, message_body, tool_use_block
from tests.conftest import SUITE_MANIFEST
from tests.test_runner_cost_control import (
    SONNET,
    STOP_CALL,
    capped_run,
    first_request_bound,
    first_request_reservation,
)

LEDGER = "episodes.jsonl"
MANIFEST = "run_manifest.json"


def _controls(cap: str) -> CostControls:
    return CostControls(
        max_cost_usd=Decimal(cap),
        max_episodes=12,
        price=price_for(provider="anthropic", model=SONNET),
    )


def _guard(cap: str) -> RunCostGuard:
    return RunCostGuard(
        controls=_controls(cap),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_attempts_per_turn=AnthropicRetryPolicy().max_attempts,
    )


def _rows(root: Path) -> list[dict[str, Any]]:
    text = (root / LEDGER).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def _write_rows(root: Path, rows: list[dict[str, Any]]) -> None:
    (root / LEDGER).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _reread(root: Path) -> Any:
    """Re-read a run directory through the production readers, nothing else."""
    manifest = load_run_manifest(root / MANIFEST)
    variants = compiled_variants(validate_suite(SUITE_MANIFEST))
    return read_ledger(root / LEDGER, manifest, variants)


def _resume(root: Path) -> None:
    """Resume the directory the way the runner does, with a matched guard."""
    suite = validate_suite(SUITE_MANIFEST)
    scaffold = load_scaffold(STANDARD_SCAFFOLD)
    manifest = load_run_manifest(root / MANIFEST)
    guard = RunCostGuard(
        controls=manifest.cost_controls,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_attempts_per_turn=AnthropicRetryPolicy().max_attempts,
    )
    from boundarybench.providers.anthropic_messages import AnthropicMessagesAdapter
    from tests.anthropic_transport import RecordingTransport

    transport = RecordingTransport([], default=message_body([STOP_CALL], model=SONNET))
    adapter = AnthropicMessagesAdapter(
        model=SONNET,
        client=transport.client(),
        cost_guard=guard,
        sleep=lambda seconds: None,
    )
    with open_run_session(root, manifest) as session:
        execute_run(
            suite=suite,
            scaffold=scaffold,
            adapter=adapter,
            session=session,
            cost_guard=guard,
        )


# -- 1. the guard a run carries is the cap its manifest declares -------------


def test_a_guard_whose_cap_is_not_the_manifests_cap_makes_no_request(
    tmp_path: Path,
) -> None:
    """A mismatched manifest and guard cap is refused before dispatch."""
    root = tmp_path / "run"
    with pytest.raises((RunProvenanceError, BudgetError)) as raised:
        capped_run(root, cap="0.000001", guard=_guard("100"))

    assert "cost_controls" in str(raised.value)
    assert not (root / LEDGER).exists() or _rows(root) == []


def test_a_guard_bound_to_the_manifests_cap_is_accepted(tmp_path: Path) -> None:
    """The same path, with the cap the manifest actually declares."""
    report, transport, _guard_used = capped_run(
        tmp_path / "run", cap="100", guard=_guard("100")
    )
    assert report.completed_count == 12
    assert transport.calls == 12


def test_a_capped_manifest_executed_without_a_guard_is_refused(
    tmp_path: Path,
) -> None:
    """A cap nothing enforces is not a cap, so the run does not start."""
    root = tmp_path / "run"
    with pytest.raises((RunProvenanceError, BudgetError)):
        capped_run(root, cap="100", guard=None, no_guard=True)
    assert not (root / LEDGER).exists() or _rows(root) == []


def test_an_uncapped_manifest_may_not_carry_a_guard(tmp_path: Path) -> None:
    """The other direction: a guard enforcing a limit no manifest records."""
    root = tmp_path / "run"
    with pytest.raises((RunProvenanceError, BudgetError)):
        capped_run(root, cap=None, episodes=None, guard=_guard("100"))
    assert not (root / LEDGER).exists() or _rows(root) == []


# -- 2. a settlement may not exceed the reservation that authorised it -------


def test_settling_one_token_above_the_reservation_fails_closed() -> None:
    """Cap and reservation exactly equal; usage one input token beyond it."""
    reservation = first_request_reservation()
    guard = _guard(str(reservation))
    authorised = guard.authorize(input_tokens_upper_bound=first_request_bound())
    assert authorised == reservation

    with pytest.raises(CostReservationBreachedError) as raised:
        guard.settle(
            authorised,
            input_tokens=first_request_bound() + 1,
            output_tokens=MAX_OUTPUT_TOKENS,
        )

    price = price_for(provider="anthropic", model=SONNET)
    measured = price.cost(
        input_tokens=first_request_bound() + 1, output_tokens=MAX_OUTPUT_TOKENS
    )
    # The honest number is kept, not discarded: the provider reported it and the
    # run really may owe it. What is refused is calling it authorised.
    assert guard.measured_usd == measured
    assert measured > reservation
    assert raised.value.measured_usd == measured
    assert raised.value.reservation_usd == reservation


def test_settling_exactly_the_reservation_is_not_a_breach() -> None:
    """Spending exactly what was authorised is not exceeding it."""
    reservation = first_request_reservation()
    guard = _guard(str(reservation))
    authorised = guard.authorize(input_tokens_upper_bound=first_request_bound())
    measured = guard.settle(
        authorised,
        input_tokens=first_request_bound(),
        output_tokens=MAX_OUTPUT_TOKENS,
    )
    assert measured == reservation
    assert guard.measured_usd == reservation


def test_a_breaching_response_stops_the_run_and_records_what_it_cost(
    tmp_path: Path,
) -> None:
    """The same overshoot through the real runner, ledger and SDK."""
    root = tmp_path / "run"
    over = message_body(
        [STOP_CALL],
        model=SONNET,
        input_tokens=first_request_bound() + 1,
        output_tokens=MAX_OUTPUT_TOKENS,
    )
    report, transport, _guard_used = capped_run(
        root, cap=str(first_request_reservation()), default=over
    )

    rows = _rows(root)
    assert len(rows) == 1
    assert rows[0]["outcome"] == OUTCOME_COST_RESERVATION_BREACHED
    # One request was made and answered; the run stops rather than making a
    # second one it could not have authorised.
    assert transport.calls == 1
    price = price_for(provider="anthropic", model=SONNET)
    measured = price.cost(
        input_tokens=first_request_bound() + 1, output_tokens=MAX_OUTPUT_TOKENS
    )
    assert rows[0]["usage"]["cost_usd"] == float(measured)
    assert rows[0]["cost_exposure_usd"] == 0
    # Rendered the way this build renders every amount, not with ``str``: the
    # scale of a decimal product moves with the token counts, and the report's
    # own rendering is the one a reader compares against.
    assert report.as_dict()["cost_accounting"]["measured_usd"] == usd_text(measured)
    # And the ledger reads back: honest evidence of a breach is still evidence.
    assert len(_reread(root)) == 1


# -- 3. a resume rederives spend rather than trusting the stored totals ------


def _twelve_measured_rows(root: Path) -> list[dict[str, Any]]:
    capped_run(root, cap="100")
    rows = _rows(root)
    assert len(rows) == 12
    assert all(row["usage"]["cost_usd"] == pytest.approx(0.000564) for row in rows)
    return rows


def test_a_zeroed_row_cost_is_refused_rather_than_recovered(tmp_path: Path) -> None:
    """A valid row with ``usage.cost_usd`` changed to zero is refused."""
    root = tmp_path / "run"
    rows = _twelve_measured_rows(root)
    rows[4]["usage"]["cost_usd"] = 0
    _write_rows(root, rows)

    with pytest.raises(LedgerError) as raised:
        _reread(root)
    assert "cost_usd" in str(raised.value)


def test_a_zeroed_row_cost_stops_the_resume(tmp_path: Path) -> None:
    root = tmp_path / "run"
    rows = _twelve_measured_rows(root)
    rows[4]["usage"]["cost_usd"] = 0
    _write_rows(root, rows)

    with pytest.raises(LedgerError):
        _resume(root)


def test_an_inflated_row_cost_is_refused_too(tmp_path: Path) -> None:
    """Derivation is an equality, not a floor: overstating is refused as well."""
    root = tmp_path / "run"
    rows = _twelve_measured_rows(root)
    rows[4]["usage"]["cost_usd"] = 9.0
    _write_rows(root, rows)

    with pytest.raises(LedgerError):
        _reread(root)


def test_zeroed_retry_exposure_is_refused_rather_than_recovered(
    tmp_path: Path,
) -> None:
    """A failed-attempt row with exposure changed to zero is refused."""
    root = tmp_path / "run"
    rate_limited = (429, error_body("rate_limit_error", "slow down"))
    capped_run(root, cap="100", steps=[rate_limited, rate_limited, rate_limited])
    rows = _rows(root)
    assert rows[0]["outcome"] == OUTCOME_PROVIDER_RATE_LIMITED
    exposure = float(first_request_reservation() * 3)
    assert rows[0]["cost_exposure_usd"] == exposure

    rows[0]["cost_exposure_usd"] = 0
    _write_rows(root, rows)
    with pytest.raises(LedgerError) as raised:
        _reread(root)
    assert "cost_exposure_usd" in str(raised.value)


def test_zeroed_attempt_reservations_are_refused(tmp_path: Path) -> None:
    """The exposure and the attempt evidence it derives from move together.

    Zeroing the row total alone is caught by the derivation; zeroing the
    per-attempt reservations it is derived from has to be caught by the same
    reader, or the derivation would just confirm a consistent forgery of both.
    """
    root = tmp_path / "run"
    rate_limited = (429, error_body("rate_limit_error", "slow down"))
    capped_run(root, cap="100", steps=[rate_limited, rate_limited, rate_limited])
    rows = _rows(root)
    for turn in rows[0]["provider_telemetry"]["turns"]:
        for attempt in turn["attempts"]:
            attempt["cost_reservation_usd"] = "0"
    rows[0]["cost_exposure_usd"] = 0
    _write_rows(root, rows)

    with pytest.raises(LedgerError):
        _reread(root)


#: The floor a range check permits: the fixed per-request input allowance and
#: this run's pinned output ceiling, at its own pricing policy. Every real
#: request of this run reserves more, because every real request has a body.
PERMITTED_FLOOR = "0.012288"


def _rate_limited_row(root: Path) -> list[dict[str, Any]]:
    """One real three-attempt rate-limited row, written by the real adapter."""
    rate_limited = (429, error_body("rate_limit_error", "slow down"))
    capped_run(root, cap="100", steps=[rate_limited, rate_limited, rate_limited])
    rows = _rows(root)
    assert rows[0]["outcome"] == OUTCOME_PROVIDER_RATE_LIMITED
    # ``usd_text``, not ``str``: a row stores the one rendering this build
    # writes, and a Decimal's own text carries whatever trailing zero the
    # arithmetic that produced it left behind.
    reserved = usd_text(first_request_reservation())
    attempts = [
        attempt
        for turn in rows[0]["provider_telemetry"]["turns"]
        for attempt in turn["attempts"]
    ]
    assert [attempt["cost_reservation_usd"] for attempt in attempts] == [reserved] * 3
    assert rows[0]["cost_exposure_usd"] == float(first_request_reservation() * 3)
    return rows


def test_a_plausible_in_range_reservation_rewrite_is_refused(tmp_path: Path) -> None:
    """Lowered reservations and a matching edited total fail exact derivation.

    The reader rederives each reservation from the run's manifest, scaffold,
    variant, trajectory, model-specific request profile, and pinned pricing
    policy rather than accepting an internally consistent range value.
    """
    root = tmp_path / "run"
    rows = _rate_limited_row(root)
    for turn in rows[0]["provider_telemetry"]["turns"]:
        for attempt in turn["attempts"]:
            attempt["cost_reservation_usd"] = PERMITTED_FLOOR
    rows[0]["cost_exposure_usd"] = float(Decimal(PERMITTED_FLOOR) * 3)
    _write_rows(root, rows)

    with pytest.raises(LedgerError) as raised:
        _reread(root)
    assert "cost_reservation_usd" in str(raised.value)


def test_a_plausible_in_range_reservation_rewrite_stops_the_resume(
    tmp_path: Path,
) -> None:
    """The same forgery through the production resume path, which forgot spend."""
    root = tmp_path / "run"
    rows = _rate_limited_row(root)
    for turn in rows[0]["provider_telemetry"]["turns"]:
        for attempt in turn["attempts"]:
            attempt["cost_reservation_usd"] = PERMITTED_FLOOR
    rows[0]["cost_exposure_usd"] = float(Decimal(PERMITTED_FLOOR) * 3)
    _write_rows(root, rows)

    with pytest.raises(LedgerError):
        _resume(root)


def test_a_single_in_range_attempt_reservation_rewrite_is_refused(
    tmp_path: Path,
) -> None:
    """One attempt of three, moved *up* and still inside the permitted range.

    The adjacent direction: not every attempt lowered to the floor, but one
    attempt of one turn given an amount another request of this same run really
    did reserve, with the row's exposure adjusted to match. Every attempt of one
    turn is the same request, so a reservation that differs between them is
    refused whether or not the total it feeds still adds up.
    """
    root = tmp_path / "run"
    rows = _rate_limited_row(root)
    attempts = rows[0]["provider_telemetry"]["turns"][0]["attempts"]
    inflated = Decimal(attempts[1]["cost_reservation_usd"]) + Decimal("0.000002")
    attempts[1]["cost_reservation_usd"] = str(inflated)
    rows[0]["cost_exposure_usd"] = float(
        sum(Decimal(attempt["cost_reservation_usd"]) for attempt in attempts)
    )
    _write_rows(root, rows)

    with pytest.raises(LedgerError) as raised:
        _reread(root)
    assert "cost_reservation_usd" in str(raised.value)


def test_a_later_turns_reservation_may_not_be_an_earlier_turns(tmp_path: Path) -> None:
    """A multi-turn episode: each turn's request is a different, exact amount.

    The second turn of an episode carries the first turn's answer in its body,
    so it reserves more. Rewriting it to the first turn's reservation is the
    most plausible edit available — the amount is one this run genuinely
    reserved, on a real request, in this very row — and it is refused, because
    the reservation checked is the one that turn's own request projection
    produces rather than any amount the run ever used.
    """
    root = tmp_path / "run"
    read_records = message_body([tool_use_block("read_records", {})], model=SONNET)
    capped_run(root, cap="100", steps=[read_records])
    rows = _rows(root)
    turns = rows[0]["provider_telemetry"]["turns"]
    assert len(turns) == 2
    first, second = (turn["attempts"][0]["cost_reservation_usd"] for turn in turns)
    assert Decimal(second) > Decimal(first)

    turns[1]["attempts"][0]["cost_reservation_usd"] = first
    _write_rows(root, rows)

    with pytest.raises(LedgerError) as raised:
        _reread(root)
    assert "cost_reservation_usd" in str(raised.value)


def test_an_honest_multi_turn_capped_row_still_reads_back(tmp_path: Path) -> None:
    """The derivation is an equality the real runner's own rows satisfy."""
    root = tmp_path / "run"
    read_records = message_body([tool_use_block("read_records", {})], model=SONNET)
    capped_run(root, cap="100", steps=[read_records])
    assert len(_reread(root)) == 12


def test_a_capped_run_records_a_reservation_on_every_attempt(tmp_path: Path) -> None:
    """The durable evidence exposure is derived from, stated key-exact."""
    root = tmp_path / "run"
    capped_run(root, cap="100")
    attempts = [
        attempt
        for row in _rows(root)
        for turn in row["provider_telemetry"]["turns"]
        for attempt in turn["attempts"]
    ]
    assert attempts
    for attempt in attempts:
        # A decimal string, not a JSON number, and an amount a request of this
        # run could actually have reserved: the exact figure depends on the
        # serialised body, which differs per variant and per turn.
        reserved = attempt["cost_reservation_usd"]
        assert isinstance(reserved, str)
        assert Decimal(reserved) >= first_request_reservation() * Decimal("0.5")
        assert attempt["cost_settlement"] == "measured"


def test_an_uncapped_run_records_no_reservation(tmp_path: Path) -> None:
    """Null, not zero: this run authorised nothing because it capped nothing."""
    root = tmp_path / "run"
    capped_run(root, cap=None, episodes=None)
    attempts = [
        attempt
        for row in _rows(root)
        for turn in row["provider_telemetry"]["turns"]
        for attempt in turn["attempts"]
    ]
    assert attempts
    for attempt in attempts:
        assert attempt["cost_reservation_usd"] is None
        assert attempt["cost_settlement"] is None


def test_a_scaffold_other_than_the_pinned_one_cannot_check_a_capped_ledger(
    tmp_path: Path,
) -> None:
    """The rederivation is bound to the manifest, not to whoever calls the reader.

    A caller who could hand in any scaffold could hand in one whose requests are
    small enough to make a lowered reservation come out right, which would move
    the forgery one file across rather than refuse it.
    """
    from tests.scaffolds import scaffold_payload, write_scaffold

    root = tmp_path / "run"
    capped_run(root, cap="100")
    other = load_scaffold(write_scaffold(tmp_path / "other.json", scaffold_payload()))
    manifest = load_run_manifest(root / MANIFEST)
    variants = compiled_variants(validate_suite(SUITE_MANIFEST))

    with pytest.raises(LedgerError) as raised:
        read_ledger(root / LEDGER, manifest, variants, scaffold=other)
    assert "scaffold" in str(raised.value)


def test_a_capped_run_this_build_cannot_reproduce_is_not_range_checked(
    tmp_path: Path,
) -> None:
    """A pinned output ceiling this adapter does not send is refused, not bounded.

    The fallback the fix removes: if the reservation cannot be rebuilt, there is
    no honest check left but a range, and a range is what accepted the forgery.
    So the ledger is refused instead.
    """
    root = tmp_path / "run"
    capped_run(root, cap="100")
    manifest = load_run_manifest(root / MANIFEST)
    settings = dict(manifest.adapter_settings)
    settings["max_output_tokens"] = MAX_OUTPUT_TOKENS // 2
    elsewhere = dataclasses.replace(manifest, adapter_settings=settings)
    variants = compiled_variants(validate_suite(SUITE_MANIFEST))

    with pytest.raises(LedgerError) as raised:
        read_ledger(root / LEDGER, elsewhere, variants)
    assert "max_output_tokens" in str(raised.value)


def test_a_row_from_the_previous_ledger_schema_is_refused_by_version(
    tmp_path: Path,
) -> None:
    """A capped row written under version 5 states a claim only a range checked."""
    root = tmp_path / "run"
    rows = _twelve_measured_rows(root)
    for row in rows:
        row["schema_version"] = 5
    _write_rows(root, rows)

    with pytest.raises(LedgerError) as raised:
        _reread(root)
    assert "schema_version" in str(raised.value)


def test_an_old_schema_row_is_refused_explicitly(tmp_path: Path) -> None:
    """A row without the derivation fields cannot be rederived, so it is refused."""
    root = tmp_path / "run"
    rows = _twelve_measured_rows(root)
    for turn in rows[0]["provider_telemetry"]["turns"]:
        for attempt in turn["attempts"]:
            attempt.pop("cost_reservation_usd")
            attempt.pop("cost_settlement")
    rows[0]["schema_version"] = 4
    _write_rows(root, rows)

    with pytest.raises(LedgerError) as raised:
        _reread(root)
    assert "schema_version" in str(raised.value)


# -- 4. the stored pricing block is the shipped policy, key and type exact ---


def _tamper_pricing(root: Path, **changes: Any) -> None:
    payload = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    pricing = payload["cost_controls"]["pricing"]
    for key, value in changes.items():
        if value is _REMOVE:
            pricing.pop(key)
        else:
            pricing[key] = value
    (root / MANIFEST).write_text(json.dumps(payload, indent=2), encoding="utf-8")


_REMOVE = object()


def test_a_retyped_input_rate_is_refused(tmp_path: Path) -> None:
    """A stored decimal-string rate rewritten as a JSON number is refused."""
    root = tmp_path / "run"
    capped_run(root, cap="100")
    _tamper_pricing(root, input_usd_per_million_tokens=999)

    with pytest.raises(RunManifestFormatError) as raised:
        load_run_manifest(root / MANIFEST)
    assert "input_usd_per_million_tokens" in str(raised.value)


def test_an_altered_output_rate_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "run"
    capped_run(root, cap="100")
    _tamper_pricing(root, output_usd_per_million_tokens="10.0")

    with pytest.raises(RunManifestFormatError):
        load_run_manifest(root / MANIFEST)


@pytest.mark.parametrize(
    "changes",
    [
        {"source": "https://example.invalid/prices"},
        {"recorded_on": "2026-08-08"},
        {"effective_through": "2026-12-31"},
        {"effective_through": 20260831},
        {"note": "cheaper than it says"},
        {"unit_tokens": 1000},
        {"unit_tokens": "1000000"},
        {"provider": "not-anthropic"},
        {"extra_field": "unreviewed"},
        {"model": _REMOVE},
        {"note": _REMOVE},
    ],
)
def test_every_field_of_the_pricing_block_is_checked(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    root = tmp_path / "run"
    capped_run(root, cap="100")
    _tamper_pricing(root, **changes)

    with pytest.raises(RunManifestFormatError):
        load_run_manifest(root / MANIFEST)


def test_an_untampered_manifest_still_loads(tmp_path: Path) -> None:
    """The check is exact, not merely strict: the shipped block passes it."""
    root = tmp_path / "run"
    capped_run(root, cap="100")
    manifest = load_run_manifest(root / MANIFEST)
    assert manifest.cost_controls.max_cost_usd == Decimal("100")
    assert manifest.cost_controls.price is not None
    assert manifest.cost_controls.price.input_usd_per_million_tokens == Decimal("2")


def test_the_superseded_anthropic_pricing_policy_is_refused_exactly() -> None:
    stored = _controls("100").as_dict()
    pricing = stored["pricing"]
    assert isinstance(pricing, dict)
    pricing["policy_version"] = "anthropic-standard-2026-08-09"

    with pytest.raises(BudgetError) as refused:
        CostControls.from_stored(stored, provider="anthropic", model=SONNET)

    assert str(refused.value) == (
        "cost_controls: pricing block field 'policy_version' is "
        "'anthropic-standard-2026-08-09', and the pricing policy this build ships "
        "states 'anthropic-standard-2026-09-07'. A run's costs are only comparable "
        "within one reviewed price table, so a stored block that disagrees with the "
        "shipped one — in value or in JSON type — is refused rather than re-priced "
        "under a table it never used"
    )


# -- the zero-call guarantees these fixes must not weaken --------------------


def test_an_unpriced_model_cannot_be_capped() -> None:
    """No client is built, because the cap could never be enforced."""
    with pytest.raises(UnknownModelPriceError):
        price_for(provider="anthropic", model="claude-sonnet-5-latest")


def test_a_cap_without_a_price_is_refused() -> None:
    with pytest.raises(BudgetError):
        CostControls(max_cost_usd=Decimal("1"), max_episodes=1, price=None)


def test_an_unaffordable_first_request_is_never_dispatched(tmp_path: Path) -> None:
    """The pre-existing guarantee, restated here so a fix cannot quietly drop it."""
    report, transport, _guard_used = capped_run(tmp_path / "run", cap="0.000001")
    assert transport.calls == 0
    assert report.as_dict()["cost_accounting"]["measured_usd"] == "0"
