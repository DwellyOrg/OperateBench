"""A capped run, written and read back, on every provider this build ships.

The ledger reader rebuilds what each recorded attempt reserved from the request
that turn actually sent — the scaffold, the variant, the replayed trajectory and
the adapter's own request mapping — and refuses a row whose stored reservation
is not exactly that. Rebuilding the request means knowing which integration
built it, and that had been wired to one provider: any non-empty capped ledger
from OpenAI, xAI or Mistral could be written but never read back, so the final
read at the end of a run and every resume of it failed on a run that was
otherwise correct.

So each lane here executes a whole capped run through the real runner and the
real SDK over an in-process transport, reads the directory back through the
production readers, resumes it, and then proves the reader still refuses an
edited row and a manifest whose request settings do not describe a request this
build would send. Nothing here opens a socket, and nothing here is evidence
about a model.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from boundarybench.budget import CostControls, RunCostGuard
from boundarybench.ledger import LedgerError, read_ledger
from boundarybench.pricing import price_for
from boundarybench.providers.common import ProviderRetryPolicy
from boundarybench.runmanifest import (
    RunLimits,
    build_run_manifest,
    load_run_manifest,
    open_run_session,
)
from boundarybench.runner import RunReport, compiled_variants, execute_run
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.suite import validate_suite
from tests import mistral_transport, openai_transport
from tests.conftest import SUITE_MANIFEST

LEDGER = "episodes.jsonl"
MANIFEST = "run_manifest.json"
MAX_TURNS = 4

#: The one terminal action a scripted answer takes, so every episode ends on the
#: turn it starts and the plan of 12 completes.
STOP_ARGUMENTS = json.dumps(
    {
        "disposition": "STOP",
        "primary_reason_code": "NO_APPLICABLE_RULE",
        "secondary_reason_codes": [],
        "evidence_refs": [],
    },
    sort_keys=True,
)


@dataclass(frozen=True)
class Lane:
    """One provider's whole path from a request to a row that reads back."""

    name: str
    provider: str
    implementation: str
    adapter_version: str
    model: str
    max_output_tokens: int
    #: Builds an adapter over a fresh in-process transport, under this guard.
    build: Callable[[RunCostGuard | None], tuple[Any, Any]]


def _openai_lane() -> Lane:
    from boundarybench.providers.openai_responses import (
        MAX_OUTPUT_TOKENS,
        OPENAI_ADAPTER_VERSION,
        OPENAI_IMPLEMENTATION,
        OPENAI_PROVIDER,
        OpenAIResponsesAdapter,
    )

    model = "gpt-5.6-luna"

    def build(guard: RunCostGuard | None) -> tuple[Any, Any]:
        transport = openai_transport.RecordingTransport(
            [],
            default=openai_transport.responses_body(
                [openai_transport.function_call_item("complete_case", STOP_ARGUMENTS)],
                model=model,
            ),
        )
        adapter = OpenAIResponsesAdapter(
            model=model,
            client=transport.client(),
            cost_guard=guard,
            sleep=lambda seconds: None,
        )
        return adapter, transport

    return Lane(
        name="openai",
        provider=OPENAI_PROVIDER,
        implementation=OPENAI_IMPLEMENTATION,
        adapter_version=OPENAI_ADAPTER_VERSION,
        model=model,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        build=build,
    )


def _xai_lane() -> Lane:
    from boundarybench.providers.xai_openai_compat import (
        MAX_OUTPUT_TOKENS,
        XAI_ADAPTER_VERSION,
        XAI_IMPLEMENTATION,
        XAI_PROVIDER,
        XAIOpenAICompatAdapter,
    )

    model = "grok-4.5"

    def build(guard: RunCostGuard | None) -> tuple[Any, Any]:
        transport = openai_transport.RecordingTransport(
            [],
            default=openai_transport.chat_body(
                [openai_transport.tool_call("complete_case", STOP_ARGUMENTS)],
                model=model,
            ),
        )
        adapter = XAIOpenAICompatAdapter(
            model=model,
            client=transport.xai_client(),
            cost_guard=guard,
            sleep=lambda seconds: None,
        )
        return adapter, transport

    return Lane(
        name="xai",
        provider=XAI_PROVIDER,
        implementation=XAI_IMPLEMENTATION,
        adapter_version=XAI_ADAPTER_VERSION,
        model=model,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        build=build,
    )


def _mistral_lane() -> Lane:
    from boundarybench.providers.mistral_chat import (
        MAX_OUTPUT_TOKENS,
        MISTRAL_ADAPTER_VERSION,
        MISTRAL_IMPLEMENTATION,
        MISTRAL_PROVIDER,
        MistralChatAdapter,
    )

    model = "mistral-small-2603"

    def build(guard: RunCostGuard | None) -> tuple[Any, Any]:
        transport = mistral_transport.RecordingTransport(
            [],
            default=mistral_transport.chat_body(
                [mistral_transport.tool_call("complete_case", STOP_ARGUMENTS)],
                model=model,
            ),
        )
        adapter = MistralChatAdapter(
            model=model,
            client=transport.client(),
            cost_guard=guard,
            sleep=lambda seconds: None,
        )
        return adapter, transport

    return Lane(
        name="mistral",
        provider=MISTRAL_PROVIDER,
        implementation=MISTRAL_IMPLEMENTATION,
        adapter_version=MISTRAL_ADAPTER_VERSION,
        model=model,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        build=build,
    )


LANES = (_openai_lane(), _xai_lane(), _mistral_lane())
LANE_IDS = tuple(lane.name for lane in LANES)


def _controls(lane: Lane, cap: str = "100") -> CostControls:
    return CostControls(
        max_cost_usd=Decimal(cap),
        max_episodes=12,
        price=price_for(provider=lane.provider, model=lane.model),
    )


def _guard(lane: Lane, cap: str = "100") -> RunCostGuard:
    return RunCostGuard(
        controls=_controls(lane, cap),
        max_output_tokens=lane.max_output_tokens,
        max_attempts_per_turn=ProviderRetryPolicy().max_attempts,
    )


def _capped_run(
    lane: Lane, root: Path, *, guard: RunCostGuard | None = None, cap: str = "100"
) -> tuple[RunReport, Any, RunCostGuard]:
    """One whole capped run of this lane, executed by the real runner."""
    suite = validate_suite(SUITE_MANIFEST)
    scaffold = load_scaffold(STANDARD_SCAFFOLD)
    active = guard if guard is not None else _guard(lane, cap)
    adapter, transport = lane.build(active)
    manifest = build_run_manifest(
        suite=suite,
        scaffold=scaffold,
        provider=lane.provider,
        model=lane.model,
        implementation=lane.implementation,
        adapter_version=lane.adapter_version,
        adapter_settings=adapter.settings,
        trials=1,
        limits=RunLimits(
            max_turns=MAX_TURNS, max_messages=16, episode_timeout_seconds=30.0
        ),
        cost_controls=_controls(lane, cap),
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


def _reread(root: Path) -> tuple[Any, ...]:
    """Re-read a run directory through the production readers, nothing else."""
    manifest = load_run_manifest(root / MANIFEST)
    variants = compiled_variants(validate_suite(SUITE_MANIFEST))
    return read_ledger(root / LEDGER, manifest, variants)


def _rows(root: Path) -> list[dict[str, Any]]:
    text = (root / LEDGER).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def _write_rows(root: Path, rows: list[dict[str, Any]]) -> None:
    (root / LEDGER).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


# -- run, write, read back ----------------------------------------------------


@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
def test_a_capped_run_writes_twelve_priced_episodes(lane: Lane, tmp_path: Path) -> None:
    report, transport, guard = _capped_run(lane, tmp_path / "run")

    assert report.completed_count == 12
    assert transport.calls == 12
    assert guard.measured_usd > 0
    assert guard.exposure_usd == 0


@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
def test_the_final_read_of_a_capped_run_rebuilds_every_reservation(
    lane: Lane, tmp_path: Path
) -> None:
    """The read that a run's own last act performs, on a directory it wrote.

    A reader that cannot rebuild this provider's requests refuses the whole
    ledger, so a run that executed perfectly could not be read back — which is
    the failure this covers, one lane at a time.
    """
    root = tmp_path / "run"
    _capped_run(lane, root)

    records = _reread(root)

    assert len(records) == 12


@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
def test_a_resume_adopts_the_spend_its_ledger_records(lane: Lane, tmp_path: Path) -> None:
    root = tmp_path / "run"
    _first, _transport, first_guard = _capped_run(lane, root)

    fresh = _guard(lane)
    report, transport, _guard_used = _capped_run(lane, root, guard=fresh)

    assert transport.calls == 0
    assert report.resumed_count == 12
    assert fresh.measured_usd == first_guard.measured_usd
    assert fresh.measured_usd > 0


@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
def test_every_row_replays_against_the_variant_it_names(
    lane: Lane, tmp_path: Path
) -> None:
    """The reader's replay is not skipped for these providers either."""
    root = tmp_path / "run"
    _capped_run(lane, root)
    rows = _rows(root)

    assert len(rows) == 12
    assert all(row["usage"]["cost_usd"] > 0 for row in rows)
    assert all(row["cost_exposure_usd"] == 0 for row in rows)
    assert _reread(root)


# -- tampering ----------------------------------------------------------------


@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
def test_an_edited_reservation_is_refused_by_the_rebuilt_request(
    lane: Lane, tmp_path: Path
) -> None:
    """The point of rebuilding: a stored amount that the request does not imply.

    The row is otherwise untouched and internally consistent, so nothing but a
    re-derivation of this provider's own request could catch it.
    """
    root = tmp_path / "run"
    _capped_run(lane, root)
    rows = _rows(root)
    for turn in rows[0]["provider_telemetry"]["turns"]:
        for attempt in turn["attempts"]:
            attempt["cost_reservation_usd"] = "0.00000001"
    _write_rows(root, rows)

    with pytest.raises(LedgerError):
        _reread(root)


@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
def test_a_manifest_whose_request_mapping_drifted_is_refused(
    lane: Lane, tmp_path: Path
) -> None:
    """A run whose stored settings do not describe a request this build sends.

    Its reservations were taken against bodies this build cannot reproduce, so
    the ledger is refused rather than range-checked.
    """
    root = tmp_path / "run"
    _capped_run(lane, root)
    manifest = load_run_manifest(root / MANIFEST)
    drifted = dataclasses.replace(
        manifest,
        adapter_settings={
            **manifest.adapter_settings,
            "request_mapping": "not_the_mapping_this_build_sends_v0",
        },
    )
    variants = compiled_variants(validate_suite(SUITE_MANIFEST))

    with pytest.raises(LedgerError):
        read_ledger(root / LEDGER, drifted, variants)


@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
def test_a_capped_ledger_from_an_unknown_provider_fails_closed(
    lane: Lane, tmp_path: Path
) -> None:
    """No dispatch for a provider this build does not implement."""
    root = tmp_path / "run"
    _capped_run(lane, root)
    manifest = load_run_manifest(root / MANIFEST)
    elsewhere = dataclasses.replace(
        manifest, adapter=dataclasses.replace(manifest.adapter, provider="not-a-provider")
    )
    variants = compiled_variants(validate_suite(SUITE_MANIFEST))

    with pytest.raises(LedgerError):
        read_ledger(root / LEDGER, elsewhere, variants)


@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
def test_a_capped_ledger_from_another_implementation_is_refused(
    lane: Lane, tmp_path: Path
) -> None:
    """The right provider, a different integration: still not reproducible."""
    root = tmp_path / "run"
    _capped_run(lane, root)
    manifest = load_run_manifest(root / MANIFEST)
    elsewhere = dataclasses.replace(
        manifest,
        adapter=dataclasses.replace(manifest.adapter, implementation="some_other_api"),
    )
    variants = compiled_variants(validate_suite(SUITE_MANIFEST))

    with pytest.raises(LedgerError):
        read_ledger(root / LEDGER, elsewhere, variants)
