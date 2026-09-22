"""A split two-run allocation: 36/36 episodes, USD 66/34, 6 turns.

One authorisation — 72 episodes and USD 100 — executed as two immutable runs.
The risk is that two independently valid USD 100 manifests turn one
authorisation into USD 200 of exposure, so the allocation has to be a property
of each run rather than a sentence in a document.

What the code enforces, and what these tests pin:

* each run carries its own cap, and no request is dispatched whose conservative
  bound would take *that run* past *that cap*;
* the cap is inside the run's configuration identity, so a run cannot be
  continued under the other run's allocation;
* spend is accounted per run directory, so one run's ledger never offsets or
  funds the other's.

What the code does not do, stated plainly: nothing here adds USD 66 and USD 34
and checks the total against USD 100. The combined ceiling is held by whatever
authorises the work and by the two caps an operator types. What is mechanised is
that neither run can exceed the allocation it was given.

No test here contacts Anthropic: the executing ones replace the SDK client with
the in-process transport.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from boundarybench.budget import CostCapExceededError, CostControls, RunCostGuard
from boundarybench.cli import main
from boundarybench.pricing import price_for
from tests.anthropic_transport import RecordingTransport, message_body, tool_use_block
from tests.conftest import SUITE_MANIFEST

SONNET = "claude-sonnet-5"
HAIKU = "claude-haiku-4-5-20251001"

#: The allocation under test. One row per run; the combined authorisation is
#: their sum, and that sum is what an operator approves.
SONNET_CAP = "66"
HAIKU_CAP = "34"
COMBINED_CAP = Decimal("100")
EPISODES_PER_MODEL = 36
COMBINED_EPISODES = 72
MAX_TURNS = 6

STOP_CALL = tool_use_block(
    "complete_case",
    {
        "disposition": "STOP",
        "primary_reason_code": "NO_APPLICABLE_RULE",
        "secondary_reason_codes": [],
        "evidence_refs": [],
    },
)


def _preflight_argv(output: Path, *, model: str, cap: str, **extra: str) -> list[str]:
    argv = [
        "preflight",
        str(SUITE_MANIFEST),
        "--adapter",
        "anthropic",
        "--model",
        model,
        "--output-dir",
        str(output),
        "--max-cost-usd",
        cap,
        "--max-episodes",
        str(EPISODES_PER_MODEL),
        "--max-turns",
        str(MAX_TURNS),
        "--trials",
        "3",
        "--json",
    ]
    for flag, value in extra.items():
        argv += [f"--{flag.replace('_', '-')}", value]
    return argv


def _private(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    return root


def _preflight(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> tuple[int, dict[str, Any], str]:
    code = main(argv)
    captured = capsys.readouterr()
    payload: dict[str, Any] = json.loads(captured.out) if captured.out.strip() else {}
    return code, payload, captured.err


def _install_transport(
    monkeypatch: pytest.MonkeyPatch, *, model: str
) -> RecordingTransport:
    from boundarybench.providers.anthropic_messages import AnthropicMessagesAdapter

    transport = RecordingTransport([], default=message_body([STOP_CALL], model=model))
    client = transport.client()

    def factory(*, model: str, **kwargs: Any) -> AnthropicMessagesAdapter:
        return AnthropicMessagesAdapter(
            model=model,
            client=client,
            cost_guard=kwargs.get("cost_guard"),
            sleep=lambda seconds: None,
        )

    monkeypatch.setattr("boundarybench.cli.build_anthropic_adapter", factory)
    return transport


# -- each run's plan fits its own allocation, offline ------------------------


@pytest.mark.parametrize(
    ("model", "cap"),
    [(SONNET, SONNET_CAP), (HAIKU, HAIKU_CAP)],
)
def test_each_approved_run_plans_36_episodes_within_its_own_cap(
    model: str,
    cap: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")

    code, payload, _err = _preflight(
        _preflight_argv(_private(tmp_path / model), model=model, cap=cap), capsys
    )

    assert code == 0
    assert payload["plan"]["planned_episodes"] == EPISODES_PER_MODEL
    assert payload["plan"]["limits"]["max_turns"] == MAX_TURNS
    assert payload["cost_controls"]["max_cost_usd"] == cap
    assert payload["cost_controls"]["max_episodes"] == EPISODES_PER_MODEL

    bound = payload["cost_bound"]
    # 36 episodes x 6 turns x the adapter's 3 bounded attempts.
    assert bound["max_requests"] == EPISODES_PER_MODEL * MAX_TURNS * 3
    assert bound["fits_within_cap"] is True
    assert Decimal(bound["upper_bound_usd"]) <= Decimal(cap)


def test_the_two_plans_sum_to_the_combined_authorisation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """72 episodes and USD 100, as two runs and not as one."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")

    episodes = 0
    caps = Decimal(0)
    exposure = Decimal(0)
    for model, cap in ((SONNET, SONNET_CAP), (HAIKU, HAIKU_CAP)):
        code, payload, _err = _preflight(
            _preflight_argv(_private(tmp_path / model), model=model, cap=cap), capsys
        )
        assert code == 0
        episodes += payload["plan"]["planned_episodes"]
        caps += Decimal(payload["cost_controls"]["max_cost_usd"])
        exposure += Decimal(payload["cost_bound"]["upper_bound_usd"])

    assert episodes == COMBINED_EPISODES
    assert caps == COMBINED_CAP
    # The worst case both runs together can reach is under the combined cap, and
    # it is the caps — not this number — that are enforced.
    assert exposure <= COMBINED_CAP


def test_the_allocations_are_not_interchangeable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sonnet's plan does not fit inside Haiku's share, and is refused offline.

    The split is not decorative: Sonnet's tokens cost twice Haiku's, so running
    the Sonnet plan against the Haiku allocation is a plan whose own worst case
    exceeds the cap. The preflight says so before a run directory exists.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    root = _private(tmp_path / "run")

    code, _payload, err = _preflight(
        _preflight_argv(root, model=SONNET, cap=HAIKU_CAP), capsys
    )

    assert code == 1
    assert "upper bound" in err.lower()
    assert not (root / "run_manifest.json").exists()


# -- no borrowing at the enforcement layer -----------------------------------


def test_a_guard_enforces_its_own_cap_not_the_combined_authorisation() -> None:
    """A request that would fit inside USD 100 is still refused at USD 34."""
    guard = RunCostGuard(
        controls=CostControls(
            max_cost_usd=Decimal(HAIKU_CAP),
            max_episodes=EPISODES_PER_MODEL,
            price=price_for(provider="anthropic", model=HAIKU),
        ),
        max_output_tokens=1024,
        max_attempts_per_turn=3,
    )
    # Priced so that one request's bound lands between the two caps.
    huge = 40_000_000

    reservation = guard.reservation_for(input_tokens_upper_bound=huge)
    assert Decimal(HAIKU_CAP) < reservation <= COMBINED_CAP

    with pytest.raises(CostCapExceededError) as refused:
        guard.authorize(input_tokens_upper_bound=huge)
    assert HAIKU_CAP in str(refused.value)
    # Nothing was reserved, so the refusal did not consume the run's budget.
    assert guard.remaining_usd == Decimal(HAIKU_CAP)


def test_a_run_cannot_be_continued_under_the_other_run_s_allocation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap is inside configuration identity, so re-typing it is a new run."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    root = tmp_path / "sonnet"
    argv = [
        "run-suite",
        str(SUITE_MANIFEST),
        "--adapter",
        "anthropic",
        "--model",
        SONNET,
        "--output-dir",
        str(root),
        "--max-turns",
        str(MAX_TURNS),
    ]

    _install_transport(monkeypatch, model=SONNET)
    assert main([*argv, "--max-cost-usd", SONNET_CAP]) == 0
    capsys.readouterr()

    _install_transport(monkeypatch, model=SONNET)
    code = main([*argv, "--max-cost-usd", str(COMBINED_CAP)])

    err = capsys.readouterr().err
    assert code == 1
    assert "already holds configuration" in err
    assert "Traceback" not in err


def test_each_run_accounts_only_its_own_spend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two runs share an authorisation, not a ledger.

    A guard is seeded from its own run directory, so the Haiku run neither
    inherits the Sonnet run's spend nor is funded by what the Sonnet run left
    unspent. Each report's remaining budget is that run's own cap minus that
    run's own authorised exposure, and nothing else.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    accounting: dict[str, dict[str, str]] = {}

    for model, cap in ((SONNET, SONNET_CAP), (HAIKU, HAIKU_CAP)):
        _install_transport(monkeypatch, model=model)
        code = main(
            [
                "run-suite",
                str(SUITE_MANIFEST),
                "--adapter",
                "anthropic",
                "--model",
                model,
                "--output-dir",
                str(tmp_path / model),
                "--max-turns",
                str(MAX_TURNS),
                "--max-cost-usd",
                cap,
                "--max-episodes",
                str(EPISODES_PER_MODEL),
                "--json",
            ]
        )
        assert code == 0
        accounting[model] = json.loads(capsys.readouterr().out)["cost_accounting"]

    for model, cap in ((SONNET, SONNET_CAP), (HAIKU, HAIKU_CAP)):
        report = accounting[model]
        measured = Decimal(report["measured_usd"])
        exposure = Decimal(report["exposure_usd"])
        assert report["max_cost_usd"] == cap
        assert measured > 0
        assert Decimal(report["remaining_usd"]) == Decimal(cap) - measured - exposure

    # Same episodes, same token counts, half the price: the Haiku run's measured
    # cost is its own, not a share of a pooled total.
    assert Decimal(accounting[SONNET]["measured_usd"]) == 2 * Decimal(
        accounting[HAIKU]["measured_usd"]
    )
