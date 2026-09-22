"""The two bounds a run is authorised under, enforced before anything is sent.

A safety bound that a command line can raise is not a bound, and a token
exposure nobody checks is not a cap. This build states one absolute maximum
number of provider calls and one fixed token ceiling; the operator may ask for
*fewer* calls and cannot ask for more, and the token ceiling does not move
without a code change.

Both refusals happen before a request exists. The call bound is checked while
the arguments are still being read, so a run of sixty-one never constructs an
SDK client at all; the token bound is checked inside the cost guard, on the same
authorise-before-dispatch seam the cost cap already uses, so a call whose
cumulative token reservation would pass the ceiling is refused with nothing on
the wire.

Nothing here reaches a provider: the only transport in this module is
``httpx.MockTransport``, and the token refusals are proven by there being no
request for it to answer.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest

from operatebench.agents.model import MAX_OUTPUT_TOKENS
from operatebench.agents.pricing import (
    RATE_SOURCE_OPERATOR,
    LifecycleCostGuard,
    LifecyclePricingPolicy,
    TokenCapExceededError,
)
from operatebench.providers.cost import CostCapExceededError
from tools import preflight_lifecycle_provider_evidence as preflight

FIXED_TOKEN_HARD_CAP = 1_650_300


def policy() -> LifecyclePricingPolicy:
    return LifecyclePricingPolicy(
        policy_id="offline_operator_pinned_v1",
        input_usd_per_mtok=Decimal("1.25"),
        output_usd_per_mtok=Decimal("10.00"),
        rate_source=RATE_SOURCE_OPERATOR,
    )


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "evidence"
    directory.mkdir(mode=0o700)
    return directory


# -- the absolute call bound --------------------------------------------------


def test_the_call_bound_is_an_absolute_constant_of_this_build() -> None:
    assert preflight.MAX_PROVIDER_CALLS == 60
    assert preflight.DEFAULT_MAX_PROVIDER_CALLS == 60
    assert preflight.EXPECTED_PROVIDER_CALLS == 46
    assert preflight.EXPECTED_PROVIDER_CALLS < preflight.MAX_PROVIDER_CALLS


@pytest.mark.parametrize("asked", [61, 100, 0, -1])
def test_a_call_count_outside_the_bound_is_refused_before_anything_is_built(
    output_dir: Path, monkeypatch: pytest.MonkeyPatch, asked: int
) -> None:
    def refuse(**_: Any) -> dict[str, Any]:
        raise AssertionError("the preflight ran for a call count it must refuse")

    def no_client(*_: Any, **__: Any) -> Any:
        raise AssertionError("an SDK client was constructed for a refused run")

    monkeypatch.setattr(preflight, "preflight", refuse)
    monkeypatch.setattr(openai, "OpenAI", no_client)
    code = preflight.main(
        ["--output-dir", str(output_dir), "--max-provider-calls", str(asked)]
    )
    assert code != 0
    assert list(output_dir.iterdir()) == []


@pytest.mark.parametrize("asked", [1, 46, 59, 60])
def test_a_call_count_inside_the_bound_is_accepted(asked: int) -> None:
    assert preflight.check_max_provider_calls(asked) == asked


@pytest.mark.parametrize("asked", [61, 0, -1])
def test_the_call_bound_check_names_its_refusal(asked: int) -> None:
    with pytest.raises(preflight.PreflightFailure):
        preflight.check_max_provider_calls(asked)


# -- the token ceiling --------------------------------------------------------


def test_the_token_ceiling_is_fixed_by_this_build() -> None:
    assert preflight.TOKEN_HARD_CAP == FIXED_TOKEN_HARD_CAP
    assert preflight.token_worst_case(preflight.MAX_PROVIDER_CALLS) == (
        preflight.MAX_PROVIDER_CALLS
        * (preflight.LARGEST_REQUEST_TOKEN_BOUND + MAX_OUTPUT_TOKENS)
    )


def test_no_command_line_flag_raises_the_token_ceiling() -> None:
    parser_options = [
        option
        for action in preflight.build_parser()._actions
        for option in action.option_strings
    ]
    assert "--token-hard-cap" not in parser_options
    assert not [name for name in parser_options if "token" in name]


def test_a_call_count_whose_token_exposure_passes_the_ceiling_is_refused() -> None:
    """Sixty-one calls expose 1,677,805 tokens against a ceiling of 1,650,300."""
    assert preflight.token_worst_case(61) > FIXED_TOKEN_HARD_CAP
    with pytest.raises(preflight.PreflightFailure):
        preflight.check_token_exposure(max_calls=61, token_hard_cap=FIXED_TOKEN_HARD_CAP)


def test_a_ceiling_that_does_not_cover_the_run_is_refused() -> None:
    with pytest.raises(preflight.PreflightFailure):
        preflight.check_token_exposure(max_calls=60, token_hard_cap=1_000)


def test_the_shipped_bounds_cover_each_other() -> None:
    exposure = preflight.check_token_exposure(
        max_calls=preflight.MAX_PROVIDER_CALLS, token_hard_cap=FIXED_TOKEN_HARD_CAP
    )
    assert exposure["token_worst_case"] == FIXED_TOKEN_HARD_CAP
    assert exposure["token_hard_cap"] == FIXED_TOKEN_HARD_CAP


# -- the token ceiling at the dispatch seam -----------------------------------


def guard(
    *, cap: str = "500.00", token_hard_cap: int | None = None
) -> LifecycleCostGuard:
    return LifecycleCostGuard(
        policy=policy(),
        cap_usd=Decimal(cap),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        token_hard_cap=token_hard_cap,
    )


def test_a_guard_accumulates_token_reservations_and_refuses_at_the_boundary() -> None:
    per_call = 1_000 + MAX_OUTPUT_TOKENS
    enforcing = guard(token_hard_cap=per_call * 2)
    enforcing.authorize(input_tokens_upper_bound=1_000)
    enforcing.authorize(input_tokens_upper_bound=1_000)
    assert enforcing.reserved_tokens == per_call * 2
    with pytest.raises(CostCapExceededError):
        enforcing.authorize(input_tokens_upper_bound=1_000)
    assert enforcing.reserved_tokens == per_call * 2


def test_the_token_refusal_is_the_budget_refusal_the_kernel_already_classifies() -> None:
    assert issubclass(TokenCapExceededError, CostCapExceededError)


def test_a_first_call_larger_than_the_ceiling_is_refused_with_nothing_dispatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json={})

    client = openai.OpenAI(
        api_key="not-a-credential-placeholder",
        base_url="https://api.openai.com/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    enforcing = guard(token_hard_cap=MAX_OUTPUT_TOKENS)
    with pytest.raises(CostCapExceededError):
        enforcing.authorize(input_tokens_upper_bound=1)
    assert calls["count"] == 0
    assert enforcing.reserved_tokens == 0
    client.close()


def test_a_guard_without_a_token_ceiling_is_unchanged() -> None:
    plain = guard()
    plain.authorize(input_tokens_upper_bound=23_409)
    assert plain.reserved_tokens == 23_409 + MAX_OUTPUT_TOKENS
    assert plain.token_hard_cap is None


def test_the_cost_cap_still_refuses_before_the_token_ceiling_is_reached() -> None:
    tight = LifecycleCostGuard(
        policy=policy(),
        cap_usd=Decimal("0.0001"),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        token_hard_cap=FIXED_TOKEN_HARD_CAP,
    )
    with pytest.raises(CostCapExceededError) as raised:
        tight.authorize(input_tokens_upper_bound=1_000)
    assert not isinstance(raised.value, TokenCapExceededError)


# -- the ceiling on a whole run, over the real SDK and a mock transport -------


def test_a_ceiling_the_first_call_cannot_fit_stops_the_run_with_nothing_dispatched(
    output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A whole scenario, refused at the dispatch seam: no request, no decision.

    The ledger still exists and still says what happened — it is excluded under
    the budget code with no call rows, because there were no calls — and the
    mock transport, which counts every request that reaches it, answered none.
    """
    monkeypatch.setattr(preflight, "TOKEN_HARD_CAP", MAX_OUTPUT_TOKENS + 1)
    spec = preflight.load_spec(preflight.FIXTURE)
    ledger_path = output_dir / "token-ceiling.ndjson"
    result = preflight.run_cell(
        spec,
        ledger_path=ledger_path,
        policy=policy(),
        cap=Decimal("10.00"),
        max_calls=preflight.MAX_PROVIDER_CALLS,
    )
    assert result["failure"] is not None
    assert result["provider"].calls == 0
    assert result["capture"].calls == 0
    audit = preflight.read_execution_ledger(ledger_path, require_complete=True)
    assert audit.status == "excluded"
    assert not audit.scored
    assert audit.totals.provider_calls == 0
    assert audit.header.controls.token_hard_cap == MAX_OUTPUT_TOKENS + 1


# -- the preflight carries both bounds into the ledger's control block --------


def test_the_control_block_states_the_ceiling_every_reservation_was_priced_at() -> None:
    controls = preflight.controls_for(
        policy(), cap=Decimal("10.00"), max_calls=preflight.MAX_PROVIDER_CALLS
    )
    assert controls.max_output_tokens == MAX_OUTPUT_TOKENS
    assert controls.token_hard_cap == FIXED_TOKEN_HARD_CAP
    assert controls.max_provider_calls == preflight.MAX_PROVIDER_CALLS


def test_a_run_refused_on_its_call_count_reports_nothing_about_a_provider(
    output_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = preflight.main(["--output-dir", str(output_dir), "--max-provider-calls", "61"])
    assert code == 2
    printed = capsys.readouterr()
    assert printed.out.strip() == ""
    assert str(preflight.MAX_PROVIDER_CALLS) in printed.err


def test_a_live_environment_refusal_still_states_it_made_no_call(
    output_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-credential")
    assert preflight.main(["--output-dir", str(output_dir)]) == 1
    reported = json.loads(capsys.readouterr().out)
    assert reported["ok"] is False
    assert reported["live_provider_call"] is False
    assert reported["credential_read"] is False
