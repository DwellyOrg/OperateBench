"""Offline successor accounting; synthetic rates are never price admission."""

import json
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from fractions import Fraction
from unittest.mock import patch

import httpx
import openai
import pytest

from operatebench.agents.pricing import LifecyclePricingPolicy
from operatebench.providers.cost import CostCapExceededError
from tools.diagnose_aggregate_budget import DispatchTransport
from tools.three_flow_budget import CampaignBudget


def policy():
    return LifecyclePricingPolicy(
        policy_id="offline-synthetic-only",
        input_usd_per_mtok=Decimal("0"),
        output_usd_per_mtok=Decimal("1000000"),
        rate_source="operator_supplied_pinned_rates",
    )


def make_budget(root, **kwargs):
    root.chmod(0o700)
    return CampaignBudget(root, campaign_id="campaign-a", cap=Decimal("1000"), **kwargs)


def test_restart_retains_pending_as_unknown_and_allows_new_reserved_attempt(tmp_path):
    budget = make_budget(tmp_path, create=True)
    ident = budget.reserve("trial-a", input_bound=100, output_bound=200, policy=policy())
    before = budget.exposure
    budget.close()
    recovered = make_budget(tmp_path)
    try:
        assert recovered.records[ident]["state"] == "unknown"
        assert recovered.totals()["unknown"] == before
        assert recovered.totals()["known"] == 0
        assert recovered.totals()["pending"] == 0
        recovered.reserve("trial-b", input_bound=100, output_bound=200, policy=policy())
        assert recovered.exposure == 2 * before
    finally:
        recovered.close()


def test_global_parallel_atomic_cap_unknown_and_single_settlement(tmp_path):
    budget = make_budget(tmp_path, create=True)
    try:

        def reserve(index):
            try:
                return budget.reserve(
                    str(index), input_bound=0, output_bound=400, policy=policy()
                )
            except CostCapExceededError:
                return None

        with ThreadPoolExecutor(max_workers=4) as pool:
            admitted = [r for r in pool.map(reserve, range(8)) if r is not None]
        assert len(admitted) == 2
        assert budget.exposure == 800
        first, second = admitted
        budget.dispatch(budget.records[first]["cell"], {"max_output_tokens": 400})
        budget.settle(first, input_tokens=0, output_tokens=50)
        with pytest.raises(ValueError, match="already resolved"):
            budget.settle(first, input_tokens=0, output_tokens=0)
        budget.forfeit(second)
        assert budget.totals()["known"] == 50
        assert budget.totals()["unknown"] == 400
        assert budget.totals()["remaining"] == 550
    finally:
        budget.close()
    recovered = make_budget(tmp_path)
    try:
        assert recovered.exposure == 450
        assert recovered.totals()["unknown"] == 400
        with pytest.raises(ValueError):
            recovered.cancel(second)
        with pytest.raises(CostCapExceededError):
            recovered.reserve(
                "corrected-rerun", input_bound=0, output_bound=551, policy=policy()
            )
    finally:
        recovered.close()


def sdk_call(budget, cell, seen):
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "resp_offline",
                "object": "response",
                "created_at": 1,
                "model": "gpt-6-astra",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            },
        )

    http = httpx.Client(
        transport=DispatchTransport(httpx.MockTransport(handler), budget, cell),
        trust_env=False,
    )
    with openai.OpenAI(
        api_key="offline-not-a-credential", http_client=http, max_retries=0
    ) as client:
        return client.responses.create(
            model="gpt-6-astra", input="offline", max_output_tokens=400
        )


def test_real_sdk_dispatch_settle_and_unreserved_retry_no_wire(tmp_path):
    budget = make_budget(tmp_path, create=True)
    seen = []
    try:
        ident = budget.reserve(
            "trial-a", input_bound=10000, output_bound=400, policy=policy()
        )
        response = sdk_call(budget, "trial-a", seen)
        assert len(seen) == 1
        assert response.usage.output_tokens == 2
        budget.settle(
            ident,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        assert budget.exposure == Fraction(2)
        with pytest.raises(openai.APIConnectionError):
            sdk_call(budget, "trial-a", seen)
        assert len(seen) == 1
    finally:
        budget.close()


def test_real_sdk_failed_dispatch_journal_never_calls_wire(tmp_path):
    budget = make_budget(tmp_path, create=True)
    seen = []
    try:
        budget.reserve("trial-a", input_bound=10000, output_bound=400, policy=policy())
        with (
            patch(
                "tools.aggregate_budget.os.fsync",
                side_effect=OSError("synthetic durability fault"),
            ),
            pytest.raises(openai.APIConnectionError),
        ):
            sdk_call(budget, "trial-a", seen)
        assert seen == []
        assert budget.exposure == 400
        assert budget.broken
        with pytest.raises(ValueError, match="broken"):
            budget.reserve("trial-b", input_bound=0, output_bound=1, policy=policy())
    finally:
        budget.close()
    # A dispatch row may have reached disk despite fsync failure: never refund.
    recovered = make_budget(tmp_path)
    try:
        assert recovered.totals()["unknown"] == 400
    finally:
        recovered.close()


@pytest.mark.parametrize("cap", ["999", "1001", "NaN", "Infinity", "-1"])
def test_cap_is_global_explicit_currency_and_not_widenable(tmp_path, cap):
    tmp_path.chmod(0o700)
    with pytest.raises(ValueError):
        CampaignBudget(tmp_path, campaign_id="campaign-a", cap=Decimal(cap), create=True)
    assert list(tmp_path.iterdir()) == []


def test_foreign_campaign_torn_journal_and_second_owner_refused(tmp_path):
    budget = make_budget(tmp_path, create=True)
    with pytest.raises(BlockingIOError):
        make_budget(tmp_path)
    budget.close()
    with pytest.raises(ValueError, match="foreign"):
        CampaignBudget(tmp_path, campaign_id="campaign-b", cap=Decimal("1000"))
    path = tmp_path / "campaign-budget.jsonl"
    with path.open("ab") as stream:
        stream.write(b'{"torn":')
    with pytest.raises(ValueError):
        make_budget(tmp_path)
