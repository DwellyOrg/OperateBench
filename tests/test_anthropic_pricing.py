"""The versioned Anthropic pricing policy.

A price is a *measurement input*: a run that multiplies tokens by a number
nobody wrote down is not reporting a cost, it is asserting one. So the policy is
an artefact with a version, a digest, a source and an effective date, it covers
exactly the two models this build has a verified published price for, and it
refuses every model it does not name rather than substituting a neighbour's
price.

Nothing here contacts a provider.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from boundarybench.pricing import (
    ANTHROPIC_PRICING_MODELS,
    PRICING_POLICY_RECORDED_ON,
    PRICING_POLICY_SOURCE,
    PRICING_POLICY_VERSION,
    UnknownModelPriceError,
    measured_cost_usd,
    price_for,
    pricing_policy_digest,
    pricing_policy_payload,
)

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-5"


def test_the_policy_covers_exactly_the_two_priced_models() -> None:
    """Exactly two, and no third: an extra model is one with no checked price."""
    assert set(ANTHROPIC_PRICING_MODELS) == {SONNET, HAIKU}


def test_the_prices_are_the_ones_the_published_policy_states() -> None:
    sonnet = price_for(provider="anthropic", model=SONNET)
    haiku = price_for(provider="anthropic", model=HAIKU)

    assert sonnet.input_usd_per_million_tokens == Decimal("2")
    assert sonnet.output_usd_per_million_tokens == Decimal("10")
    assert haiku.input_usd_per_million_tokens == Decimal("1")
    assert haiku.output_usd_per_million_tokens == Decimal("5")


def test_the_standard_sonnet_price_has_no_announced_end() -> None:
    """Official docs withdrew the announced increase before it took effect."""
    sonnet = price_for(provider="anthropic", model=SONNET)
    assert sonnet.effective_through is None
    assert "standard-tier price" in sonnet.note
    assert "increase will not occur" in sonnet.note
    # Haiku's standard price likewise has no announced end.
    assert price_for(provider="anthropic", model=HAIKU).effective_through is None


def test_the_policy_states_its_version_source_and_recording_date() -> None:
    payload = pricing_policy_payload()

    assert payload["policy_version"] == PRICING_POLICY_VERSION
    assert payload["source"] == PRICING_POLICY_SOURCE
    assert payload["recorded_on"] == PRICING_POLICY_RECORDED_ON
    assert payload["policy_digest"] == pricing_policy_digest()
    assert len(payload["policy_digest"]) == 64
    assert payload["policy_version"] == "anthropic-standard-2026-09-07"
    assert payload["recorded_on"] == "2026-09-07"


def test_the_digest_covers_the_table_it_claims_to_name() -> None:
    """A version string an editor can leave alone is not provenance.

    The digest is taken over the prices themselves, so changing one without
    changing the recorded digest is impossible: the recomputed digest moves.
    """
    from boundarybench import pricing

    baseline = pricing_policy_digest()
    payload = pricing_policy_payload()
    edited = {
        **payload,
        "models": {
            **payload["models"],
            SONNET: {**payload["models"][SONNET], "input_usd_per_million_tokens": "1"},
        },
    }

    assert pricing.digest_of_policy_payload(edited) != baseline


def test_an_unknown_model_is_refused_rather_than_priced_by_a_neighbour() -> None:
    with pytest.raises(UnknownModelPriceError) as unknown:
        price_for(provider="anthropic", model="claude-opus-5")
    assert "claude-opus-5" in str(unknown.value)


@pytest.mark.parametrize(
    "alias",
    [
        "claude-sonnet-5-latest",
        "claude-sonnet-5-20260801",
        "claude-sonnet",
        "Claude-Sonnet-5",
    ],
)
def test_no_alias_or_prefix_ever_resolves_to_a_priced_model(alias: str) -> None:
    """Exact identifiers only. A prefix match is a silent substitution."""
    with pytest.raises(UnknownModelPriceError):
        price_for(provider="anthropic", model=alias)


def test_another_provider_is_refused_even_for_an_identically_named_model() -> None:
    with pytest.raises(UnknownModelPriceError):
        price_for(provider="openai", model=SONNET)


def test_cost_is_computed_in_decimal_and_is_exact() -> None:
    """Binary floating point cannot represent these rates; Decimal can."""
    sonnet = price_for(provider="anthropic", model=SONNET)

    # 137 input tokens at USD 2/1M and 29 output tokens at USD 10/1M.
    assert sonnet.cost(input_tokens=137, output_tokens=29) == Decimal("0.000564")
    # A quantity where repeated float addition visibly drifts.
    assert sonnet.cost(input_tokens=1, output_tokens=0) == Decimal("0.000002")
    total = sum(
        (sonnet.cost(input_tokens=1, output_tokens=0) for _ in range(10)),
        Decimal(0),
    )
    assert total == Decimal("0.00002")


def test_a_million_tokens_costs_exactly_the_stated_rate() -> None:
    haiku = price_for(provider="anthropic", model=HAIKU)
    assert haiku.cost(input_tokens=1_000_000, output_tokens=1_000_000) == Decimal("6")


def test_measured_cost_refuses_a_missing_count_rather_than_reading_it_as_zero() -> None:
    """Unmeasured is not free. A null count yields a null cost, never 0.0."""
    sonnet = price_for(provider="anthropic", model=SONNET)

    assert measured_cost_usd(sonnet, input_tokens=None, output_tokens=29) is None
    assert measured_cost_usd(sonnet, input_tokens=137, output_tokens=None) is None
    assert measured_cost_usd(sonnet, input_tokens=None, output_tokens=None) is None
    assert measured_cost_usd(sonnet, input_tokens=0, output_tokens=0) == Decimal(0)


def test_a_negative_or_non_integer_token_count_is_refused() -> None:
    sonnet = price_for(provider="anthropic", model=SONNET)
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError):
            sonnet.cost(input_tokens=bad, output_tokens=0)  # type: ignore[arg-type]


def test_the_policy_table_cannot_be_edited_through_the_module() -> None:
    """A module-level dict would let one assignment redefine every run's costs."""
    with pytest.raises(TypeError):
        ANTHROPIC_PRICING_MODELS["claude-opus-5"] = None  # type: ignore[index]
