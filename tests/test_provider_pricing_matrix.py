"""The pricing policy, once it has to answer for more than one provider.

A model identifier is only unique *within* a provider, so the moment a second
provider exists the question "what does this model cost" stops having an answer
until the provider is named. These tests fix the three things that has to mean:

* every provider carries its own reviewed policy — its own version, its own
  published source and its own observation date — because those are per-provider
  facts and folding them into one table would attribute one vendor's page to
  another vendor's rates;
* the current Anthropic policy has an independently pinned digest, so edits to
  its corrected source/date/window identity cannot pass as an unchanged table;
* a model string that exists under one provider is not a price under another,
  and an unknown provider or model is refused rather than approximated.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from boundarybench.pricing import (
    PROVIDER_PRICING_POLICIES,
    ModelPrice,
    UnknownModelPriceError,
    digest_of_policy_payload,
    price_for,
    pricing_policy_digest,
    pricing_policy_payload,
)

#: The digest of the current corrected Anthropic standard-price policy. Pinned
#: independently so a source, date, effective window, note, or rate edit must
#: deliberately move every current golden assertion. Historical run/evidence
#: bytes retain the superseded identity they actually exercised.
ANTHROPIC_POLICY_DIGEST = (
    "f0b78a71619841f0c3e7d21a094258d24093b28d02744b5d3fb8ce6b7138561a"
)

#: The five exact models this lane prices, and the rates checked against each
#: provider's own published page on 2026-08-11.
EXPECTED_PRICES = (
    ("anthropic", "claude-haiku-4-5-20251001", "1", "5"),
    ("anthropic", "claude-sonnet-5", "2", "10"),
    ("openai", "gpt-5.6-luna", "0.20", "1.20"),
    ("xai", "grok-4.5", "2", "6"),
    ("mistral", "mistral-small-2603", "0.15", "0.60"),
)


def test_current_anthropic_policy_digest_matches_its_independent_golden() -> None:
    """The corrected current table is bound to a literal, not to itself."""
    assert pricing_policy_digest("anthropic") == ANTHROPIC_POLICY_DIGEST


def test_anthropic_policy_payload_still_defaults() -> None:
    """The no-argument call keeps naming the policy it always named.

    Callers written against the single-provider build asked for "the" policy and
    got Anthropic's. They still do, rather than getting a merged table whose
    digest is a number no existing run recorded.
    """
    assert pricing_policy_payload()["policy_digest"] == ANTHROPIC_POLICY_DIGEST
    assert pricing_policy_payload()["provider"] == "anthropic"


@pytest.mark.parametrize(
    ("provider", "model", "input_rate", "output_rate"), EXPECTED_PRICES
)
def test_exact_rates_are_decimal_and_exact(
    provider: str, model: str, input_rate: str, output_rate: str
) -> None:
    """Each exact model prices at its published rate, in Decimal."""
    price = price_for(provider=provider, model=model)
    assert isinstance(price, ModelPrice)
    assert price.provider == provider
    assert price.model == model
    assert price.input_usd_per_million_tokens == Decimal(input_rate)
    assert price.output_usd_per_million_tokens == Decimal(output_rate)
    # Decimal all the way through: a binary float would make the rate itself a
    # number nobody published.
    assert isinstance(price.input_usd_per_million_tokens, Decimal)
    assert isinstance(price.output_usd_per_million_tokens, Decimal)


def test_cost_is_exact_decimal_arithmetic() -> None:
    """A cost is the published rate times the counts, exactly.

    ``0.20`` has no exact binary representation, so a build that priced in
    floats would report a cost that is nobody's arithmetic. One million output
    tokens at USD 1.20/M is exactly USD 1.20 and must compare equal to it.

    The counts here stay inside this model's base tier where the point is
    exactness, because a million *input* tokens is past its published
    272,000-token threshold and is priced at the long rates — which is a
    different property, checked in ``tests/test_tiered_pricing.py``.
    """
    price = price_for(provider="openai", model="gpt-5.6-luna")
    assert price.cost(input_tokens=0, output_tokens=1_000_000) == Decimal("1.20")
    assert price.cost(input_tokens=3, output_tokens=7) == Decimal("0.0000090")
    assert price.cost(input_tokens=1_000_000, output_tokens=0) == Decimal("0.40")
    # An untiered table prices one rate at any prompt length, and just as exactly.
    sonnet = price_for(provider="anthropic", model="claude-sonnet-5")
    assert sonnet.cost(input_tokens=1_000_000, output_tokens=0) == Decimal("2")


def test_every_provider_states_its_own_source_and_date() -> None:
    """A per-provider fact is recorded per provider, not shared.

    The published page and the date this repository read it are different for
    each vendor. One shared ``source`` would cite Anthropic's pricing page as
    the provenance of Mistral's rates, which is a false statement about where a
    number came from.
    """
    sources = {
        provider: policy.source for provider, policy in PROVIDER_PRICING_POLICIES.items()
    }
    assert sources["openai"] == "https://developers.openai.com/api/docs/pricing"
    # The vendor's pricing page, not its model page: the xAI rates are tiered
    # and the tier — threshold, inclusivity and both long rates — is stated
    # there, so that is the page the policy has to cite.
    assert sources["xai"] == "https://docs.x.ai/developers/pricing"
    assert sources["mistral"] == "https://docs.mistral.ai/inference/pricing"
    # Four distinct providers, four distinct sources and four distinct policy
    # versions: nothing is shared that is not actually shared.
    assert len(set(sources.values())) == len(sources) == 4
    versions = {policy.policy_version for policy in PROVIDER_PRICING_POLICIES.values()}
    assert len(versions) == 4


def test_xai_records_that_no_dated_snapshot_was_published() -> None:
    """The one price whose provenance is weaker says so, in the artefact.

    xAI's model page carries a "last updated" date rather than a dated price
    snapshot, so this repository cannot record the price as of a stated day the
    way it can for the others. That limitation belongs in the policy a run
    hashes, not only in a commit message.
    """
    policy = PROVIDER_PRICING_POLICIES["xai"]
    assert policy.snapshot_limitation is not None
    assert "grok-4.5" in policy.models
    # The other three published a snapshot this build could pin, so they state
    # no such limitation and the field is not a place to put prose.
    for provider in ("anthropic", "openai", "mistral"):
        assert PROVIDER_PRICING_POLICIES[provider].snapshot_limitation is None


def test_same_model_string_under_the_wrong_provider_is_refused() -> None:
    """A model is priced by (provider, model), never by model alone.

    This is the failure the pair exists to prevent: ``grok-4.5`` is a real,
    priced model, and asking OpenAI for it must not quietly return xAI's rate.
    """
    for provider, model in (
        ("openai", "grok-4.5"),
        ("xai", "gpt-5.6-luna"),
        ("mistral", "claude-sonnet-5"),
        ("anthropic", "mistral-small-2603"),
    ):
        with pytest.raises(UnknownModelPriceError):
            price_for(provider=provider, model=model)


def test_unknown_provider_and_unknown_model_fail_closed() -> None:
    """Neither a nearest neighbour nor a zero: a refusal."""
    with pytest.raises(UnknownModelPriceError):
        price_for(provider="cohere", model="command-r")
    with pytest.raises(UnknownModelPriceError):
        price_for(provider="openai", model="gpt-5.6-luna-latest")
    # Case-sensitive and matched whole, like every other identifier here.
    with pytest.raises(UnknownModelPriceError):
        price_for(provider="OpenAI", model="gpt-5.6-luna")
    with pytest.raises(UnknownModelPriceError):
        price_for(provider="openai", model="GPT-5.6-LUNA")


def test_digest_covers_the_rates_and_moves_when_one_is_tampered_with() -> None:
    """A digest that did not move under an edited rate would authorise nothing."""
    for provider in PROVIDER_PRICING_POLICIES:
        payload = pricing_policy_payload(provider)
        assert payload["policy_digest"] == digest_of_policy_payload(payload)
        tampered = {
            **payload,
            "models": {
                model: {**block, "input_usd_per_million_tokens": "0"}
                for model, block in payload["models"].items()
            },
        }
        assert digest_of_policy_payload(tampered) != payload["policy_digest"]


def test_each_policy_digest_is_distinct() -> None:
    """Four tables, four identities. A shared digest would make two runs priced
    under different vendors' rates indistinguishable in their manifests."""
    digests = {
        provider: pricing_policy_digest(provider)
        for provider in PROVIDER_PRICING_POLICIES
    }
    assert len(set(digests.values())) == 4
