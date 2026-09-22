"""The versioned provider pricing policies a cost-capped run is measured against.

A price is a measurement input, not a convenience. A run that multiplies its
token counts by a number nobody wrote down is asserting a cost rather than
reporting one, and a number that moved between two runs makes their costs
incomparable without either of them saying so. So the prices this build can use
live here as artefacts: a version, a digest over the prices themselves, the
source they were read from and the date they were recorded, all of which go into
the run's configuration identity and its manifest.

Four rules make a table safe to price a run with:

* **Exact identifiers only.** ``price_for`` matches the model string whole. No
  prefix, no alias, no case folding and no "nearest" model: a substitution is
  precisely the failure that produces a plausible cost for the wrong model.
* **Only models whose price was checked.** A table covers exactly the models
  this build has a published, verified price for. Anything else is refused,
  which is what makes an unpriced model impossible to price and therefore
  impossible to run under a cost cap.
* **Decimal, never binary float.** ``2/1_000_000`` has no exact binary
  representation, so a run that summed floats would report a cost that is
  nobody's arithmetic. Every price and every product here is a
  :class:`~decimal.Decimal`, and the conversion to a JSON number happens once,
  at the boundary where a row is written.
* **One policy per provider.** A model identifier is only unique within a
  provider, so a price is keyed on the *pair*. Each provider also has its own
  published page and its own date on which this repository read it, and those
  are per-provider facts: one shared ``source`` would cite one vendor's page as
  the provenance of another vendor's rates. So each provider carries a whole
  policy — version, source, date, models — and hashes to its own digest.

That last rule is what changed when this build stopped talking to one provider.
Policy identities normally remain byte-stable, but they must move when a source
claim changes. The current Anthropic identity therefore records the corrected
standard-price window independently of historical run and evidence bytes; those
records continue to name the policy they actually exercised.

Nothing here contacts a provider or reads a credential.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Context, Decimal, DecimalException, localcontext
from types import MappingProxyType
from typing import Any

from operatebench.providers.usage import MAX_EXACT_TOKEN_COUNT

#: One million tokens: the unit every published rate is quoted in.
TOKENS_PER_PRICED_UNIT = 1_000_000

# -- the numeric domain -------------------------------------------------------
#
# Three quantities reach the arithmetic here from outside — a token count, a
# tier threshold and a USD amount — and each of them is bounded on both ends
# rather than merely checked for sign and finiteness. Unbounded is not a
# neutral choice: an amount large enough overflows the decimal context and
# raises out of a cost guard as a bare ``decimal.Overflow`` mid-authorisation, a
# token count large enough silently *rounds* in the default 28-digit context —
# which is the one thing decimal arithmetic is used here to prevent — and an
# integer large enough cannot even be rendered into an error message, because
# Python refuses to convert an int of more than 4,300 digits to text.

#: The largest token count this build prices, and the largest a tier threshold
#: may state.
#:
#: Two to the fifty-third, minus one. It is the largest integer that is exact in
#: an IEEE-754 double, and therefore the largest one that survives a JSON number
#: in every reader this build's artefacts have to cross — which matters because
#: token counts and thresholds are written into ledger rows and hashed policy
#: payloads that other languages read. It is also some nine orders of magnitude
#: above the largest published context window, so nothing real is refused by it.
#:
#: Imported from :mod:`operatebench.providers.usage` and re-exported here under
#: the name it has always had. The same bound decides what a token count read off
#: a provider's wire may be, and stating it twice is how the price table and the
#: reader that feeds it come to disagree. It is the same object; every use below
#: is unchanged.

#: The largest USD amount this build accepts as a rate, a cost or a cap.
#:
#: A quadrillion dollars. Chosen to be far above any rate a provider publishes
#: per million tokens and any cap an operator could mean, and far below the
#: magnitudes at which decimal arithmetic stops being cheap and starts being a
#: denial-of-service surface.
MAX_USD_AMOUNT = Decimal(10) ** 15

#: The finest scale a USD amount may state: twelve decimal places.
#:
#: Published rates are quoted to at most a few places per million tokens, so a
#: twelfth place is already six orders of magnitude finer than anything a vendor
#: prints. Bounding the scale as well as the magnitude bounds the *digits*,
#: which is what makes the working precision below sufficient by construction
#: rather than by hope.
MAX_USD_DECIMAL_PLACES = 12

#: The precision every accepted amount's arithmetic is performed at.
#:
#: An accepted amount has at most 16 integer digits and 12 fractional ones, so
#: at most 28 significant digits; an accepted token count has at most 16. Their
#: product therefore has at most 44, a sum of two products at most 45, and a
#: run's accumulated total a handful more. Eighty leaves every one of those
#: exact with room to spare, which is the property that matters: inside the
#: accepted domain, no multiplication or addition here ever rounds.
EXACT_DECIMAL_PRECISION = 80


class PricingError(ValueError):
    """Base class for every pricing-policy failure."""


class UnknownModelPriceError(PricingError):
    """This build holds no reviewed price for the provider/model a run asked for.

    Refused rather than defaulted. Every fallback available here is a lie of a
    different shape: a nearest-neighbour price is a cost for a model that did
    not run, a zero is a claim that the run was free, and a null under a cost cap
    is a cap that cannot be enforced. So a cost-capped run against an unpriced
    model does not start.

    It is also the refusal for a model string that is real *under a different
    provider*. ``grok-4.5`` is a priced model, and asking for it under
    ``openai`` must not quietly return xAI's rate: that would produce a
    confident cost for a request that was never made.
    """


@contextmanager
def exact_decimal_context() -> Iterator[Context]:
    """Arithmetic over accepted amounts, at a precision that cannot round them.

    Explicit rather than inherited. The ambient decimal context is process-wide
    state any caller can change, and its default precision of 28 digits is below
    what an accepted count times an accepted rate needs — so a cost that was
    exact in a test could round in a run, in the direction nobody would notice.
    See :data:`EXACT_DECIMAL_PRECISION` for why eighty is sufficient.
    """
    with localcontext(Context(prec=EXACT_DECIMAL_PRECISION)) as context:
        yield context


def _rendered(value: Any) -> str:
    """A value's ``repr``, or a fixed stand-in when rendering it is itself unsafe.

    Python refuses to convert an integer of more than 4,300 digits to text, so
    interpolating a caller-supplied integer into an error message is a way to
    raise ``ValueError`` from inside the code that was refusing the value —
    replacing a named refusal with a bare one from the interpreter, at the point
    a policy payload was being built. The value is rendered when that is safe
    and described when it is not.
    """
    try:
        return repr(value)
    except ValueError:
        return f"a value of type {type(value).__name__} too large to render"


def check_token_count(value: Any, name: str) -> int:
    """One token count or tier threshold, inside the exactly-representable domain.

    ``bool`` is refused where an ``int`` is required although Python makes one:
    ``True`` tokens is not a measurement anybody took. The upper bound is
    :data:`MAX_EXACT_TOKEN_COUNT`, so every accepted count is exact in this
    build's arithmetic *and* in the JSON a later reader parses it from.
    """
    if type(value) is not int or value < 0 or value > MAX_EXACT_TOKEN_COUNT:
        raise PricingError(
            f"{name} must be a whole number of tokens from 0 to "
            f"{MAX_EXACT_TOKEN_COUNT}, and not a bool. That bound is the largest "
            "integer a JSON number carries exactly, so a count beyond it could not "
            "be written into a ledger row and read back as itself; it is also far "
            f"above any published context window. Got {_rendered(value)}"
        )
    return value


def check_usd_amount(value: Any, name: str) -> Decimal:
    """One USD amount, inside the domain this build can price exactly.

    A ``float`` is refused although it would arithmetic fine: ``0.40`` has no
    exact binary representation, so a table that accepted one would price runs
    at a number nobody published. A string is refused for the milder version of
    the same reason — it is an amount that has not been converted yet, and the
    conversion belongs at the point the table or the cap is written.

    The rest of the domain is bounded on both ends and checked here rather than
    where the arithmetic later fails. A negative amount is the failure that
    matters most: it produced a negative reservation and a negative cost, and a
    negative cost *raised* the guard's remaining budget — a cost control that
    paid the run for spending. An amount past :data:`MAX_USD_AMOUNT` or finer
    than :data:`MAX_USD_DECIMAL_PLACES` is refused where it is stated, so it
    cannot surface later as a raw ``decimal`` exception from inside a guard that
    was authorising a request.
    """
    if not isinstance(value, Decimal):
        raise PricingError(
            f"{name} must be a Decimal amount in USD, and not a float, a string, "
            "an int or a bool: a float is a binary approximation of the number "
            "that was published, and a string is one that has not been converted "
            f"yet. Got {_rendered(value)}"
        )
    if not value.is_finite():
        raise PricingError(
            f"{name} must be a finite Decimal amount in USD; a NaN compares false "
            "against every bound this build checks an amount against, and an "
            f"infinity is not an amount. Got {_rendered(value)}"
        )
    if value < 0:
        raise PricingError(
            f"{name} must not be negative. A negative amount inverts every "
            "control it reaches: a negative rate makes a reservation a credit and "
            "raises the remaining budget of the run that spends it. Got "
            f"{_rendered(value)}"
        )
    if value > MAX_USD_AMOUNT:
        raise PricingError(
            f"{name} must be at most USD {MAX_USD_AMOUNT}, which is far above any "
            "published rate or any cap an operator means and far below the "
            "magnitudes at which this arithmetic stops being cheap. An unbounded "
            "amount does not fail where it is stated; it fails as a decimal "
            f"overflow inside the guard authorising a request. Got {_rendered(value)}"
        )
    exponent = value.as_tuple().exponent
    if isinstance(exponent, int) and exponent < -MAX_USD_DECIMAL_PLACES:
        raise PricingError(
            f"{name} must state at most {MAX_USD_DECIMAL_PLACES} decimal places, "
            "which is already six orders of magnitude finer than any rate a vendor "
            "publishes. Bounding the scale is what keeps every accepted "
            f"multiplication and addition exact. Got {_rendered(value)}"
        )
    return value


def _checked_tokens(value: Any, name: str) -> int:
    return check_token_count(value, name)


def _checked_rate(value: Any, name: str) -> Decimal:
    """One published rate, as an exact Decimal in the accepted USD domain."""
    return check_usd_amount(value, f"{name} in USD per million tokens")


def _checked_text(value: Any, name: str, *, allow_none: bool = False) -> None:
    """One identity field of a price, as the string it is recorded as.

    Checked for exact type rather than truthiness: these fields are hashed into
    a policy payload and printed in a manifest, and a ``True`` where a note
    belongs would be recorded as a claim nobody wrote.
    """
    if allow_none and value is None:
        return
    if type(value) is not str or not value:
        raise PricingError(
            f"{name} must be a non-empty string"
            f"{' or None' if allow_none else ''}, got {_rendered(value)}"
        )


#: Which of a turn's two counts selects a tiered model's rates.
#:
#: The prompt, on both providers that publish a tier, and stated in the payload
#: rather than assumed by a reader: a later reader recomputing a recorded cost
#: has to know which number crossed the line, and "the input tokens" is a fact
#: about the vendor's rule rather than a convention this build could pick.
TIER_SELECTOR = "input_tokens"


@dataclass(frozen=True)
class LongContextTier:
    """The higher rates a request pays once its prompt crosses a stated line.

    Three numbers and one boolean, and the boolean is the part that cannot be
    inferred. xAI states that all tokens in a request use the long rates *once
    the threshold is reached*, so a prompt of exactly 200,000 is already a
    long-context one. OpenAI states ``≤272K input tokens`` against ``>272K
    input tokens``, so a prompt of exactly 272,000 is still the base tier. A
    build that picked one reading and applied it to both would misprice every
    request that landed exactly on the other vendor's boundary — and would do it
    in the direction that under-states spend, which is the direction a cost cap
    cannot survive.

    The rates apply to the *whole request*: both providers price the completion
    at the long rate too once the prompt crosses. Applying the long rate to the
    input alone would understate every long-context turn by the difference on
    its output, silently.
    """

    input_token_threshold: int
    #: Whether a prompt of exactly :attr:`input_token_threshold` tokens is
    #: already priced at these rates. ``True`` for a ``>=`` rule, ``False`` for
    #: a ``>`` one.
    applies_at_threshold: bool
    input_usd_per_million_tokens: Decimal
    output_usd_per_million_tokens: Decimal

    def __post_init__(self) -> None:
        # ``bool`` is an ``int`` subclass, so the threshold's type test is exact:
        # ``True`` is not a token count, and a float threshold would put a
        # boundary between two integers where no request can land. The upper
        # bound is the token domain's, because a threshold is a token count: a
        # tier that began past every count this build can price would be a tier
        # no request could ever select, and an unbounded one is also a value
        # this build cannot render — so it is refused by name here rather than
        # left to raise from inside the code that builds a policy payload.
        check_token_count(self.input_token_threshold, "a long-context tier's threshold")
        if self.input_token_threshold < 1:
            raise PricingError(
                "a long-context tier's input_token_threshold must be an integer of "
                "at least 1, and not a bool. A threshold of zero or below prices "
                "nothing at the base rates, which makes the base rates a number the "
                "table states and never uses. Got "
                f"{_rendered(self.input_token_threshold)}"
            )
        if type(self.applies_at_threshold) is not bool:
            raise PricingError(
                "a long-context tier's applies_at_threshold must be exactly True or "
                "False: it is the difference between the vendor rule '>= threshold' "
                "and the rule '> threshold', and a truthy value of another type "
                f"records a decision nobody made. Got {self.applies_at_threshold!r}"
            )
        _checked_rate(self.input_usd_per_million_tokens, "a long-context input rate")
        _checked_rate(self.output_usd_per_million_tokens, "a long-context output rate")

    def selects(self, input_tokens: int) -> bool:
        """Whether a prompt of this many tokens is priced at these rates."""
        if self.applies_at_threshold:
            return input_tokens >= self.input_token_threshold
        return input_tokens > self.input_token_threshold

    def as_dict(self) -> dict[str, Any]:
        """The stored form. Rates are strings, for the reason every rate here is."""
        return {
            "selected_by": TIER_SELECTOR,
            "input_token_threshold": self.input_token_threshold,
            "applies_at_threshold": self.applies_at_threshold,
            "input_usd_per_million_tokens": str(self.input_usd_per_million_tokens),
            "output_usd_per_million_tokens": str(self.output_usd_per_million_tokens),
        }


@dataclass(frozen=True)
class ModelPrice:
    """What one model's tokens cost, and for how long that was true.

    ``provider`` is part of the price rather than context the caller has to keep
    beside it. A price that could not say whose it was would let a stored
    pricing block be assembled against the wrong provider's policy — the exact
    substitution :class:`UnknownModelPriceError` exists to prevent, arriving one
    layer later.

    It is nonetheless **optional and last of the stated fields**, and that is a
    compatibility decision rather than an oversight. The five fields above it,
    in that order, are the public constructor the single-provider build
    published; callers wrote them positionally as well as by keyword. Making
    ``provider`` a required first argument would break every one of them for a
    field that did not exist when they were written. Every entry this build
    ships states it, and every path that needs one — assembling a stored pricing
    block, looking up a policy — fails closed on the empty default rather than
    guessing a vendor.
    """

    model: str
    input_usd_per_million_tokens: Decimal
    output_usd_per_million_tokens: Decimal
    #: The last date this price is announced to hold, or ``None`` when no end
    #: was announced. ``None`` is "no announced end", never "unknown".
    effective_through: str | None
    #: Why this entry exists, in the run's own terms. Recorded so the manifest
    #: says what an introductory price is rather than only that it ends.
    note: str
    #: Whose price this is. ``""`` is "not stated", which is what a caller
    #: written against the single-provider constructor produces, and it resolves
    #: to no policy at all rather than to a default one.
    provider: str = ""
    #: The published long-context tier, or ``None`` for a model whose vendor
    #: publishes one rate at any prompt length.
    #:
    #: Absent rather than null on every price, and the absence is load-bearing
    #: twice. A field is a claim: a model with no tier has nothing to state, and
    #: the two untiered tables in this build hash to digests that runs already on
    #: disk recorded — adding a key to their model blocks would move both.
    long_context_tier: LongContextTier | None = None

    def __post_init__(self) -> None:
        # Every field, on every path — including the path of a model with no
        # published tier, which is every model in two of the four tables this
        # build ships. A rate that is negative, a float, a NaN or an infinity is
        # refused here because there is nowhere later that can refuse it safely:
        # a negative rate makes a reservation a credit, a cost a rebate and a
        # run's remaining budget rise as it spends.
        _checked_text(self.model, "a price's model")
        _checked_text(self.note, "a price's note")
        _checked_text(
            self.effective_through, "a price's effective_through", allow_none=True
        )
        if type(self.provider) is not str:
            raise PricingError(
                "a price's provider must be a string; it selects the policy a "
                "stored pricing block is assembled from, and a value of another "
                f"type names no provider. Got {_rendered(self.provider)}"
            )
        _checked_rate(self.input_usd_per_million_tokens, "a base input rate")
        _checked_rate(self.output_usd_per_million_tokens, "a base output rate")
        tier = self.long_context_tier
        if tier is None:
            return
        if not isinstance(tier, LongContextTier):
            raise PricingError(
                "a model's long_context_tier must be a LongContextTier or None; the "
                "tier is hashed into run identity as this run's statement of what a "
                "long prompt costs, and an object of another type cannot be that "
                f"statement. Got {type(tier).__name__}"
            )
        # A "long" tier that is cheaper than the base one would make every
        # reservation in this build unsound. A reservation is computed from an
        # *upper* bound on the input tokens, and that is only conservative if a
        # larger prompt cannot come out cheaper than a smaller one.
        for label, long_rate, base_rate in (
            (
                "input",
                tier.input_usd_per_million_tokens,
                self.input_usd_per_million_tokens,
            ),
            (
                "output",
                tier.output_usd_per_million_tokens,
                self.output_usd_per_million_tokens,
            ),
        ):
            if long_rate < base_rate:
                raise PricingError(
                    f"this model's long-context {label} rate of USD {long_rate} is "
                    f"below its base rate of USD {base_rate}. Every reservation this "
                    "build takes is computed from an upper bound on a request's "
                    "input tokens, which is only a bound if a longer prompt cannot "
                    "be cheaper — so a descending tier would authorise requests "
                    "against an amount that does not cover them"
                )

    def rates_for(self, input_tokens: int) -> tuple[Decimal, Decimal]:
        """The two rates a request with this many input tokens is priced at.

        The prompt selects, and what it selects applies to the whole request.
        Stated as its own method because four callers need the same answer —
        the cost below, the reservation the cost guard authorises, the
        settlement it banks and the re-derivation a replay checks a stored row
        against — and a rule applied four times is a rule that can be applied
        three ways.
        """
        tier = self.long_context_tier
        if tier is not None and tier.selects(input_tokens):
            return (
                tier.input_usd_per_million_tokens,
                tier.output_usd_per_million_tokens,
            )
        return (self.input_usd_per_million_tokens, self.output_usd_per_million_tokens)

    def cost(self, *, input_tokens: int, output_tokens: int) -> Decimal:
        """The exact cost of one measured turn, in USD.

        ``scaleb(-6)`` rather than division by a million: shifting the decimal
        exponent is exact for any Decimal, while division goes through the
        arithmetic context's precision and would round a large enough count.

        The multiplication and the addition go through the context too, and the
        ambient one is not wide enough: a count near
        :data:`MAX_EXACT_TOKEN_COUNT` times a rate at the finest accepted scale
        needs more than the default 28 digits, so the product would round —
        silently, and in whichever direction the last digits fell. Both counts
        and both rates are inside the stated domain by the time they get here, so
        :func:`exact_decimal_context` makes every step of this exact.
        """
        billed_input = _checked_tokens(input_tokens, "input_tokens")
        billed_output = _checked_tokens(output_tokens, "output_tokens")
        input_rate, output_rate = self.rates_for(billed_input)
        try:
            with exact_decimal_context():
                return (
                    Decimal(billed_input) * input_rate
                    + Decimal(billed_output) * output_rate
                ).scaleb(-6)
        except DecimalException as exc:  # pragma: no cover - domain excludes it
            raise PricingError(
                "this build could not price a turn exactly at the amounts it was "
                "given. Every accepted rate and count is inside a domain whose "
                "arithmetic is exact by construction, so this is a failure of that "
                "domain rather than of the numbers, and it is refused by name "
                "rather than raised as a decimal condition"
            ) from exc

    def as_dict(self) -> dict[str, Any]:
        """The stored form. Decimals become strings, not JSON numbers.

        A JSON number is a binary float to most readers, which is the one
        representation these values do not survive. The string is exact and it
        round-trips through :class:`~decimal.Decimal` unchanged.

        ``provider`` is deliberately absent, and its absence is load-bearing.
        This projection is what a policy payload stores *under a model key
        inside that provider's own policy*, so repeating the provider here would
        both duplicate it and change the bytes every existing Anthropic run
        hashed. The provider travels on the policy, and on the pricing block
        :mod:`boundarybench.budget` assembles from it.
        """
        block: dict[str, Any] = {
            "input_usd_per_million_tokens": str(self.input_usd_per_million_tokens),
            "output_usd_per_million_tokens": str(self.output_usd_per_million_tokens),
            "effective_through": self.effective_through,
            "note": self.note,
        }
        # Added only where there is a tier, so an untiered model's block is the
        # same four keys it has always been — which is what keeps the Anthropic
        # and Mistral policy digests where the runs on disk recorded them.
        if self.long_context_tier is not None:
            block["long_context_tier"] = self.long_context_tier.as_dict()
        return block


@dataclass(frozen=True)
class PricingPolicy:
    """One provider's whole reviewed price table, and where it came from.

    The identity of a policy is its digest, which is taken over the payload —
    the rates, the models, the source, the date — and not over the version
    string, so a version that was bumped without a rate moving and a rate that
    moved without a version bump are both visible.
    """

    provider: str
    #: The identity of this table. Moves whenever any number in it moves, and is
    #: hashed into the run identity of every cost-capped run, so a run priced
    #: under one table can never be resumed under another.
    policy_version: str
    #: Where the numbers were read from. Recorded rather than cited: a reader
    #: has to be able to check them against the same page.
    source: str
    #: When they were read. Not the same as a price's own effective window —
    #: this is when *this repository* observed it, which is what a later reader
    #: needs in order to know how stale the artefact is.
    recorded_on: str
    models: Mapping[str, ModelPrice]
    #: What this build could *not* establish about the provenance of these
    #: rates, or ``None`` when there was nothing to state.
    #:
    #: Present only where there is a limitation to record, rather than as a null
    #: on every policy. Two reasons, and both matter. A field is a claim: a
    #: policy that states nothing here has nothing to state, and spelling that
    #: as an explicit null would put the same non-claim in four places. And the
    #: payload is hashed — adding a key to every policy to accommodate one
    #: vendor's weaker provenance would move three digests, including the
    #: Anthropic one that runs already on disk recorded.
    snapshot_limitation: str | None = None

    def payload(self) -> dict[str, Any]:
        """The whole policy, as it is recorded in run identity and the manifest."""
        payload: dict[str, Any] = {
            "policy_version": self.policy_version,
            "provider": self.provider,
            "source": self.source,
            "recorded_on": self.recorded_on,
            "unit_tokens": TOKENS_PER_PRICED_UNIT,
            "models": {
                model: price.as_dict() for model, price in sorted(self.models.items())
            },
        }
        if self.snapshot_limitation is not None:
            payload["snapshot_limitation"] = self.snapshot_limitation
        payload["policy_digest"] = digest_of_policy_payload(payload)
        return payload

    def digest(self) -> str:
        """The digest of this table's own payload."""
        return str(self.payload()["policy_digest"])


def _price(
    provider: str,
    model: str,
    input_rate: str,
    output_rate: str,
    *,
    effective_through: str | None,
    note: str,
    long_context_tier: LongContextTier | None = None,
) -> ModelPrice:
    """One reviewed entry. Rates are given as strings so they stay exact."""
    return ModelPrice(
        provider=provider,
        model=model,
        input_usd_per_million_tokens=Decimal(input_rate),
        output_usd_per_million_tokens=Decimal(output_rate),
        effective_through=effective_through,
        note=note,
        long_context_tier=long_context_tier,
    )


def _long_context_tier(
    threshold: int, input_rate: str, output_rate: str, *, applies_at_threshold: bool
) -> LongContextTier:
    """One reviewed tier, with its rates given as strings so they stay exact."""
    return LongContextTier(
        input_token_threshold=threshold,
        applies_at_threshold=applies_at_threshold,
        input_usd_per_million_tokens=Decimal(input_rate),
        output_usd_per_million_tokens=Decimal(output_rate),
    )


#: The approved Anthropic table. The Sonnet rates remain USD 2 / USD 10: on
#: 2026-09-07 the official model overview and pricing pages stated that these are
#: now the standard rates and that the previously announced September increase
#: will not occur. The corrected effective window and note deliberately move the
#: policy identity even though the numeric rates did not change.
#:
#: A read-only proxy rather than a dict: these numbers are inside the identity of
#: every cost-capped run, including the ones already on disk, so a single
#: assignment must not be able to redefine what this build thinks a run cost.
ANTHROPIC_PRICING_MODELS: Mapping[str, ModelPrice] = MappingProxyType(
    {
        "claude-sonnet-5": _price(
            "anthropic",
            "claude-sonnet-5",
            "2",
            "10",
            effective_through=None,
            note=(
                "standard-tier price with no announced end date; official docs "
                "checked on 2026-09-07 state that the previously announced "
                "September increase will not occur"
            ),
        ),
        "claude-haiku-4-5-20251001": _price(
            "anthropic",
            "claude-haiku-4-5-20251001",
            "1",
            "5",
            effective_through=None,
            note=(
                "standard-tier price with no announced end date at the time it "
                "was recorded"
            ),
        ),
    }
)

ANTHROPIC_PRICING_POLICY = PricingPolicy(
    provider="anthropic",
    policy_version="anthropic-standard-2026-09-07",
    source="https://platform.claude.com/docs/en/about-claude/pricing",
    recorded_on="2026-09-07",
    models=ANTHROPIC_PRICING_MODELS,
)

#: The OpenAI table. One model, because this lane pins one model.
#:
#: Tiered. The published page states ``≤272K input tokens`` at USD 0.20 input /
#: USD 1.20 output and ``>272K input tokens`` at USD 0.40 / USD 1.80, so a
#: prompt of exactly 272,000 tokens is still the base tier and the long one
#: begins one token later. That inclusivity is transcribed rather than assumed —
#: see :class:`LongContextTier`.
OPENAI_PRICING_MODELS: Mapping[str, ModelPrice] = MappingProxyType(
    {
        "gpt-5.6-luna": _price(
            "openai",
            "gpt-5.6-luna",
            "0.20",
            "1.20",
            effective_through=None,
            note=(
                "standard-tier price with no announced end date at the time it "
                "was recorded; the account's availability of this model was "
                "confirmed against the official catalog, which is a statement "
                "about access and not about the rate. The rate is tiered on "
                "prompt length: the published page states these rates for a "
                "request of at most 272,000 input tokens and higher rates above "
                "that, so a request is priced whole at whichever tier its own "
                "input-token count selects"
            ),
            long_context_tier=_long_context_tier(
                272_000, "0.40", "1.80", applies_at_threshold=False
            ),
        )
    }
)

OPENAI_PRICING_POLICY = PricingPolicy(
    provider="openai",
    # Moved from ``openai-standard-2026-08-11``, and moved deliberately. The
    # rates in it are the same rates; what changed is that the table now states
    # the published tier rather than only the short-prompt half of it, and a run
    # priced under the flat reading is not comparable with one priced under this.
    policy_version="openai-tiered-2026-08-11",
    source="https://developers.openai.com/api/docs/pricing",
    recorded_on="2026-08-11",
    models=OPENAI_PRICING_MODELS,
)

#: The xAI table, and the one policy in this build whose provenance is weaker
#: than the others'. See :attr:`PricingPolicy.snapshot_limitation`.
#: Tiered, and on the other side of the boundary from OpenAI's. The published
#: pricing page states USD 2 input / USD 6 output for a prompt below 200,000
#: input tokens and USD 4 / USD 12 from 200,000 up, and says in terms that all
#: tokens in a request use the long rates once the threshold is *reached*. So a
#: prompt of exactly 200,000 is already a long-context one.
XAI_PRICING_MODELS: Mapping[str, ModelPrice] = MappingProxyType(
    {
        "grok-4.5": _price(
            "xai",
            "grok-4.5",
            "2",
            "6",
            effective_through=None,
            note=(
                "standard-tier price with no announced end date at the time it "
                "was recorded. The rate is tiered on prompt length: the published "
                "page states these rates below 200,000 input tokens and higher "
                "rates from 200,000 up, and states that every token in a request "
                "uses the higher rates once that threshold is reached"
            ),
            long_context_tier=_long_context_tier(
                200_000, "4", "12", applies_at_threshold=True
            ),
        )
    }
)

XAI_PRICING_POLICY = PricingPolicy(
    provider="xai",
    # Moved from ``xai-standard-2026-08-11`` for the reason the OpenAI policy
    # moved: the table now states the published tier rather than only its
    # short-prompt half.
    policy_version="xai-tiered-2026-08-11",
    # The vendor's pricing page rather than its model page. The tier — its
    # threshold, its inclusivity and both long rates — is stated there, and a
    # policy has to cite the page its numbers were read from.
    source="https://docs.x.ai/developers/pricing",
    recorded_on="2026-08-11",
    models=XAI_PRICING_MODELS,
    snapshot_limitation=(
        "the source page exposes no dated price snapshot, so this policy's rates "
        "and its tier are what the page showed on the recorded_on date rather than "
        "a price the provider published as of a stated day. A later reader cannot "
        "re-fetch the version they were read from, and a cost recorded under this "
        "policy is therefore only as reproducible as this repository's own record "
        "of it"
    ),
)

#: The Mistral table.
MISTRAL_PRICING_MODELS: Mapping[str, ModelPrice] = MappingProxyType(
    {
        "mistral-small-2603": _price(
            "mistral",
            "mistral-small-2603",
            "0.15",
            "0.60",
            effective_through=None,
            note=(
                "standard-tier price with no announced end date at the time it "
                "was recorded; the identifier is the API model identifier from "
                "the provider's own model page, not a display name"
            ),
        )
    }
)

MISTRAL_PRICING_POLICY = PricingPolicy(
    provider="mistral",
    policy_version="mistral-standard-2026-08-11",
    source="https://docs.mistral.ai/inference/pricing",
    recorded_on="2026-08-11",
    models=MISTRAL_PRICING_MODELS,
)

#: Every provider this build holds a reviewed price table for. Keyed by the
#: provider name an adapter identity records, so a run's provider and its price
#: table cannot be two different opinions about who answered.
PROVIDER_PRICING_POLICIES: Mapping[str, PricingPolicy] = MappingProxyType(
    {
        "anthropic": ANTHROPIC_PRICING_POLICY,
        "mistral": MISTRAL_PRICING_POLICY,
        "openai": OPENAI_PRICING_POLICY,
        "xai": XAI_PRICING_POLICY,
    }
)

#: The provider a caller that names none is asking about.
#:
#: Kept, and kept pointing at Anthropic, for one reason: callers written against
#: the single-provider build asked for "the" policy and got this one. They still
#: do, rather than silently getting a merged table whose digest is a number no
#: existing run ever recorded.
PRICING_POLICY_PROVIDER = ANTHROPIC_PRICING_POLICY.provider
#: The default policy's version, source and date, under the names the
#: single-provider build exported them as.
PRICING_POLICY_VERSION = ANTHROPIC_PRICING_POLICY.policy_version
PRICING_POLICY_SOURCE = ANTHROPIC_PRICING_POLICY.source
PRICING_POLICY_RECORDED_ON = ANTHROPIC_PRICING_POLICY.recorded_on


def pricing_policy_for(provider: str) -> PricingPolicy:
    """The reviewed policy for one provider, or a refusal.

    Matched whole and case-sensitively, for the same reason a model is:
    ``OpenAI`` is not ``openai``, and a build that folded case here would accept
    a provider name no adapter identity records.
    """
    policy = PROVIDER_PRICING_POLICIES.get(provider)
    if policy is None:
        raise UnknownModelPriceError(
            f"this build holds no reviewed price policy for provider {provider!r}. "
            f"It covers exactly {sorted(PROVIDER_PRICING_POLICIES)} and matches "
            "provider names whole: a price from another provider's table is not "
            "this model's price however similar the identifier looks, and a run "
            "whose costs cannot be priced cannot be capped"
        )
    return policy


def pricing_policy_payload(provider: str = PRICING_POLICY_PROVIDER) -> dict[str, Any]:
    """One provider's whole policy, as recorded in run identity and the manifest."""
    return pricing_policy_for(provider).payload()


def digest_of_policy_payload(payload: Mapping[str, Any]) -> str:
    """Hash a policy payload over its prices, not over its version string.

    Taken here rather than in :mod:`~boundarybench.runmanifest` so the same
    function answers for a shipped table and for any payload a test or a
    reader wants to compare against it. ``policy_digest`` is excluded from its
    own input, because a digest cannot cover itself.
    """
    # Imported here rather than at module scope: ``runmanifest`` imports the
    # adapter, the evaluator and the suite, and pricing has to stay a leaf that
    # the CLI can consult before any of those are built.
    from boundarybench.jsonsafe import canonical_json_text

    covered = {key: value for key, value in payload.items() if key != "policy_digest"}
    return hashlib.sha256(
        canonical_json_text(covered, "pricing policy").encode("utf-8")
    ).hexdigest()


def pricing_policy_digest(provider: str = PRICING_POLICY_PROVIDER) -> str:
    """The digest of one provider's shipped table."""
    return pricing_policy_for(provider).digest()


def price_for(*, provider: str, model: str) -> ModelPrice:
    """The reviewed price for one exact provider/model pair, or a refusal.

    Both halves are matched whole and case-sensitively.
    ``claude-sonnet-5-latest`` is not ``claude-sonnet-5``: one of them names a
    moving alias and the other names the model this policy priced, and quietly
    treating them as the same would attribute a cost to a model that may never
    have answered. ``grok-4.5`` under ``openai`` is refused for the stronger
    version of the same reason — the model string resolves, but not there.
    """
    policy = pricing_policy_for(provider)
    price = policy.models.get(model)
    if price is None:
        raise UnknownModelPriceError(
            f"pricing policy {policy.policy_version} holds no price for model "
            f"{model!r} under provider {provider!r}. It covers exactly "
            f"{sorted(policy.models)} and matches identifiers whole: no alias, "
            "prefix, case fold or nearest-neighbour price is substituted, because "
            "a cost computed from another model's rate is a fabricated measurement "
            "and a cost cap derived from one cannot be enforced"
        )
    return price


def measured_cost_usd(
    price: ModelPrice, *, input_tokens: int | None, output_tokens: int | None
) -> Decimal | None:
    """The cost of a turn that reported both counts, or ``None`` for one that did not.

    Null propagates on purpose. A response whose usage never arrived was not
    free; it was unmeasured, and pricing half of it as though the other half
    were zero would understate real consumption exactly where the evidence is
    weakest. What that unmeasured turn may still have cost is the cost guard's
    business — see :mod:`boundarybench.budget` — and is recorded as a
    conservative exposure rather than as a measurement.
    """
    if input_tokens is None or output_tokens is None:
        return None
    return price.cost(input_tokens=input_tokens, output_tokens=output_tokens)


__all__ = [
    "ANTHROPIC_PRICING_MODELS",
    "ANTHROPIC_PRICING_POLICY",
    "EXACT_DECIMAL_PRECISION",
    "MAX_EXACT_TOKEN_COUNT",
    "MAX_USD_AMOUNT",
    "MAX_USD_DECIMAL_PLACES",
    "MISTRAL_PRICING_MODELS",
    "MISTRAL_PRICING_POLICY",
    "OPENAI_PRICING_MODELS",
    "OPENAI_PRICING_POLICY",
    "PRICING_POLICY_PROVIDER",
    "PRICING_POLICY_RECORDED_ON",
    "PRICING_POLICY_SOURCE",
    "PRICING_POLICY_VERSION",
    "PROVIDER_PRICING_POLICIES",
    "TIER_SELECTOR",
    "TOKENS_PER_PRICED_UNIT",
    "XAI_PRICING_MODELS",
    "XAI_PRICING_POLICY",
    "LongContextTier",
    "ModelPrice",
    "PricingError",
    "PricingPolicy",
    "UnknownModelPriceError",
    "check_token_count",
    "check_usd_amount",
    "digest_of_policy_payload",
    "exact_decimal_context",
    "measured_cost_usd",
    "price_for",
    "pricing_policy_digest",
    "pricing_policy_for",
    "pricing_policy_payload",
]
