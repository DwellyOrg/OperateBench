"""The hard limits a run may not exceed, and the guard that enforces the cost one.

A benchmark that discovers it overspent by reading the invoice has no cost
control; it has a receipt. So the limits here are checked *before* the action
they bound: an episode limit is proven against the plan before a run directory
exists, and the cost cap is consulted before every single provider request, with
a bound that assumes the request will be as expensive as this run's own pinned
settings allow.

Two quantities are tracked separately and are never added into one number that
claims to be a measurement:

* **Measured cost** is computed from usage the provider actually reported, at
  the reviewed price in :mod:`boundarybench.pricing`. It is what the run is
  known to have consumed.
* **Conservative exposure** is the reservation kept for an attempt that returned
  no usage this build could read — a transport failure, a refused credential, a
  response that named the wrong model. Whether such an attempt was billed is not
  something a client can determine locally, so the guard keeps the full
  upper bound it had reserved for it. This deliberately *over*-states what the
  run may have cost. Under-stating it is the only error that can breach the cap.

What this cannot be is a promise about the provider's billing. A request that
fails in transport after the service has already accepted it may be billed with
no usage ever reaching this process, and no local guard can know. The honest
claim is the one this module implements and names: **authorised exposure** —
measured cost plus the conservative upper bound of every unmeasured attempt —
never exceeds the cap, and no request is dispatched whose own conservative
bound would take it past.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from boundarybench.pricing import (
    ModelPrice,
    PricingError,
    check_usd_amount,
    exact_decimal_context,
    measured_cost_usd,
    price_for,
    pricing_policy_payload,
)
from operatebench.providers.cost import (
    REQUEST_OVERHEAD_TOKEN_ALLOWANCE,
    BudgetError,
    CostCapExceededError,
    CostReservationBreachedError,
    conservative_input_token_bound,
    usd_text,
)

#: Said wherever a cost is reported, because the two numbers answer different
#: questions and a reader who adds them without being told what they are has
#: been misled by the sum.
COST_ACCOUNTING_NOTE = (
    "Cost is reported as two separate quantities and they are never merged into "
    "one measurement. 'measured' is computed from provider-reported input and "
    "output token counts at the run's pinned pricing policy; a turn that "
    "reported no usage contributes null, never zero. 'exposure' is the "
    "conservative upper bound this build reserved for attempts that returned no "
    "readable usage, kept in full because a client cannot determine locally "
    "whether such an attempt was billed. Exposure is therefore an "
    "over-statement by construction. The cap is enforced per request, before it "
    "is dispatched, against measured + exposure + this request's own upper "
    "bound; and a response whose measured cost exceeds the reservation that "
    "authorised it is recorded at its measured value and ends the run as a "
    "reservation breach rather than being committed as though it had been "
    "authorised. So the guarantee is bounded and local: no request is dispatched "
    "that this build's own arithmetic cannot cover, and any settlement beyond "
    "what was reserved is reported rather than absorbed. It is not a guarantee "
    "about the provider's invoice. A request that fails in transport after the "
    "service accepted it may be billed with no usage ever reaching this process, "
    "and token counts are the provider's own; 'reserved exposure' names that "
    "uncertainty instead of hiding it."
)

# -- what a request costs, before and after it is made ------------------------
#
# :class:`BudgetError`, :class:`CostCapExceededError`,
# :class:`CostReservationBreachedError`,
# :data:`REQUEST_OVERHEAD_TOKEN_ALLOWANCE`,
# :func:`conservative_input_token_bound` and :func:`usd_text` are imported above
# from :mod:`operatebench.providers.cost` and re-exported here under the names
# they have always had.
#
# They moved because the shared retry loop is the thing that raises and renders
# them: it reserves against the conservative bound before every request, catches
# the two refusals below, and writes ``usd_text`` of the reservation onto the
# attempt. A second definition of any of the six would drift from the loop that
# actually uses it, and a drifted bound makes every stored reservation of an
# otherwise correct run unre-derivable. They are the same objects, so
# ``isinstance``, the :class:`~boundarybench.adapter.AdapterError` hierarchy and
# every existing ``except`` clause are unchanged.
#
# What stays here is the *policy*: the price table a run pins, the cap an
# operator authorised, and :class:`RunCostGuard`, which is the arithmetic over
# both.


def parse_cost_cap(value: str) -> Decimal:
    """Read an operator-supplied cap as an exact decimal, or refuse it.

    Parsed from the *string* the operator typed rather than through ``float``:
    ``--max-cost-usd 100.10`` is not the binary float nearest to 100.10, and a
    cap that is silently a different number from the one authorised is not the
    authorised cap.

    Checked against the accepted USD domain here, where the value is read,
    rather than left to the arithmetic that later uses it. ``1e1000000`` is a
    perfectly finite Decimal, so a finiteness check passed it and the failure
    surfaced much later and much worse: as a bare ``decimal.Overflow`` raised
    from inside the cost guard at the moment it was authorising a provider
    request, in a type no caller catches and with nothing said about the cap.
    """
    try:
        amount = Decimal(value)
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise BudgetError(
            f"--max-cost-usd must be a decimal amount in USD, got {value!r}"
        ) from exc
    try:
        return check_usd_amount(amount, "--max-cost-usd")
    except PricingError as exc:
        raise BudgetError(str(exc)) from exc


#: Every key a stored pricing block states, and nothing else. Written out rather
#: than derived from whatever ``as_dict`` happens to produce, so adding a field
#: to a price without deciding what a stored one must say is a test failure and
#: not a silently widened contract.
PRICING_BLOCK_KEYS: tuple[str, ...] = (
    "policy_version",
    "policy_digest",
    "provider",
    "source",
    "recorded_on",
    "unit_tokens",
    "model",
    "input_usd_per_million_tokens",
    "output_usd_per_million_tokens",
    "effective_through",
    "note",
)

#: The one further key a *tiered* model's block states, and only a tiered one.
#:
#: Kept out of :data:`PRICING_BLOCK_KEYS` rather than added to it, because the
#: key set is checked exactly and two of the four policies this build ships
#: publish no tier. A block that carried ``"long_context_tier": null`` on every
#: model would state a non-claim in four places and would move two digests that
#: runs already on disk recorded.
TIERED_PRICING_BLOCK_KEY = "long_context_tier"


def pricing_block_keys(price: ModelPrice) -> tuple[str, ...]:
    """Exactly the fields a stored block for *this* price must state.

    Derived from the price rather than fixed, because whether a model has a
    published tier is a property of the model. Untiered models keep the exact
    key set they have always had — which is what keeps their stored blocks, and
    the digests over them, byte-identical.
    """
    if price.long_context_tier is None:
        return PRICING_BLOCK_KEYS
    return (*PRICING_BLOCK_KEYS, TIERED_PRICING_BLOCK_KEY)


def _exactly_equal(got: Any, want: Any) -> bool:
    """Whether two decoded JSON values are the same value *and* the same type.

    Recursive over objects, because a stored pricing block is no longer flat: a
    tier is an object, and an edit inside one has to be caught by the same rule
    that catches an edit beside it. No field of a pricing block is an array, so
    there is no array case here rather than an unreachable one.

    Type-exactness matters at every level for the reason it matters at the top:
    ``"200000"`` and ``200000`` are the same number to a reader who does not
    care, ``1`` and ``True`` are the same value to Python, and a JSON number is a
    binary float to most parsers.
    """
    if type(got) is not type(want):
        return False
    if isinstance(want, Mapping):
        if set(got) != set(want):
            return False
        return all(_exactly_equal(got[key], want[key]) for key in want)
    return bool(got == want)


def shipped_pricing_block(price: ModelPrice) -> dict[str, Any]:
    """The one pricing block this build writes, and the only one it accepts.

    Self-describing on purpose: the provider and the unit the rates are quoted
    in travel with the rates. A block that names rates without naming what they
    are per is a number a later reader has to guess the meaning of, and a guess
    is how a per-million rate becomes a per-thousand one.

    The policy is looked up from the *price's own* provider rather than from a
    build-wide default. Once more than one provider is priced, a default would
    assemble one vendor's version, source and digest around another vendor's
    rates — a block that is internally consistent, passes every shape check and
    cites the wrong page.
    """
    policy = pricing_policy_payload(price.provider)
    return {
        "policy_version": policy["policy_version"],
        "policy_digest": policy["policy_digest"],
        "provider": policy["provider"],
        "source": policy["source"],
        "recorded_on": policy["recorded_on"],
        "unit_tokens": policy["unit_tokens"],
        "model": price.model,
        **price.as_dict(),
    }


def check_stored_pricing_block(
    stored: Mapping[str, Any], *, provider: str, model: str
) -> ModelPrice:
    """Prove a stored pricing block *is* the shipped one, and return that price.

    Key-exact, type-exact and value-exact against
    :func:`shipped_pricing_block`, in that order, because each check answers a
    different adversarial edit. Rewriting a stored rate while retaining the
    model, policy version and digest must be refused rather than replaced with a
    shipped value.

    Type-exact matters as much as value-exact: ``"2"`` and ``2`` are the same
    amount to a reader who does not care, and a JSON number is a binary float to
    most parsers, which is the one representation these values do not survive. So
    the comparison is on the exact JSON type as well as the value, and a stored
    ``1000000.0`` is not the integer ``1000000``.

    The price returned is the shipped one, looked up again rather than
    reconstructed from the file. That is unchanged and remains the point: a rate
    somebody typed into a manifest is not a reviewed price. What is new is that a
    stored block which disagrees with the shipped one now *fails* instead of
    being overwritten by it.
    """
    if not isinstance(stored, Mapping):
        raise BudgetError("cost_controls: 'pricing' must be a JSON object or null")
    keys = set(map(str, stored))
    # The base key set, plus the tier key for a model that has one. Checked
    # against the *shipped* price rather than against whatever the file states,
    # so a stored tier on an untiered model is an unknown field and a dropped
    # tier on a tiered one is a missing field — which are the two edits that
    # would halve a recorded long-context cost.
    if "model" not in keys:
        raise BudgetError(
            "cost_controls: pricing block is missing required field(s) ['model']"
        )
    if stored["model"] != model:
        raise BudgetError(
            "cost_controls: the stored pricing block names a model other "
            "than the one this run's adapter identity pins, so it prices "
            "something this run did not execute"
        )
    try:
        price = price_for(provider=provider, model=model)
    except PricingError as exc:
        raise BudgetError(str(exc)) from exc
    required = pricing_block_keys(price)
    unknown = sorted(keys - set(required))
    if unknown:
        raise BudgetError(
            f"cost_controls: pricing block states unknown field(s) {unknown}; a "
            "price this build did not write is not a price it reviewed"
        )
    missing = sorted(set(required) - keys)
    if missing:
        raise BudgetError(
            f"cost_controls: pricing block is missing required field(s) {missing}"
        )
    expected = shipped_pricing_block(price)
    for key in required:
        got, want = stored[key], expected[key]
        if not _exactly_equal(got, want):
            raise BudgetError(
                f"cost_controls: pricing block field {key!r} is {got!r}, and the "
                f"pricing policy this build ships states {want!r}. A run's costs "
                "are only comparable within one reviewed price table, so a stored "
                "block that disagrees with the shipped one — in value or in JSON "
                "type — is refused rather than re-priced under a table it never "
                "used"
            )
    return price


@dataclass(frozen=True)
class CostControls:
    """The hard limits one run was authorised to spend, as one unit.

    Kept together because they are only meaningful together and because they go
    into configuration identity as a block: a run that changes its cap or its
    episode limit is a different run, and a resume of the old directory refuses
    it rather than continuing under limits nobody approved.
    """

    #: The whole run's ceiling, or ``None`` for a run with no cost control. A
    #: non-null cap requires a price: a cap that cannot be measured against
    #: anything cannot be enforced.
    max_cost_usd: Decimal | None
    #: The hard ceiling on planned and executed episodes, or ``None``.
    max_episodes: int | None
    #: The reviewed price this run's costs are computed at.
    price: ModelPrice | None

    def __post_init__(self) -> None:
        if self.max_cost_usd is not None:
            # The whole accepted USD domain, on construction: a cap is the one
            # amount every other number here is compared against, so an amount
            # the arithmetic cannot hold exactly must not reach a guard at all.
            try:
                check_usd_amount(self.max_cost_usd, "max_cost_usd")
            except PricingError as exc:
                raise BudgetError(str(exc)) from exc
            if self.max_cost_usd <= 0:
                raise BudgetError(
                    f"max_cost_usd must be a positive amount, got "
                    f"{self.max_cost_usd!r}; a zero cap authorises nothing and is "
                    "refused rather than read as 'no limit'"
                )
            if self.price is None:
                raise BudgetError(
                    "a cost cap requires a reviewed price for the model under test. "
                    "Without one there is nothing to measure spend against, so the "
                    "cap could not be enforced and would be a claim rather than a "
                    "control"
                )
        # ``bool`` is an ``int`` subclass, so the type test is exact.
        if self.max_episodes is not None and (
            type(self.max_episodes) is not int or self.max_episodes < 1
        ):
            raise BudgetError(
                f"max_episodes must be a positive integer, got {self.max_episodes!r}"
            )

    @property
    def enforces_cost(self) -> bool:
        return self.max_cost_usd is not None

    def as_dict(self) -> dict[str, Any]:
        """The stored form, hashed into run identity.

        The cap is a string, not a JSON number: a reader that parses it as a
        binary float gets a different amount from the one authorised, and this
        block is the record of what was authorised.
        """
        pricing = None if self.price is None else shipped_pricing_block(self.price)
        return {
            "max_cost_usd": None if self.max_cost_usd is None else str(self.max_cost_usd),
            "max_episodes": self.max_episodes,
            "pricing": pricing,
        }

    @classmethod
    def from_stored(
        cls, raw: Mapping[str, Any], *, provider: str, model: str
    ) -> CostControls:
        """Read a stored block strictly, and re-derive its price from this build.

        The price is looked up again rather than trusted from the file. A stored
        rate is editable; the policy this build ships is not, and a run resumed
        against a rate somebody typed into the manifest would compute costs
        nobody reviewed.

        Looking it up again is necessary and was not sufficient: the stored block
        was then *ignored* rather than checked, so an edited rate was silently
        overwritten by the shipped one. The whole block is now compared against
        the shipped policy key-exact, type-exact and value-exact before it is
        accepted — see :func:`check_stored_pricing_block`.
        """
        if not isinstance(raw, Mapping):
            raise BudgetError("cost_controls must be a JSON object")
        unknown = sorted(set(map(str, raw)) - {"max_cost_usd", "max_episodes", "pricing"})
        if unknown:
            raise BudgetError(f"cost_controls: unknown field(s) {unknown}")
        for key in ("max_cost_usd", "max_episodes", "pricing"):
            if key not in raw:
                raise BudgetError(f"cost_controls: missing required field {key!r}")

        stored_cap = raw["max_cost_usd"]
        cap: Decimal | None = None
        if stored_cap is not None:
            if not isinstance(stored_cap, str):
                raise BudgetError(
                    "cost_controls: 'max_cost_usd' must be a decimal *string* or "
                    f"null, got {stored_cap!r}; a JSON number is a binary float to "
                    "most readers and is not the amount that was authorised"
                )
            cap = parse_cost_cap(stored_cap)

        episodes = raw["max_episodes"]
        if episodes is not None and type(episodes) is not int:
            raise BudgetError(
                f"cost_controls: 'max_episodes' must be an integer or null, got "
                f"{episodes!r}"
            )

        stored_pricing = raw["pricing"]
        price: ModelPrice | None = None
        if stored_pricing is not None:
            price = check_stored_pricing_block(
                stored_pricing, provider=provider, model=model
            )
        return cls(max_cost_usd=cap, max_episodes=episodes, price=price)


class RunCostGuard:
    """One run's authorised budget, enforced before every provider request.

    Held by the runner and by the adapter that makes the requests: the runner
    seeds it from the ledger and reads its per-episode totals into each row, and
    the adapter asks it for permission and reports back what came of it. One
    object, so the number that authorises a request and the number that is
    persisted cannot drift apart.
    """

    def __init__(
        self,
        *,
        controls: CostControls,
        max_output_tokens: int,
        max_attempts_per_turn: int,
    ) -> None:
        if not controls.enforces_cost or controls.price is None:
            raise BudgetError(
                "a cost guard is only built for a run with a cost cap and a "
                "reviewed price; a run without them has no budget to enforce and "
                "must not carry a guard that would appear to enforce one"
            )
        if type(max_output_tokens) is not int or max_output_tokens < 1:
            raise BudgetError("max_output_tokens must be a positive integer")
        if type(max_attempts_per_turn) is not int or max_attempts_per_turn < 1:
            raise BudgetError("max_attempts_per_turn must be a positive integer")
        self._controls = controls
        self._price = controls.price
        self._cap: Decimal = controls.max_cost_usd or Decimal(0)
        self._max_output_tokens = max_output_tokens
        self._max_attempts = max_attempts_per_turn
        self._measured = Decimal(0)
        self._exposure = Decimal(0)
        self._episode_measured = Decimal(0)
        self._episode_exposure = Decimal(0)
        self._pending: tuple[Decimal, int] | None = None
        self._seeded = False
        self._breached = False

    # -- binding -------------------------------------------------------------

    @property
    def controls(self) -> CostControls:
        """The limits this guard enforces. Read-only, and never the manifest's."""
        return self._controls

    def bind_to(self, controls: CostControls) -> None:
        """Prove this guard enforces exactly these limits, or refuse to run.

        A caller-supplied guard with controls that differ from the manifest would
        enforce spending that the durable run identity does not authorise. The
        guard is therefore bound to the manifest's own block before anything is
        dispatched, and the comparison is over the stored form rather than the
        objects: ``as_dict`` is exactly what goes into configuration identity, so
        agreement here means the cap being enforced is the cap that was hashed,
        recorded and will be reported. Two caps that are numerically equal but
        stored differently — ``100`` and ``100.0`` — are two different run
        identities and are refused as a mismatch, because one of them is not the
        run this directory records.
        """
        if not isinstance(controls, CostControls):
            raise BudgetError(
                "a cost guard is bound to a run's CostControls, not to "
                f"{type(controls).__name__}"
            )
        mine, theirs = self._controls.as_dict(), controls.as_dict()
        if mine != theirs:
            raise BudgetError(
                "this run's cost guard does not enforce the cost_controls its "
                f"manifest records. The manifest authorises {theirs}, and the "
                f"guard holds {mine}. A run may only spend under the limits its "
                "own durable record states, so no request is dispatched under a "
                "guard that was never bound to them"
            )

    # -- state ---------------------------------------------------------------

    @property
    def measured_usd(self) -> Decimal:
        return self._measured

    @property
    def reservation_breached(self) -> bool:
        """Whether a settlement ever exceeded the reservation that allowed it."""
        return self._breached

    @property
    def exposure_usd(self) -> Decimal:
        return self._exposure

    @property
    def episode_measured_usd(self) -> Decimal:
        return self._episode_measured

    @property
    def episode_exposure_usd(self) -> Decimal:
        return self._episode_exposure

    @property
    def committed_usd(self) -> Decimal:
        """Everything the cap is enforced against: measured, exposure, in flight.

        Summed in :func:`~boundarybench.pricing.exact_decimal_context`, like
        every other total here. The ambient precision is below what a run's
        accumulated amounts can need, and a total that rounded would be a cap
        enforced against a number that is nobody's arithmetic.
        """
        with exact_decimal_context():
            pending = Decimal(0) if self._pending is None else self._pending[0]
            return self._measured + self._exposure + pending

    @property
    def remaining_usd(self) -> Decimal:
        """What is left, floored at zero: a negative budget is not head-room.

        The floor is a report-side courtesy and never a control. What made the
        remaining budget *rise* was a negative measured cost, and that is now
        impossible at the source: no negative rate can be stated, so no
        settlement can be negative.
        """
        with exact_decimal_context():
            remaining = self._cap - self.committed_usd
        return remaining if remaining > 0 else Decimal(0)

    def seed(self, *, measured_usd: Decimal, exposure_usd: Decimal) -> None:
        """Adopt the spend a resumed run's ledger already records.

        Called once, before the first request of an invocation. Seeding twice
        is refused because the second call would silently replace the first: a
        resume that forgot prior spend is exactly the failure a durable cost cap
        exists to prevent.
        """
        if self._seeded:
            raise BudgetError(
                "this cost guard has already adopted a run's recorded spend; "
                "seeding twice would discard it, and a run that can forget what it "
                "spent has no cap"
            )
        for name, value in (
            ("measured_usd", measured_usd),
            ("exposure_usd", exposure_usd),
        ):
            # The same domain the cap and the rates are held to. A total
            # recovered from a ledger is arithmetic this build performed and
            # wrote down, so anything outside the domain it can hold exactly is
            # a total it did not write.
            try:
                check_usd_amount(value, f"{name} recovered from the ledger")
            except PricingError as exc:
                raise BudgetError(str(exc)) from exc
        self._measured = measured_usd
        self._exposure = exposure_usd
        self._seeded = True

    def begin_episode(self) -> None:
        """Start this episode's own accounting. Run totals are untouched."""
        if self._pending is not None:
            raise BudgetError(
                "an episode cannot begin while a provider request is still "
                "outstanding; the previous request was neither settled nor forfeited"
            )
        self._episode_measured = Decimal(0)
        self._episode_exposure = Decimal(0)

    # -- authorisation -------------------------------------------------------

    def reservation_for(self, *, input_tokens_upper_bound: int) -> Decimal:
        """The most one attempt at this request can cost, at this run's settings."""
        return self._price.cost(
            input_tokens=input_tokens_upper_bound,
            output_tokens=self._max_output_tokens,
        )

    def turn_upper_bound_usd(self, *, input_tokens_upper_bound: int) -> Decimal:
        """The most one whole turn can cost, retries included.

        Reported rather than reserved in one lump: the guard is consulted before
        each attempt, so a turn that is refused on its second retry has only
        spent what its first attempts exposed. This is the number a plan is
        checked against before a run starts.
        """
        with exact_decimal_context():
            return (
                self.reservation_for(input_tokens_upper_bound=input_tokens_upper_bound)
                * self._max_attempts
            )

    def authorize(self, *, input_tokens_upper_bound: int) -> Decimal:
        """Permit one provider request, or refuse it. Returns the reservation.

        Refusal is by strict excess: a request whose bound exactly equals the
        remaining budget is allowed, because spending exactly what was
        authorised is not exceeding it.
        """
        if self._pending is not None:
            raise BudgetError(
                "a provider request is already outstanding on this guard; one "
                "request is authorised at a time, or the same budget would be "
                "reserved twice"
            )
        reservation = self.reservation_for(
            input_tokens_upper_bound=input_tokens_upper_bound
        )
        with exact_decimal_context():
            committed = self._measured + self._exposure + reservation
        if committed > self._cap:
            raise CostCapExceededError(
                "this run's remaining authorised budget cannot cover the next "
                f"provider request. The cap is USD {self._cap}, the run has "
                f"measured USD {self._measured} of cost and holds USD "
                f"{self._exposure} of conservative exposure for attempts that "
                f"returned no usage, and this request's own upper bound is USD "
                f"{reservation} at this run's pinned output ceiling of "
                f"{self._max_output_tokens} token(s). The request was not made. "
                "Every number here is this build's own arithmetic over its own "
                "pinned settings and its recorded pricing policy; none of it "
                "comes from the provider"
            )
        self._pending = (
            reservation,
            input_tokens_upper_bound + self._max_output_tokens,
        )
        return reservation

    def settle(
        self, reservation: Decimal, *, input_tokens: int | None, output_tokens: int | None
    ) -> Decimal | None:
        """Replace an outstanding reservation with what the response reported.

        Returns the measured cost, or ``None`` when the response carried no
        usable counts — in which case the reservation is forfeited instead,
        because an unmeasured response is not a free one.

        A count the price policy refuses — a negative, a non-integer, anything a
        token count cannot be — is treated as *unmeasured* rather than allowed
        to raise. The reservation is resolved either way: a guard that dropped a
        reservation on the floor because a provider reported nonsense would lose
        the exposure it was holding, which is the one direction that breaches
        the cap. The runner separately refuses the malformed measurement.

        A measurement **above** the reservation cannot settle quietly because it
        commits spend the cap did not authorise. The measured cost is still
        banked — it is what the run actually consumed, and understating it is the
        worse error — and
        :class:`CostReservationBreachedError` is raised in its place, so the turn
        ends as a recorded breach and the run stops. Equality is not a breach:
        spending exactly what was reserved is spending what was authorised.
        """
        try:
            measured = measured_cost_usd(
                self._price, input_tokens=input_tokens, output_tokens=output_tokens
            )
        except PricingError:
            measured = None
        self._release(reservation)
        with exact_decimal_context():
            if measured is None:
                self._exposure += reservation
                self._episode_exposure += reservation
                return None
            self._measured += measured
            self._episode_measured += measured
        if measured > reservation:
            self._breached = True
            raise CostReservationBreachedError(
                "this response measured USD "
                f"{usd_text(measured)} against a reservation of USD "
                f"{usd_text(reservation)}, which is the most this run authorised "
                "the request to cost at its pinned output ceiling of "
                f"{self._max_output_tokens} token(s). The request was already "
                "made and the tokens were already consumed, so the measured cost "
                "is recorded at its measured value rather than clipped; what is "
                f"refused is treating it as authorised against the USD "
                f"{usd_text(self._cap)} cap. The run stops here. Every number in "
                "this message is this build's own arithmetic over its own pinned "
                "settings and its recorded pricing policy",
                measured_usd=measured,
                reservation_usd=reservation,
                cap_usd=self._cap,
            )
        return measured

    def forfeit(self, reservation: Decimal) -> None:
        """Keep an outstanding reservation as exposure: nothing was measured."""
        self._release(reservation)
        with exact_decimal_context():
            self._exposure += reservation
            self._episode_exposure += reservation

    def cancel(self, reservation: Decimal, *, input_tokens_upper_bound: int) -> None:
        """Release a reservation when the authorised request never dispatched."""
        recomputed = self.reservation_for(
            input_tokens_upper_bound=input_tokens_upper_bound
        )
        exposure = input_tokens_upper_bound + self._max_output_tokens
        if recomputed != reservation:
            raise BudgetError(
                "the cancellation token bound does not reproduce the reservation "
                "being released"
            )
        self._release(reservation, expected_exposure=exposure)

    def _release(
        self, reservation: Decimal, *, expected_exposure: int | None = None
    ) -> None:
        if self._pending is None:
            raise BudgetError(
                "no provider request is outstanding on this guard, so there is no "
                "reservation to resolve"
            )
        pending_reservation, pending_exposure = self._pending
        if reservation != pending_reservation:
            raise BudgetError(
                "the reservation being resolved is not the one this guard is "
                "holding; resolving a different amount would leave the run's "
                "recorded exposure describing a request that was never authorised"
            )
        if expected_exposure is not None and expected_exposure != pending_exposure:
            raise BudgetError(
                "the cancellation token exposure is not the one this guard is "
                "holding; the outstanding reservation was not changed"
            )
        self._pending = None

    # -- reporting -----------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        """What the run report and the manifest say about this run's spend."""
        with exact_decimal_context():
            authorised = self._measured + self._exposure
        return {
            "max_cost_usd": usd_text(self._cap),
            "measured_usd": usd_text(self._measured),
            "exposure_usd": usd_text(self._exposure),
            "authorised_exposure_usd": usd_text(authorised),
            "remaining_usd": usd_text(self.remaining_usd),
            "reservation_breached": self._breached,
            "note": COST_ACCOUNTING_NOTE,
        }


__all__ = [
    "COST_ACCOUNTING_NOTE",
    "PRICING_BLOCK_KEYS",
    "REQUEST_OVERHEAD_TOKEN_ALLOWANCE",
    "TIERED_PRICING_BLOCK_KEY",
    "BudgetError",
    "CostCapExceededError",
    "CostControls",
    "CostReservationBreachedError",
    "RunCostGuard",
    "check_stored_pricing_block",
    "conservative_input_token_bound",
    "parse_cost_cap",
    "pricing_block_keys",
    "shipped_pricing_block",
    "usd_text",
]
