"""Explicit paid capabilities for the shared executor; no reference policy here."""

from __future__ import annotations

import copy
import importlib
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import httpx

from operatebench.agents.pricing import GuardEvent, LifecyclePricingPolicy
from tools.aggregate_budget import SettlementBoundsError, SharedGuard, canonical, decimal
from tools.three_flow_admission import VARIABLES, digest
from tools.three_flow_budget import META_UNCAPPED_PROFILE, CampaignBudget
from tools.three_flow_http import HTTPCampaignTransport
from tools.three_flow_profiles import LATE_NOVEMBER_PROFILE, SELECTED_SIX


class AdmittedGuard(SharedGuard):
    """Reserve complete provider input maximum, including invisible wire overhead.

    Strict admission requires complete usage guarantees. Explicit reported-usage
    diagnostic consent instead accepts unverified completeness/hidden billing/fees.
    Both settle reported tokens at envelope rates, never as an invoice claim;
    malformed or absent usage retains the reservation as unknown exposure.
    """

    def __init__(self, *, profile: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.profile = profile

    @property
    def _allows_no_money_cap(self) -> bool:
        # SharedGuard already admits the generic optional controller. Meta has
        # a distinct validated genesis and reuses that same nullable primitive.
        return super()._allows_no_money_cap or (
            getattr(self.budget, "controller_profile", None) == META_UNCAPPED_PROFILE
        )

    @property
    def diagnostic(self) -> bool:
        return self.profile.get("bounds_policy") is not None

    @property
    def wire_output_max(self) -> int:
        # River InferencePrompt.max_tokens is protobuf int64. HTTP uses the
        # interoperable exact JSON integer range, not a provider physical max.
        return 2**63 - 1 if self.profile["endpoint"] == "api.river.ai:443" else 2**53 - 1

    def output_bound(self, input_bound: int) -> int:
        """Operational request bound, never a claim about physical output capacity."""
        from fractions import Fraction

        from operatebench.providers.cost import CostCapExceededError

        p = self.profile
        if type(input_bound) is not int or input_bound < 0:
            raise ValueError("invalid complete prompt bound")
        if p["input_max"] is not None and input_bound > p["input_max"]:
            raise ValueError("request exceeds admitted input")
        remaining = self.budget.totals()["remaining"]
        output = min(
            p["context_max"] - input_bound,
            p["output_max"] or self.wire_output_max,
            p.get("owner_output_cap_tokens", self.wire_output_max),
            self.wire_output_max,
        )
        if remaining is not None:
            affordable = (
                remaining * 1000000
                - input_bound * Fraction(self.policy.input_usd_per_mtok)
            ) // Fraction(self.policy.output_usd_per_mtok)
            output = min(output, affordable)
        if output <= 0:
            raise CostCapExceededError("context or shared monetary exposure exhausted")
        return int(output)

    def check_context(self, input_bound: int, output_bound: int) -> None:
        p = self.profile
        if (
            type(input_bound) is not int
            or input_bound < 0
            or type(output_bound) is not int
            or not 0 < output_bound <= self.wire_output_max
            or (p["input_max"] is not None and input_bound > p["input_max"])
            or (p["output_max"] is not None and output_bound > p["output_max"])
            or output_bound > p.get("owner_output_cap_tokens", self.wire_output_max)
            or input_bound + output_bound > p["context_max"]
        ):
            raise ValueError("request exceeds admitted provider context/output")

    def settle_ember(
        self,
        reservation: Decimal,
        *,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
    ) -> Decimal:
        from operatebench.providers.cost import CostReservationBreachedError

        ident = self._check(reservation)
        try:
            actual = cast(CampaignBudget, self.budget).settle_ember(
                ident,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
            )
        except SettlementBoundsError as exc:
            self.forfeit(reservation)
            raise CostReservationBreachedError(
                "usage exceeds admitted bounds; full wire bound retained as unknown",
                measured_usd=None,
                reservation_usd=decimal(Fraction(self.budget.records[ident]["bound"])),
                cap_usd=self.cap_usd,
            ) from exc
        self._measured += actual
        self._last = GuardEvent(
            reservation_usd=reservation, measured_usd=actual, forfeited_usd=None
        )
        self._pending_events.append(self._last)
        self._outstanding = Decimal(0)
        self.ticket = None
        return actual

    def authorize(self, *, input_tokens_upper_bound: int) -> Decimal:
        self.check_context(input_tokens_upper_bound, self._max_output_tokens)
        bound = input_tokens_upper_bound if self.diagnostic else self.profile["input_max"]
        return super().authorize(input_tokens_upper_bound=bound)


def live_factory(
    admission: dict[str, Any], keys: dict[str, str], *, network: Any = None
) -> Any:
    if admission.get("roster_profile") == LATE_NOVEMBER_PROFILE:
        # Resolve OpenAI's lazy resource modules before concurrent workers start.
        # Importing chat and Responses simultaneously can deadlock module locks.
        importlib.import_module("openai.resources.chat.completions")
        importlib.import_module("openai.resources.responses")
        importlib.import_module("openai.types.chat.chat_completion")

    def make(slot: dict[str, Any], budget: Any) -> Any:
        late = admission.get("roster_profile") == LATE_NOVEMBER_PROFILE
        if late:
            from tools.three_flow_campaign import assignments, correction_slot

            originals = [
                original
                for original in assignments(
                    admission["campaign_id"], LATE_NOVEMBER_PROFILE
                )
                if original["trial_id"] in admission["selected"]
            ]
            if admission["operation"] == "correction":
                allowed = [
                    correction_slot(
                        original,
                        admission["attempt_id"],
                        admission["corrections"][original["trial_id"]]["reason"],
                        admission["corrections"][original["trial_id"]]["evidence_sha256"],
                    )
                    for original in originals
                ]
            elif admission["operation"] in ("initial", "resume"):
                allowed = originals
            else:
                raise ValueError("unknown late November operation")
            if (
                canonical(slot) not in {canonical(item) for item in allowed}
                or getattr(budget, "roster_profile", None) != LATE_NOVEMBER_PROFILE
                or (
                    None
                    if budget.cap is None
                    else Decimal(budget.cap.numerator) / Decimal(budget.cap.denominator)
                )
                != (
                    None
                    if admission["cap_usd"] is None
                    else Decimal(admission["cap_usd"])
                )
            ):
                raise ValueError(
                    "late November selected slot and budget binding required"
                )
        if slot["provider"] == "gemini":
            from tools.three_flow_gemini import ROSTER_PROFILE as GEMINI_ROSTER
            from tools.three_flow_gemini import GeminiCampaignTransport, gemini_profile

            profile = gemini_profile(
                slot["model"], LATE_NOVEMBER_PROFILE if late else GEMINI_ROSTER
            )
            expected = (
                profile.settings(profile.output)
                if network is not None
                else profile.live_settings(profile.output)
            )
            if admission["profiles"][slot["model"]] != expected:
                raise ValueError("Gemini admitted model profile differs")
            guard = SharedGuard(
                budget=budget,
                cell=slot["trial_id"],
                policy=LifecyclePricingPolicy(
                    "gemini-standard-estimated-v1",
                    Decimal(profile.input_rate),
                    Decimal(profile.output_rate),
                ),
                max_output_tokens=profile.output,
            )
            gemini_inner = None if network is None else network.http(slot)
            if network is not None and not isinstance(gemini_inner, httpx.MockTransport):
                raise ValueError("synthetic Gemini requires MockTransport")
            gemini_transport = GeminiCampaignTransport.live(
                model=slot["model"],
                api_key=keys[VARIABLES["gemini"]],
                guard=guard,
                max_output_tokens=profile.output,
                admitted=True,
                inner=gemini_inner,
                roster_profile=LATE_NOVEMBER_PROFILE if late else GEMINI_ROSTER,
            )
            if network is None:
                gemini_transport.settings.update(
                    admission=True,
                    admission_provenance="source-bound-admission-reference-v1",
                    admission_sha256=digest(admission),
                    admission_profile_sha256=digest(admission["profiles"][slot["model"]]),
                    limit_policy=copy.deepcopy(admission.get("limit_policy")),
                    **({"cap_usd": admission["cap_usd"]} if late else {}),
                )
            return gemini_transport
        p = admission["profiles"][slot["model"]]
        diagnostic = p.get("bounds_policy") is not None
        maximum = min(
            p["output_max"] or p["context_max"],
            p.get("owner_output_cap_tokens", p["context_max"]),
            2**63 - 1 if slot["provider"] == "river" else 2**53 - 1,
        )
        guard = AdmittedGuard(
            profile=p,
            budget=budget,
            cell=slot["trial_id"],
            policy=LifecyclePricingPolicy(
                "admitted-envelope-" + digest(p),
                Decimal(p["input_usd_per_million"]),
                Decimal(p["output_usd_per_million"]),
            ),
            max_output_tokens=maximum,
        )
        key = keys[VARIABLES[slot["provider"]]]
        inner: Any = None
        transport: Any
        try:
            if slot["provider"] == "river":
                from tools.three_flow_river import (
                    GRPC_OPTIONS,
                    ExplicitRiverChannel,
                    RiverCampaignTransport,
                )

                if network is None:
                    grpc = importlib.import_module("grpc")

                    inner = ExplicitRiverChannel(
                        api_key=key,
                        channel_factory=lambda endpoint, credentials, options: (
                            grpc.secure_channel(
                                endpoint,
                                credentials,
                                options=(*options, ("grpc.enable_http_proxy", 0)),
                            )
                        ),
                        credentials_factory=grpc.ssl_channel_credentials,
                    )
                else:
                    inner = network.river(slot)
                    if getattr(inner, "offline_mock", False) is not True:
                        raise ValueError("synthetic capability must be a mock channel")
                transport = RiverCampaignTransport(
                    model=slot["model"],
                    assets=Path(admission["assets"]["root"]),
                    channel=inner,
                    guard=guard,
                    max_output_tokens=maximum,
                    channel_options=GRPC_OPTIONS,
                    network_timeout=p["network_timeout_seconds"],
                    mode_profile="ripple-default-v1"
                    if late
                    else admission.get("mode_profile", "legacy-v1"),
                )
                if network is not None:
                    inner.tokenizer = transport.tokenizer
            else:
                inner = (
                    httpx.HTTPTransport(retries=0, trust_env=False)
                    if network is None
                    else network.http(slot)
                )
                if network is not None and not isinstance(inner, httpx.MockTransport):
                    raise ValueError("synthetic capability must be MockTransport")
                transport = HTTPCampaignTransport(
                    provider=slot["provider"],
                    model=slot["model"],
                    api_key=key,
                    inner=inner,
                    guard=guard,
                    max_output_tokens=maximum,
                    network_timeout=p["network_timeout_seconds"],
                    mode_profile="off-or-minimum-v1"
                    if late
                    else admission.get("mode_profile", "legacy-v1"),
                    roster_profile=LATE_NOVEMBER_PROFILE if late else "legacy-v1",
                    transport_policy=(
                        admission.get("transport_policy", "legacy-v1")
                        if slot["provider"] == "mistral"
                        or (
                            slot["provider"] == "river_http"
                            and (slot["provider"], slot["model"]) in SELECTED_SIX
                        )
                        else "legacy-v1"
                    ),
                    context_tokens=p["context_max"],
                    admitted_cache_envelope=True,
                )
            transport.settings.update(
                admission=True,
                ambient_proxy=False,
                model_contract_verified=not diagnostic and "owner_pricing" not in p,
                bounds_policy=p.get("bounds_policy"),
                dedicated_output_max=p["output_max"],
                context_request_bound=p["context_max"],
                input_measure="native-tokenizer"
                if slot["provider"] == "river"
                else "conservative-serialized-payload",
                billing_assurance="admitted usage accounting, not invoice guarantee",
                admission_profile_sha256=digest(p),
                admission_sha256=digest(admission),
                accounting="conservative all-tier envelope; usage estimate, not invoice",
            )
            if late and slot["provider"] == "meta":
                transport.settings["accounting"] = (
                    "reported input/cache/output at published tier; "
                    "conservative reservation; estimate not invoice"
                )
            if "owner_output_cap_tokens" in p:
                # Selected operational ceiling, not an API/provider maximum.
                # Full profile and settings already bind admission/identity/replay.
                transport.settings["owner_output_cap_tokens"] = p[
                    "owner_output_cap_tokens"
                ]
            if "limit_policy" in admission:
                transport.settings.update(
                    limit_policy=admission["limit_policy"],
                    execution_controls={
                        "monetary_cap_usd": None,
                        "model_calls": None,
                        "cumulative_tokens": None,
                        "generation_seconds": None,
                        "network_timeout_seconds": None,
                    },
                )
            if "owner_pricing" in p:
                transport.settings.update(
                    owner_pricing_coverage_accepted=True,
                    owner_pricing=p["owner_pricing"],
                    accounting="owner-assumed envelope estimate; not guaranteed invoice",
                    pricing_basis={
                        "policy": p["pricing_policy"],
                        "owner_record_sha256": digest(p["owner_pricing"]),
                        "profile_sha256": digest(p),
                        "cost_is_estimate_under_owner_assumption": True,
                        "actual_invoice_guaranteed": False,
                    },
                )
            if "tariff_evidence" in p:
                transport.settings["pricing_basis"] = {
                    "policy": p["pricing_policy"],
                    "tariff_evidence": p["tariff_evidence"],
                    "profile_sha256": digest(p),
                    "actual_invoice_guaranteed": False,
                }
            if "accounting_policy" in p:
                transport.settings.update(
                    accounting="estimated-usage-not-invoice",
                    billing_assurance=(
                        (
                            "no monetary ceiling; "
                            if "limit_policy" in admission
                            else "USD 1000 shared estimated usage exposure ceiling; "
                        )
                        + "usage completeness, hidden billing and additional fees "
                        "may differ from invoice"
                    ),
                    accounting_policy=p["accounting_policy"],
                    provider_guarantees=p["guarantees"],
                )
                transport.settings["pricing_basis"].update(
                    accounting_policy=p["accounting_policy"],
                    ceiling_basis="estimated-usage-not-invoice",
                    source_sha=p.get("owner_pricing", {}).get(
                        "source_sha", admission.get("source_sha")
                    ),
                    provider_guarantees=p["guarantees"],
                    additional_fees="verified-absent"
                    if p["guarantees"]["no_other_fees"]
                    else "unknown-not-included",
                )
            return transport
        except BaseException:
            if inner is not None:
                inner.close()
            raise

    return make
