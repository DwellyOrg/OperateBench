"""Explicit paid capabilities for the shared executor; no reference policy here."""

from __future__ import annotations

import importlib
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from operatebench.agents.pricing import LifecyclePricingPolicy
from tools.aggregate_budget import SharedGuard
from tools.three_flow_admission import VARIABLES, digest
from tools.three_flow_http import HTTPCampaignTransport


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

    def authorize(self, *, input_tokens_upper_bound: int) -> Decimal:
        self.check_context(input_tokens_upper_bound, self._max_output_tokens)
        return super().authorize(
            input_tokens_upper_bound=(
                input_tokens_upper_bound if self.diagnostic else self.profile["input_max"]
            )
        )


def live_factory(
    admission: dict[str, Any], keys: dict[str, str], *, network: Any = None
) -> Any:
    def make(slot: dict[str, Any], budget: Any) -> Any:
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
                    mode_profile=admission.get("mode_profile", "legacy-v1"),
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
                    mode_profile=admission.get("mode_profile", "legacy-v1"),
                    transport_policy=(
                        admission.get("transport_policy", "legacy-v1")
                        if slot["provider"] == "mistral"
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
                    source_sha=p["owner_pricing"]["source_sha"],
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
