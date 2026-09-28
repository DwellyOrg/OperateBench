"""Closed external admission contract. No prices, approvals or secrets supplied."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

from tools.three_flow_budget import OPTIONAL_PROFILE, validate_limit_policy
from tools.three_flow_campaign import (
    CONFIGURATIONS,
    LATEST_ROSTER_PROFILE,
    assignments,
    selected_roster,
)
from tools.three_flow_profiles import ENDPOINTS
from tools.three_flow_profiles import minimum_effort as minimum_effort
from tools.three_flow_profiles import registry as registry
from tools.three_flow_runtime import ROOT, source_binding

VARIABLES = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "river": "RIVER_API_KEY",
}
RESOURCE_POLICY = {
    "river_active": 2,
    "http_active": 4,
    "native_construction": "serialized",
}
GUARANTEES = (
    "account_access_verified",
    "provider_bounds_verified",
    "all_billable_input_in_usage",
    "hidden_reasoning_in_output_usage_and_limit",
    "all_input_cache_write_read_tiers_covered",
    "all_output_tiers_covered",
    "no_other_fees",
)


def digest(value: Any) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def runtime_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "executable": str(Path(sys.executable).absolute()),
        "executable_sha256": hashlib.sha256(
            Path(sys.executable).read_bytes()
        ).hexdigest(),
        "dependencies": sorted(
            (d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
        ),
        "lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "runtime_digest": source_binding(),
    }


def proposal(sha: str, roster_profile: str = "legacy-v1") -> dict[str, Any]:
    """Pure description, never an approval template with approved defaults."""
    return {
        "schema": "three-flow-preflight-v1",
        "source_sha": sha,
        "runtime": runtime_identity(),
        "registry": registry(
            roster_profile=roster_profile,
            mode_profile="off-or-minimum-v1"
            if roster_profile != "legacy-v1"
            else "legacy-v1",
        ),
        "configurations": list(CONFIGURATIONS),
        "slots": assignments("series-id-required", roster_profile),
        "resource_policy": RESOURCE_POLICY,
        "cap_usd": "1000",
        "admitted": False,
        "credentials_accessed": False,
        "blockers": [
            "external owner admission absent",
            "external verified or owner-assumed tariff coverage required",
            "external verified bounds or explicit published-context diagnostic required",
            "strict usage guarantees or explicit owner reported-usage consent required",
            "parent review, suite, "
            f"{len(selected_roster(roster_profile)) * len(CONFIGURATIONS)} "
            "qualification and no-intervening-spend required",
        ],
    }


def identifier(value: Any) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", value) is not None
    )


def validate(admission: Any, sha: str, *, synthetic: bool) -> dict[str, Any]:
    return _validate(admission, sha, synthetic=synthetic, historical=False)


def _validate(
    admission: Any, sha: str, *, synthetic: bool, historical: bool
) -> dict[str, Any]:
    fields = {
        "schema",
        "approved",
        "mode",
        "source_sha",
        "runtime",
        "registry",
        "campaign_id",
        "attempt_id",
        "campaign_root",
        "custody_root",
        "credential_file",
        "cap_usd",
        "resource_policy",
        "selected",
        "profiles",
        "assets",
        "operation",
        "original_claim_sha256",
        "journal_sha256",
        "pending",
        "corrections",
        "parent_gates",
        "no_intervening_spend",
    }
    if isinstance(admission, dict) and "roster_profile" in admission:
        fields.add("roster_profile")
        if (
            admission["roster_profile"] != LATEST_ROSTER_PROFILE
            or admission.get("mode_profile") != "off-or-minimum-v1"
        ):
            raise ValueError("explicit latest roster requires off-or-minimum profile")
    if isinstance(admission, dict) and "mode_profile" in admission:
        fields.add("mode_profile")
        if admission["mode_profile"] != "off-or-minimum-v1":
            raise ValueError("unknown reasoning mode profile")
    if isinstance(admission, dict) and "transport_policy" in admission:
        fields.add("transport_policy")
        if admission["transport_policy"] != "paced-safe-errors-v1":
            raise ValueError("unknown transport policy")
    correction = isinstance(admission, dict) and "correction_binding" in admission
    if correction:
        fields.add("correction_binding")
    optional = isinstance(admission, dict) and "limit_policy" in admission
    if optional:
        fields.add("limit_policy")
    if not isinstance(admission, dict) or set(admission) != fields:
        raise ValueError("closed admission schema required")
    a = admission
    if historical and (correction or a["operation"] != "initial"):
        raise ValueError("original admission must be initial")
    if correction:
        validate_correction_binding(a, sha)
    if historical:
        validate_historical_identity(a)
    financial_source = (
        a["correction_binding"]["original_source_sha"] if correction else sha
    )
    if optional:
        validate_limit_policy(a["limit_policy"])
        if a["limit_policy"]["source_sha"] != financial_source:
            raise ValueError("optional limits source binding mismatch")
    if (
        a["schema"] != "three-flow-admission-v1"
        or a["approved"] is not (not synthetic)
        or a["mode"] != ("synthetic" if synthetic else "paid")
        or a["source_sha"] != sha
        or (not historical and digest(a["runtime"]) != digest(runtime_identity()))
        or (
            not historical
            and digest(a["registry"])
            != digest(
                registry(
                    roster_profile=a.get("roster_profile", "legacy-v1"),
                    mode_profile=a.get("mode_profile", "legacy-v1"),
                    transport_policy=a.get("transport_policy", "legacy-v1"),
                )
            )
        )
        or (a["cap_usd"] is not None if optional else a["cap_usd"] != "1000")
        or digest(a["resource_policy"]) != digest(RESOURCE_POLICY)
        or not all(identifier(a[k]) for k in ("campaign_id", "attempt_id"))
        or a["parent_gates"] is not True
        or a["no_intervening_spend"] is not True
    ):
        raise ValueError("admission source/runtime/scope/gates mismatch")
    roster = assignments(a["campaign_id"], a.get("roster_profile", "legacy-v1"))
    ids = [s["trial_id"] for s in roster]
    selected = a["selected"]
    if (
        not isinstance(selected, list)
        or not selected
        or any(i not in ids for i in selected)
        or selected != [i for i in ids if i in selected]
    ):
        raise ValueError("ordered unique initial subset required")
    slots = [s for s in roster if s["trial_id"] in selected]
    models = {s["model"] for s in slots}
    if not isinstance(a["profiles"], dict) or set(a["profiles"]) != models:
        raise ValueError("exact selected model profiles required")
    for slot in slots:
        p = a["profiles"][slot["model"]]
        validate_profile(p, slot["provider"])
        if a.get("roster_profile") == LATEST_ROSTER_PROFILE:
            validate_latest_http_profile(p, slot["model"])
        if optional != (p.get("execution_policy") == OPTIONAL_PROFILE):
            raise ValueError("execution controls differ from monetary policy")
        if "owner_pricing" in p:
            owner = p["owner_pricing"]
            pricing_registry = (
                a["correction_binding"]["original_admission"]["registry"]
                if correction
                else a["registry"]
            )
            entry = next(r for r in pricing_registry if r["model"] == slot["model"])
            if owner["source_sha"] != financial_source or owner[
                "registry_entry_sha256"
            ] != digest(entry):
                raise ValueError("owner pricing source/settings binding mismatch")
    operation = a["operation"]
    if operation not in ("initial", "resume", "correction"):
        raise ValueError("unknown series operation")
    if operation == "initial":
        if (
            a["original_claim_sha256"] is not None
            or a["journal_sha256"] is not None
            or a["pending"] != selected
            or a["corrections"] != {}
        ):
            raise ValueError("initial series cannot borrow historical authority")
    elif not all(
        isinstance(a[k], str) and re.fullmatch(r"[a-f0-9]{64}", a[k])
        for k in ("original_claim_sha256", "journal_sha256")
    ):
        raise ValueError("continuation must bind original series and journal")
    if not isinstance(a["pending"], list) or a["pending"] != [
        i for i in selected if i in a["pending"]
    ]:
        raise ValueError("explicit ordered pending subset required")
    if operation != "correction" and a["corrections"] != {}:
        raise ValueError("corrections require distinct operation")
    if operation == "correction":
        if a["pending"] or set(a["corrections"]) != set(selected):
            raise ValueError("exact correction subset required")
        for v in a["corrections"].values():
            if (
                not isinstance(v, dict)
                or set(v) != {"reason", "evidence_sha256"}
                or v["reason"] not in ("runtime_fix", "evaluator_fix", "provider_fix")
                or not re.fullmatch(r"[a-f0-9]{64}", v["evidence_sha256"])
            ):
                raise ValueError("confirmed technical fix evidence required")
    if not historical:
        verify_assets(a["assets"], any(s["provider"] == "river" for s in slots))
    elif any(s["provider"] == "river" for s in slots):
        assets = a["assets"]
        if (
            not isinstance(assets, dict)
            or set(assets) != {"root", "catalog_sha256"}
            or not isinstance(assets["root"], str)
            or not Path(assets["root"]).is_absolute()
            or not isinstance(assets["catalog_sha256"], str)
            or not re.fullmatch(r"[a-f0-9]{64}", assets["catalog_sha256"])
        ):
            raise ValueError("original native asset binding required")
    elif a["assets"] is not None:
        raise ValueError("unselected original native assets")
    return a


def validate_historical_identity(a: dict[str, Any]) -> None:
    """Validate recorded identity shape, never today's runtime/settings or assets.

    Historical request settings are opaque JSON objects: old legacy-v1 mappings
    and future off/minimum settings need not equal the current execution registry.
    This is semantic owner-custody validation, not a signature or source rebuild.
    """
    for key in ("campaign_root", "custody_root", "credential_file"):
        if not isinstance(a[key], str) or not Path(a[key]).is_absolute():
            raise ValueError("original absolute path required")
    runtime = a["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {
        "python",
        "executable",
        "executable_sha256",
        "dependencies",
        "lock_sha256",
        "runtime_digest",
    }:
        raise ValueError("closed original runtime identity required")
    if (
        any(
            not isinstance(runtime[k], str) or not runtime[k]
            for k in ("python", "executable")
        )
        or not Path(runtime["executable"]).is_absolute()
    ):
        raise ValueError("original runtime identity strings required")
    for k in ("executable_sha256", "lock_sha256", "runtime_digest"):
        if not isinstance(runtime[k], str) or not re.fullmatch(
            r"[a-f0-9]{64}", runtime[k]
        ):
            raise ValueError("original runtime digest required")
    if not isinstance(runtime["dependencies"], list) or any(
        not isinstance(row, (list, tuple))
        or len(row) != 2
        or any(not isinstance(v, str) or not v for v in row)
        for row in runtime["dependencies"]
    ):
        raise ValueError("original dependency identity required")
    rows = a["registry"]
    roster = selected_roster(a.get("roster_profile", "legacy-v1"))
    if not isinstance(rows, list) or len(rows) != len(roster):
        raise ValueError("original registry roster required")
    for row, (provider, model) in zip(rows, roster, strict=True):
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "provider",
                "model",
                "endpoint",
                "mapping",
                "request_settings",
                "settings_source",
                "retries",
                "output_limit",
                "admitted",
            }
            or row["provider"] != provider
            or row["model"] != model
            or any(
                not isinstance(row[k], str) or not row[k]
                for k in ("endpoint", "mapping", "settings_source", "output_limit")
            )
            or not isinstance(row["request_settings"], dict)
            or type(row["retries"]) is not int
            or row["retries"] != 0
            or row["admitted"] is not False
        ):
            raise ValueError("closed original registry identity required")


def validate_initial_receipt(value: Any, prior: dict[str, Any]) -> None:
    """Exact emitted initial receipt, including JSON types (false is not zero)."""
    expected = {
        k: prior[k]
        for k in (
            "campaign_id",
            "campaign_root",
            "custody_root",
            "attempt_id",
            "selected",
            "mode",
            "source_sha",
            "operation",
            "original_claim_sha256",
            "journal_sha256",
        )
    }
    expected.update(
        authorizing=False, admission_sha256=digest(prior), cap_usd=prior["cap_usd"]
    )
    if "limit_policy" in prior:
        expected["limit_policy"] = prior["limit_policy"]
    if not isinstance(value, dict) or digest(value) != digest(expected):
        raise ValueError("closed consumed initial receipt required")


def validate_correction_binding(a: dict[str, Any], sha: str) -> None:
    """Execution consent does not repin or widen the original financial authority.

    The embedded original admission is authenticated against its consumed receipt
    in consume(), not against today's runtime. Legacy receipts remain unchanged.
    Profiles are deliberately byte-equivalent: price changes need separate consent
    and are not supported by this narrow source-correction extension.
    """
    b = a["correction_binding"]
    if not isinstance(b, dict) or set(b) != {
        "schema",
        "owner_accepted",
        "owner_evidence_sha256",
        "original_source_sha",
        "execution_source_sha",
        "original_admission",
        "financial_policy_sha256",
        "genesis_sha256",
        "manifest_sha256",
        "original_trial_artifacts_sha256",
    }:
        raise ValueError("closed correction binding required")
    if (
        b["schema"] != "three-flow-correction-binding-v1"
        or b["owner_accepted"] is not True
        or a["operation"] != "correction"
        or b["execution_source_sha"] != sha
    ):
        raise ValueError("explicit correction execution consent required")
    for key, size in (
        ("original_source_sha", 40),
        ("execution_source_sha", 40),
        ("owner_evidence_sha256", 64),
        ("financial_policy_sha256", 64),
        ("genesis_sha256", 64),
        ("manifest_sha256", 64),
    ):
        if not isinstance(b[key], str) or not re.fullmatch(
            rf"[a-f0-9]{{{size}}}", b[key]
        ):
            raise ValueError("correction binding digest required")
    artifacts = b["original_trial_artifacts_sha256"]
    if (
        not isinstance(artifacts, dict)
        or not isinstance(a["selected"], list)
        or set(artifacts) != set(a["selected"])
    ):
        raise ValueError("exact original correction artifacts required")
    for row in artifacts.values():
        if (
            not isinstance(row, dict)
            or set(row) != {"assigned", "binding", "closure"}
            or any(
                not isinstance(v, str) or not re.fullmatch(r"[a-f0-9]{64}", v)
                for v in row.values()
            )
        ):
            raise ValueError("original trial artifact digest required")
    prior = b["original_admission"]
    _validate(
        prior,
        b["original_source_sha"],
        synthetic=a["mode"] == "synthetic",
        historical=True,
    )
    if (
        not isinstance(prior, dict)
        or prior.get("operation") != "initial"
        or "correction_binding" in prior
        or prior.get("source_sha") != b["original_source_sha"]
        or any(
            prior.get(k) != a[k]
            for k in ("campaign_id", "campaign_root", "custody_root", "mode", "cap_usd")
        )
        or prior.get("roster_profile", "legacy-v1")
        != a.get("roster_profile", "legacy-v1")
        or prior.get("limit_policy") != a.get("limit_policy")
        or b["financial_policy_sha256"]
        != digest({"cap_usd": a["cap_usd"], "limit_policy": a.get("limit_policy")})
        or any(prior.get("profiles", {}).get(k) != v for k, v in a["profiles"].items())
    ):
        raise ValueError("original financial authority changed")


REPORTED_USAGE_POLICY = "reported-usage-diagnostic-v1"
REPORTED_USAGE_GUARANTEES = (
    "account_access_verified",
    "all_billable_input_in_usage",
    "hidden_reasoning_in_output_usage_and_limit",
    "no_other_fees",
)
OWNER_PRICING_POLICY = "owner-assumed-token-envelope-v1"
PRICE_GUARANTEES = (
    "all_input_cache_write_read_tiers_covered",
    "all_output_tiers_covered",
)
PRICE_CATEGORIES = ("input", "cache_read", "cache_write", "output", "reasoning")


def validate_owner_pricing(p: dict[str, Any]) -> None:
    """Conditional tariff coverage only; no new provider facts or usage semantics.

    External review must inventory *all* applicable verified tier rates, including
    cache premiums, and identify the remaining unknowns. Hashes bind that review;
    they are not signatures or proof that the supplied evidence is true.
    """
    o = p["owner_pricing"]
    if not isinstance(o, dict) or set(o) != {
        "owner_accepted",
        "currency",
        "unit_tokens",
        "unknown_usd_per_million",
        "owner_evidence_sha256",
        "source_sha",
        "registry_entry_sha256",
        "profile_sha256",
        "verified_rate_evidence_sha256",
        "rates",
    }:
        raise ValueError("explicit closed owner pricing record required")
    if (
        o["owner_accepted"] is not True
        or o["currency"] != "USD"
        or type(o["unit_tokens"]) is not int
        or o["unit_tokens"] != 1000000
        or o["unknown_usd_per_million"] != "10"
        or o["profile_sha256"]
        != digest(
            {
                k: v
                for k, v in p.items()
                if k not in ("owner_pricing", "rate_evidence_sha256")
            }
        )
    ):
        raise ValueError("owner consent/units/profile binding mismatch")
    if p["rate_evidence_sha256"] != digest(o):
        raise ValueError("owner rate record changed")
    for field in (
        "owner_evidence_sha256",
        "registry_entry_sha256",
        "verified_rate_evidence_sha256",
    ):
        if not isinstance(o[field], str) or not re.fullmatch(r"[a-f0-9]{64}", o[field]):
            raise ValueError("owner evidence/settings binding required")
    rates = o["rates"]
    if not isinstance(rates, dict) or set(rates) != set(PRICE_CATEGORIES):
        raise ValueError("explicit input/output/cache/reasoning rate inventory required")
    maxima = {}
    for category in PRICE_CATEGORIES:
        r = rates[category]
        if (
            not isinstance(r, dict)
            or set(r) != {"verified_usd_per_million", "assume_unknown"}
            or type(r["assume_unknown"]) is not bool
            or not isinstance(r["verified_usd_per_million"], list)
        ):
            raise ValueError("explicit verified/unknown category provenance required")
        values = []
        for raw in r["verified_usd_per_million"]:
            if not isinstance(raw, str):
                raise ValueError("explicit decimal verified rates required")
            value = Decimal(raw)
            if not value.is_finite() or value < 0:
                raise ValueError("invalid verified rate")
            values.append(value)
        if r["assume_unknown"]:
            values.append(Decimal("10"))
        if not values:
            raise ValueError("uncovered category")
        maxima[category] = max(values)
    for field, categories, guarantee in (
        ("input_usd_per_million", PRICE_CATEGORIES[:3], PRICE_GUARANTEES[0]),
        ("output_usd_per_million", PRICE_CATEGORIES[3:], PRICE_GUARANTEES[1]),
    ):
        if Decimal(p[field]) != max(maxima[c] for c in categories):
            raise ValueError("envelope must retain maximum verified/assumed rate")
        if any(rates[c]["assume_unknown"] for c in categories):
            if p["guarantees"][guarantee] is not False:
                raise ValueError("owner assumption is not provider verification")
        elif p["guarantees"][guarantee] is not True:
            raise ValueError("verified category coverage required")


# Conservative published tier maxima, USD / million tokens. These are floors,
# NOT account access, owner consent, or invoice guarantees. Existing higher
# owner-reviewed rates remain higher. Only latest-http-v1 enforces these facts.
# OpenAI model pages: long (>272K) x2 input/cache, x1.5 output; fast x2.
# Sol/Luna additionally document regional x1.1. Anthropic pricing: 1h cache
# write x2, US x1.1; Opus 5.5 fast x2 (Fable has no fast tier here).
LATEST_HTTP_ENVELOPES = {
    "gpt-6-luna": (1050000, "0.55", "1.65"),
    "gpt-6-sol": (1050000, "11", "33"),
    "gpt-6-astra": (1050000, "50", "150"),
    "claude-fable-5-1": (1000000, "22", "55"),
    "claude-opus-5-5": (1000000, "17.6", "44"),
}


def validate_latest_http_profile(p: dict[str, Any], model: str) -> None:
    if model not in LATEST_HTTP_ENVELOPES:
        return
    context, input_rate, output_rate = LATEST_HTTP_ENVELOPES[model]
    if p["context_max"] != context or p["output_max"] != 128000:
        raise ValueError("latest HTTP published bounds mismatch")
    if Decimal(p["input_usd_per_million"]) < Decimal(input_rate) or Decimal(
        p["output_usd_per_million"]
    ) < Decimal(output_rate):
        raise ValueError("latest HTTP envelope loses known published tier rates")


def validate_profile(p: Any, provider: str) -> None:
    fields = {
        "endpoint",
        "input_max",
        "output_max",
        "context_max",
        "input_usd_per_million",
        "output_usd_per_million",
        "network_timeout_seconds",
        "rate_evidence_sha256",
        "bounds_evidence_sha256",
        "guarantees",
        "pricing_policy",
    }
    if isinstance(p, dict) and "owner_output_cap_tokens" in p:
        fields.add("owner_output_cap_tokens")
        if (
            provider != "river"
            or p.get("bounds_policy") is None
            or type(p["owner_output_cap_tokens"]) is not int
            or p["owner_output_cap_tokens"] <= 0
        ):
            raise ValueError("positive River diagnostic owner output cap required")
    optional = isinstance(p, dict) and "execution_policy" in p
    if optional:
        fields.add("execution_policy")
        if p["execution_policy"] != OPTIONAL_PROFILE:
            raise ValueError("unknown execution controls")
    owner = isinstance(p, dict) and p.get("pricing_policy") == OWNER_PRICING_POLICY
    if owner:
        fields.add("owner_pricing")
    reported = isinstance(p, dict) and "accounting_policy" in p
    if reported:
        fields.add("accounting_policy")
    if not isinstance(p, dict) or set(p) not in (fields, fields | {"bounds_policy"}):
        raise ValueError("complete provider profile required")
    if reported:
        # This separate consent is covered by the existing owner profile digest,
        # source and registry binding. Price consent alone never accepts this risk.
        policy = p["accounting_policy"]
        if (
            not owner
            or not isinstance(policy, dict)
            or set(policy) != {"mode", "owner_accepted", "owner_evidence_sha256"}
            or policy["mode"] != REPORTED_USAGE_POLICY
            or policy["owner_accepted"] is not True
            or not isinstance(policy["owner_evidence_sha256"], str)
            or not re.fullmatch(r"[a-f0-9]{64}", policy["owner_evidence_sha256"])
        ):
            raise ValueError("explicit owner reported-usage risk consent required")
    diagnostic = p.get("bounds_policy") is not None
    if diagnostic:
        b = p["bounds_policy"]
        if (
            not isinstance(b, dict)
            or set(b) != {"mode", "source", "scope", "interpretation"}
            or b["mode"] != "published-context-diagnostic-v1"
            or not isinstance(b["source"], str)
            or not b["source"].startswith("https://")
            or b["scope"] not in ("endpoint", "upstream-model-not-endpoint")
            or b["interpretation"]
            not in ("published-integer", "conservative-decimal-shorthand")
        ):
            raise ValueError("explicit published context diagnostic evidence required")
    if p["endpoint"] != ENDPOINTS.get(provider, "api.river.ai:443"):
        raise ValueError("fixed endpoint mismatch")
    expected_guarantees = dict.fromkeys(GUARANTEES, True)
    if diagnostic:
        expected_guarantees["provider_bounds_verified"] = False
    if owner:
        g = p["guarantees"]
        if (
            not isinstance(g, dict)
            or set(g) != set(GUARANTEES)
            or any(
                (
                    type(g[k]) is not bool
                    if reported and k in REPORTED_USAGE_GUARANTEES
                    else g[k] is not expected_guarantees[k]
                )
                for k in GUARANTEES
                if k not in PRICE_GUARANTEES
            )
        ):
            raise ValueError("owner prices do not establish provider bounds or usage")
        validate_owner_pricing(p)
    elif (
        digest(p["guarantees"]) != digest(expected_guarantees)
        or p["pricing_policy"] != "conservative-all-tier-envelope-v1"
    ):
        raise ValueError("unknown rate or token coverage")
    for field in ("rate_evidence_sha256", "bounds_evidence_sha256"):
        if not isinstance(p[field], str) or not re.fullmatch(r"[a-f0-9]{64}", p[field]):
            raise ValueError("provider-origin evidence binding required")
    for field in ("input_max", "output_max", "context_max"):
        if diagnostic and field != "context_max" and p[field] is None:
            continue
        if type(p[field]) is not int or p[field] <= 0:
            raise ValueError(
                "positive admitted context and known provider bounds required"
            )
    if any(
        p[f] is not None and p[f] > p["context_max"] for f in ("input_max", "output_max")
    ):
        raise ValueError("inconsistent context bounds")
    if provider in ("openai", "anthropic") and p["output_max"] != 128000:
        raise ValueError("fixed flagship profile mismatch")
    for field in ("input_usd_per_million", "output_usd_per_million"):
        if not isinstance(p[field], str):
            raise ValueError("explicit decimal prices required")
        value = Decimal(p[field])
        if not value.is_finite() or value <= 0:
            raise ValueError("missing or nonfinite prices")
    t = p["network_timeout_seconds"]
    if optional:
        if t is not None or not diagnostic:
            raise ValueError(
                "optional execution requires null timeout and dynamic context bound"
            )
    elif type(t) not in (float, int) or not math.isfinite(t) or t <= 0:
        raise ValueError("explicit reviewed network timeout required")


def verify_assets(value: Any, required: bool) -> None:
    if not required:
        if value is not None:
            raise ValueError("unselected native assets")
        return
    from tools.three_flow_river import sdk
    from tools.three_flow_river_assets import catalog

    sdk()  # actual generated interface import, no channel or metadata request
    if not isinstance(value, dict) or set(value) != {"root", "catalog_sha256"}:
        raise ValueError("explicit native asset binding required")
    root = Path(value["root"])
    if not root.is_absolute() or root.resolve(strict=True) != root:
        raise ValueError("canonical native assets required")
    if value["catalog_sha256"] != digest(catalog()):
        raise ValueError("native catalog changed")
    for row in catalog()["models"]:
        for name, meta in row["assets"].items():
            path = root / row.get("tokenizer_model", row["model"]) / name
            if (
                path.resolve(strict=True) != path
                or hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]
            ):
                raise ValueError("native asset changed")
