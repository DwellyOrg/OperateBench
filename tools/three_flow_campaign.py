"""Offline executable campaign controller and non-authorizing owner preflight.

No paid CLI mode, credential lookup, approval writer or historical launcher.
The 616 assigned slots remain distinct from mock mechanics and excluded slots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from operatebench._write_once import _write_once_bytes
from operatebench.agents.pricing import LifecyclePricingPolicy
from operatebench.agents.transport import ProviderFailure, content_digest
from operatebench.sdk.profile_packs.pack import COMMERCE_COMMANDS, COMPLIANCE_COMMANDS
from tools.aggregate_budget import SharedGuard, canonical, decimal
from tools.three_flow_budget import CampaignBudget
from tools.three_flow_http import LATEST_HTTP_MODELS, HTTPCampaignTransport
from tools.three_flow_mock import SDKMockTransport
from tools.three_flow_profiles import (
    EXCLUDED_FUTURE_MODEL as EXCLUDED_FUTURE_MODEL,
)
from tools.three_flow_profiles import (
    LATEST_ROSTER_PROFILE as LATEST_ROSTER_PROFILE,
)
from tools.three_flow_profiles import MODELS as MODELS
from tools.three_flow_profiles import (
    RIVER_MODELS as RIVER_MODELS,
)
from tools.three_flow_profiles import (
    ROSTER as ROSTER,
)
from tools.three_flow_profiles import (
    selected_roster as selected_roster,
)
from tools.three_flow_runtime import (
    ROOT,
    Trial,
    provider_identity,
    replay_trial,
    run_trial,
    source_binding,
)

CONFIGURATIONS = (
    tuple(("maintenance", s) for s in ("V1", "V2", "V3"))
    + tuple(("commerce", s) for s in COMMERCE_COMMANDS.reference_scenarios)
    + tuple(("compliance", s) for s in COMPLIANCE_COMMANDS.reference_scenarios)
)


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def preflight(roster_profile: str = "legacy-v1") -> dict[str, Any]:
    """Descriptive report only; cannot create or consume owner authority."""
    return {
        "admitted": False,
        "source_digest": source_binding(),
        "source_commit": source_commit(),
        "global_cap_usd": "1000",
        "workers": 4,
        "river_workers": 2,
        "native_construction": "serialized",
        "configurations": len(CONFIGURATIONS),
        "initial_assignments": len(selected_roster(roster_profile)) * len(CONFIGURATIONS),
        "providers": [
            {
                "provider": p,
                "model": m,
                "admitted": False,
                "blockers": ["owner source/settings/account/price approval absent"]
                + (
                    ["current official price unknown"]
                    if m == "deepseek-ai/DeepSeek-V4.1-Flash"
                    else []
                )
                + (
                    ["River endpoint/output/reasoning/cache billing admission absent"]
                    if p == "river"
                    else []
                ),
            }
            for p, m in selected_roster(roster_profile)
        ],
        "paid_entrypoint": False,
        "authority_created": False,
    }


def assignments(
    campaign_id: str, roster_profile: str = "legacy-v1"
) -> list[dict[str, Any]]:
    return [
        {
            "campaign_id": campaign_id,
            "trial_id": f"initial-{i:03d}-{j:02d}",
            "provider": p,
            "model": m,
            "flow": f,
            "scenario": s,
            "repetition": 1,
            "rerun_of": None,
        }
        for i, (p, m) in enumerate(selected_roster(roster_profile))
        for j, (f, s) in enumerate(CONFIGURATIONS)
    ]


def write_private(path: Path, value: bytes) -> None:
    _write_once_bytes(
        value, path, error_adapter=lambda failure: ValueError(failure.reason)
    )


def milestone_receipts(record: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Domain event names, accepted causal receipts; never infer from model prose."""
    milestones: dict[str, list[dict[str, Any]]] = {}
    for event in record["episode"]["events"]:
        if event["disposition"] == "accepted" and event["caused_by"] is not None:
            milestones.setdefault(event["event_type"], []).append(
                {k: event[k] for k in ("event_id", "sequence", "at", "caused_by")}
            )
    return milestones


def score_row(slot: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    turns = record["provider_turns"]
    return {
        **slot,
        "classification": "scored",
        "evidence_origin": record["evidence_origin"],
        "eligible_for_live_results": record["evidence_origin"] == "PROVIDER_CANDIDATE",
        # Maintenance has scenarios but no profile-pack profile: JSON null,
        # CSV empty; Commerce/Compliance retain their literal bound profile ID.
        "profile": None
        if slot["flow"] == "maintenance"
        else record["binding"]["profile"]["profile_id"],
        "completion": record["evaluation"].get(
            "legitimate_completion",
            record["evaluation"]["dimensions"].get(
                "completion", record["evaluation"]["dimensions"].get("terminal")
            ),
        )
        if isinstance(record["evaluation"]["dimensions"], dict)
        else record["evaluation"]["legitimate_completion"],
        "terminal_status": record["episode"]["status"],
        "evaluation": record["evaluation"],
        "milestones": milestone_receipts(record),
        "requests": len(turns),
        "attempts": len(turns),
        "network_rpcs": sum(t.get("network_rpcs", 1) for t in turns),
        "input_tokens": sum(t["input_tokens"] for t in turns),
        "output_tokens": sum(t["output_tokens"] for t in turns),
        "latency_seconds": sum(t["latency_seconds"] for t in turns),
        "estimated_usd": str(
            sum((Decimal(t["measured_usd"]) for t in turns), Decimal(0))
        ),
        "unknown_liability_usd": "0",
        "record_digest": record["record_digest"],
        **(
            {"pricing_basis": record["provider"]["settings"]["pricing_basis"]}
            if "pricing_basis" in record["provider"]["settings"]
            else {}
        ),
    }


def reports(
    root: Path,
    rows: list[dict[str, Any]],
    totals: dict[str, Any],
    *,
    grouped: bool = True,
    limit_policy: dict[str, Any] | None = None,
) -> None:
    origins = {r.get("evidence_origin") for r in rows}
    scope = (
        "MIXED_MOCK_AND_PROVIDER_CANDIDATE"
        if {"SDK_MOCK_NOT_LLM", "PROVIDER_CANDIDATE"} <= origins
        else "PROVIDER_CANDIDATE"
        if "PROVIDER_CANDIDATE" in origins
        else "OFFLINE_SDK_MOCK_NOT_LLM_RESULTS"
    )
    write_private(
        root / "scorecard.json",
        canonical(
            {
                "scope": scope,
                "rows": rows,
                **({"limit_policy": limit_policy} if limit_policy is not None else {}),
                "global_budget": {
                    k: None if v is None else str(decimal(v)) for k, v in totals.items()
                },
            }
        ),
    )
    fields = list(dict.fromkeys(k for r in rows for k in r))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                for k, v in row.items()
            }
        )
    write_private(root / "scorecard.csv", stream.getvalue().encode())
    lines = [
        "# Three-flow candidate scorecards (see per-row evidence origin)",
        "",
        f"Scope: {scope}",
        "",
        "| Trial | Model | Flow/scenario | Classification |",
        "|---|---|---|---|",
    ]
    if any("accounting_policy" in r.get("pricing_basis", {}) for r in rows):
        lines[2:2] = [
            "**estimated-usage-not-invoice: "
            + (
                "no monetary ceiling, "
                if limit_policy is not None
                else "USD 1000 shared usage-estimate ceiling, "
            )
            + "not an actual-invoice guarantee. See per-row owner policy provenance; "
            "unknown additional fees are not zero.**",
            "",
        ]
    lines.extend(
        f"| {r['trial_id']} | {r['model']} | {r['flow']}/{r['scenario']} "
        f"| {r['classification']} |"
        for r in rows
    )
    for row in rows:
        details = {
            k: row.get(k)
            for k in (
                "evidence_origin",
                "eligible_for_live_results",
                "pricing_basis",
                "profile",
                "fault",
                "local_failure",
                "evidence_reason",
                "completion",
                "terminal_status",
                "evaluation",
                "milestones",
                "requests",
                "attempts",
                "input_tokens",
                "output_tokens",
                "latency_seconds",
                "estimated_usd",
                "known_cost_usd",
                "unknown_liability_usd",
            )
        }
        lines.extend(
            [
                "",
                "## " + row["trial_id"],
                "",
                "```json",
                json.dumps(details, indent=2, sort_keys=True),
                "```",
            ]
        )
    write_private(root / "scorecard.md", ("\n".join(lines) + "\n").encode())
    if grouped:
        for index, (provider, model) in enumerate(
            ROSTER + tuple(pair for pair in LATEST_HTTP_MODELS if pair not in ROSTER)
        ):
            for flow in ("maintenance", "commerce", "compliance"):
                subset = [r for r in rows if r["model"] == model and r["flow"] == flow]
                if subset:
                    directory = root / f"model-{index:02d}-{provider}-{flow}"
                    directory.mkdir(mode=0o700)
                    reports(
                        directory,
                        subset,
                        totals,
                        grouped=False,
                        limit_policy=limit_policy,
                    )


def financial_row(row: dict[str, Any], budget: CampaignBudget) -> dict[str, Any]:
    accounting = budget.cell_accounting(row["trial_id"])
    known = sum(
        (
            Fraction(r["actual"])
            for r in accounting["records"].values()
            if r["state"] == "settled"
        ),
        Fraction(0),
    )
    return {
        **row,
        "known_cost_usd": str(decimal(known)),
        "raw_rpc_attempts": sum(
            r.get("raw_rpcs", 0) for r in accounting["records"].values()
        ),
        "unknown_liability_usd": accounting["full_unknown_usd"],
    }


def reconcile_closure(
    root: Path, slot: dict[str, Any], row: dict[str, Any], budget: CampaignBudget
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate retained evidence without SDK construction or historical rewrites.

    Only the executing frozen source can replay here. Other source versions
    require a separately provided historical replay environment; none is
    implicitly selected or rebound by this offline controller.
    """
    status, reason = "not_scored", None
    original = dict(row)
    if row["classification"] == "scored":
        ident = slot["trial_id"]
        try:
            trial = Trial(**json.loads((root / (ident + ".binding.json")).read_text()))
            record = json.loads((root / ident / "record.json").read_text())
            if any(
                getattr(trial, k) != slot[k]
                for k in ("campaign_id", "trial_id", "flow", "scenario", "model")
            ):
                raise ValueError("retained trial differs from slot")
            if (
                record["trial"] != asdict(trial)
                or record["record_digest"] != row["record_digest"]
                or record["record_digest"]
                != content_digest(
                    {k: v for k, v in record.items() if k != "record_digest"}
                )
            ):
                raise ValueError("retained record/binding/closure digest mismatch")
            source_key = (
                "source_digest"
                if trial.flow == "maintenance"
                else "campaign_runtime_digest"
            )
            if record["binding"][source_key] != source_binding():
                status, reason = "unavailable", "frozen_source_unavailable"
            else:
                replay_trial(trial, record, expected_record_digest=row["record_digest"])
                row = score_row(slot, record)
                status = "verified"
        except FileNotFoundError:
            status, reason = "unavailable", "retained_evidence_missing"
        except OSError:
            status, reason = "unavailable", "retained_evidence_unreadable"
        except Exception:
            status, reason = "invalid", "retained_evidence_invalid"
        if reason is not None:
            row = {
                **slot,
                "classification": "aborted",
                "fault": "evidence_reconciliation",
                "evidence_reason": reason,
                "evaluation": None,
                "completion": None,
                "estimated_usd": None,
                "eligible_for_live_results": False,
                "record_digest": original.get("record_digest"),
            }
    row = financial_row(row, budget)
    return row, {
        "trial_id": slot["trial_id"],
        "original_closure_digest": content_digest(original),
        "evidence_status": status,
        "evidence_reason": reason,
        "original": original,
        "reconciled": row,
    }


def correction_slot(
    original: dict[str, Any],
    tag: str,
    reason: str,
    evidence_sha256: str | None = None,
) -> dict[str, Any]:
    """Canonical correction identity; original admission remains separate."""
    from tools.three_flow_admission import identifier

    if not identifier(tag) or reason not in (
        "provider_fix",
        "runtime_fix",
        "evaluator_fix",
    ):
        raise ValueError("invalid correction identity or reason")
    if evidence_sha256 is not None and (
        not isinstance(evidence_sha256, str)
        or re.fullmatch(r"[a-f0-9]{64}", evidence_sha256) is None
    ):
        raise ValueError("invalid correction evidence")
    return {
        **original,
        "trial_id": original["trial_id"] + "-fix-" + tag,
        "rerun_of": original["trial_id"],
        "correction_reason": reason,
        **(
            {"fix_evidence_sha256": evidence_sha256}
            if evidence_sha256 is not None
            else {}
        ),
    }


def campaign_manifest(
    campaign_id: str,
    selected: list[dict[str, Any]],
    mode: str,
    workers: int = 4,
    *,
    limit_policy: dict[str, Any] | None = None,
    roster_profile: str = "legacy-v1",
) -> dict[str, Any]:
    return {
        "source_digest": source_binding(),
        "source_commit": source_commit(),
        **({"limit_policy": limit_policy} if limit_policy is not None else {}),
        "all_initial_slots": assignments(campaign_id, roster_profile),
        **({"roster_profile": roster_profile} if roster_profile != "legacy-v1" else {}),
        "selected": selected,
        "mode": mode,
        "workers": workers,
        "river_workers": min(2, workers),
        "native_construction": "serialized",
    }


def run_campaign(
    root: Path,
    *,
    campaign_id: str,
    providers: tuple[str, ...] = ("openai",),
    workers: int = 4,
    slots: list[dict[str, Any]] | None = None,
    resume: bool = False,
    corrections: dict[str, str] | None = None,
    river_assets: Path | None = None,
    transport_factory: Any = None,
    campaign_budget: CampaignBudget | None = None,
    mode: str = "SDK_MOCK_NOT_LLM",
    attempt_id: str | None = None,
    selected_corrections: list[dict[str, Any]] | None = None,
    assignment_prepared: bool = False,
    roster_profile: str = "legacy-v1",
) -> list[dict[str, Any]]:
    if corrections is not None and (not resume or slots is not None or not corrections):
        raise ValueError("corrections require resume and nonempty original slot mapping")
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("one to four worker threads required")
    roster = assignments(campaign_id, roster_profile)
    selected = (
        [s for s in roster if s["provider"] in providers] if slots is None else slots
    )
    if not selected or any(s not in roster for s in selected):
        raise ValueError("unknown assigned slot")
    if len({s["trial_id"] for s in selected}) != len(selected):
        raise ValueError("duplicate assigned trial slot")
    if selected_corrections is not None:
        if (
            not resume
            or corrections is not None
            or not selected_corrections
            or attempt_id is None
        ):
            raise ValueError("selected corrections require one explicit continuation")
        originals = {s["trial_id"]: s for s in selected}
        seen = set()
        for slot in selected_corrections:
            original = originals.get(slot.get("rerun_of"))
            if original is None or original["trial_id"] in seen:
                raise ValueError("unknown or duplicate correction original")
            expected = correction_slot(
                original,
                attempt_id,
                slot.get("correction_reason", ""),
                slot.get("fix_evidence_sha256"),
            )
            if "fix_evidence_sha256" not in slot or canonical(slot) != canonical(
                expected
            ):
                raise ValueError("correction differs from assigned original or identity")
            seen.add(original["trial_id"])
        selected = selected_corrections
    if (
        transport_factory is None
        and any(s["provider"] == "river" for s in selected)
        and river_assets is None
    ):
        raise ValueError("River offline route requires explicit pinned assets")
    if not resume and campaign_budget is None:
        root.mkdir(mode=0o700)
    budget = campaign_budget or CampaignBudget(
        root, campaign_id=campaign_id, cap=Decimal("1000"), create=not resume
    )
    try:
        if corrections is not None:
            selected = []
            for original_id, reason in corrections.items():
                original_slot = next(
                    (r for r in roster if r["trial_id"] == original_id), None
                )
                if original_slot is None or reason not in (
                    "provider_fix",
                    "runtime_fix",
                    "evaluator_fix",
                ):
                    raise ValueError("unknown original or correction reason")
                closure = json.loads((root / (original_id + ".closure.json")).read_text())
                if closure["classification"] not in ("excluded", "aborted"):
                    raise ValueError(
                        "scored model behavior cannot be rerun by this controller"
                    )
                tag = content_digest({"source": source_binding(), "reason": reason})[:16]
                selected.append(correction_slot(original_slot, tag, reason))
        manifest = campaign_manifest(
            campaign_id,
            selected,
            mode,
            workers,
            limit_policy=budget.limit_policy,
            roster_profile=roster_profile,
        )
        if resume:
            original = json.loads((root / "assignment.json").read_text())
            if original["all_initial_slots"] != roster or original["mode"] != mode:
                raise ValueError("foreign campaign assignments")
            report_root = root / ("resume-" + (attempt_id or str(uuid4())))
            report_root.mkdir(mode=0o700)
            write_private(report_root / "assignment.json", canonical(manifest))
        else:
            report_root = root
            if assignment_prepared:
                if json.loads((root / "assignment.json").read_text()) != manifest:
                    raise ValueError("prepared assignment differs from execution")
            else:
                write_private(root / "assignment.json", canonical(manifest))
    except BaseException:
        budget.close()
        raise

    construction = Lock()

    def worker(slot: dict[str, Any]) -> dict[str, Any]:
        ident = slot["trial_id"]
        closure_path = root / (ident + ".closure.json")
        assigned_path = root / (ident + ".assigned.json")
        if resume and closure_path.exists():
            row: dict[str, Any] = json.loads(closure_path.read_text())
            if any(row.get(k) != v for k, v in slot.items()):
                raise ValueError("closure differs from assigned slot")
            row, reconciliation = reconcile_closure(root, slot, row, budget)
            write_private(
                report_root / (ident + ".reconciliation.json"), canonical(reconciliation)
            )
            return row
        if resume and assigned_path.exists():
            if json.loads(assigned_path.read_text()) != slot:
                raise ValueError("started slot identity differs")
            row = {
                **slot,
                "classification": "aborted",
                "fault": "interrupted_no_resubmit",
                "evaluation": None,
                "completion": None,
                "estimated_usd": None,
                "unknown_liability_usd": budget.cell_accounting(ident)[
                    "full_unknown_usd"
                ],
                "eligible_for_live_results": False,
            }
            row = financial_row(row, budget)
            write_private(closure_path, canonical(row))
            return row
        write_private(assigned_path, canonical(slot))
        transport: Any = None
        try:
            is_river = slot["provider"] == "river"
            maximum = (
                32768 if is_river else 256000 if slot["provider"] == "mistral" else 128000
            )
            if transport_factory is None:
                guard = SharedGuard(
                    budget=budget,
                    cell=ident,
                    policy=LifecyclePricingPolicy(
                        "OFFLINE_SYNTHETIC_NOT_CURRENT_PRICES", Decimal("1"), Decimal("2")
                    ),
                    max_output_tokens=maximum,
                )
            if transport_factory is not None:
                if is_river:
                    with construction:
                        transport = transport_factory(slot, budget)
                else:
                    transport = transport_factory(slot, budget)
                maximum = transport.max_output_tokens
            elif is_river:
                from tools.three_flow_river import GRPC_OPTIONS, RiverCampaignTransport
                from tools.three_flow_river_mock import RiverMockChannel

                if river_assets is None:
                    raise ValueError(
                        "River offline route requires explicit pinned assets"
                    )
                channel = RiverMockChannel(slot["model"], slot["flow"])
                with construction:
                    transport = RiverCampaignTransport(
                        model=slot["model"],
                        assets=river_assets,
                        channel=channel,
                        guard=guard,
                        max_output_tokens=maximum,
                        channel_options=GRPC_OPTIONS,
                    )
                channel.tokenizer = transport.tokenizer
            else:
                mock = SDKMockTransport(
                    slot["provider"], slot["flow"], model=slot["model"]
                )
                transport = HTTPCampaignTransport(
                    provider=slot["provider"],
                    model=slot["model"],
                    mode_profile="off-or-minimum-v1"
                    if roster_profile != "legacy-v1"
                    else "legacy-v1",
                    api_key="offline-not-a-credential",
                    inner=mock,
                    guard=guard,
                    max_output_tokens=maximum,
                )
            trial = Trial(
                campaign_id,
                ident,
                slot["flow"],
                slot["scenario"],
                slot["model"],
                maximum,
                provider_binding=provider_identity(transport),
            )
            write_private(root / (ident + ".binding.json"), canonical(asdict(trial)))
            record = run_trial(trial, transport, output=root / ident)
            row = score_row(slot, record)
        except Exception as exc:
            liability = budget.cell_accounting(ident)["full_unknown_usd"]
            row = {
                **slot,
                "classification": "excluded"
                if isinstance(exc, ProviderFailure)
                else "aborted",
                "fault": exc.fault
                if isinstance(exc, ProviderFailure)
                else "internal_error",
                "evaluation": None,
                "completion": None,
                "estimated_usd": None,
                "unknown_liability_usd": str(liability),
                "eligible_for_live_results": False,
            }
            if transport is not None and transport.local_failure is not None:
                row.update(
                    classification="aborted",
                    fault="internal_error",
                    local_failure=transport.local_failure,
                )
            elif (
                transport is not None
                and (transport.last_turn or {}).get("diagnostic_category")
                == "provider_request_refusal"
            ):
                row.update(
                    diagnostic_category="provider_request_refusal",
                    safe_status=transport.last_turn["safe_status"],
                    automatic_resubmission=False,
                )
        finally:
            if transport is not None:
                transport.close()
        if transport is not None and "pricing_basis" in transport.settings:
            row["pricing_basis"] = transport.settings["pricing_basis"]
        row = financial_row(row, budget)
        write_private(root / (ident + ".closure.json"), canonical(row))
        return row

    try:
        # Separate bounded lanes: queued native trials cannot starve HTTP work.
        with (
            ThreadPoolExecutor(
                max_workers=min(2, workers), thread_name_prefix="river"
            ) as river_pool,
            ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="http"
            ) as http_pool,
        ):
            futures = [
                (river_pool if slot["provider"] == "river" else http_pool).submit(
                    worker, slot
                )
                for slot in selected
            ]
            rows = [future.result() for future in futures]
        reports(
            report_root,
            rows,
            {**budget.totals(), **({"cap": None} if budget.cap is None else {})},
            limit_policy=budget.limit_policy,
        )
        write_private(
            report_root / "terminal.json",
            canonical(
                {
                    "journal_sequence": budget.seq,
                    "journal_tip": budget.previous,
                    "journal_sha256": hashlib.sha256(
                        (root / "campaign-budget.jsonl").read_bytes()
                    ).hexdigest(),
                    "rows_sha256": content_digest(rows),
                    "source_digest": source_binding(),
                    "mode": mode,
                }
            ),
        )
        return rows
    finally:
        budget.close()


def run_offline(root: Path, **kwargs: Any) -> list[dict[str, Any]]:
    """Compatibility facade over the common executor; never supplies live capability."""
    return run_campaign(root, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("preflight", "offline"), nargs="?", default="preflight"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--campaign-id", default="offline-three-flow")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--river-assets", type=Path)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=("openai", "anthropic", "mistral", "river"),
        default=["openai"],
    )
    args = parser.parse_args()
    if args.mode == "preflight":
        print(json.dumps(preflight(), indent=2))
        return
    if args.output is None:
        parser.error("offline requires a fresh --output path")

    # Trusted-code offline guard: no accidental socket dispatch in this process.
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network forbidden in offline campaign")

    socket.socket.connect = forbidden  # type: ignore[method-assign]
    socket.create_connection = forbidden
    os.umask(0o022)
    rows = run_offline(
        args.output,
        campaign_id=args.campaign_id,
        providers=tuple(args.providers),
        resume=args.resume,
        river_assets=args.river_assets,
    )
    print(
        json.dumps(
            {
                "assigned": len(rows),
                "scored": sum(r["classification"] == "scored" for r in rows),
                "paid_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
