"""Sequential Maintenance V1 case-study campaign (source checkout only).

Artifact 8, ledger 3 and the strict grader are unchanged. This adapter owns only
assignment identity, launch admission and descriptive reporting. No resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import httpx

from operatebench.agents.evidence import (
    EvidenceRecordingModelAgent,
    EvidenceRunIdentity,
    ProviderEvidenceRecorder,
    WireCaptureTransport,
    provider_identity_for,
)
from operatebench.agents.pricing import LifecyclePricingPolicy
from operatebench.artifact import read_artifact, replay_artifact, write_artifact
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.execution_bundle import audit_execution_bundle_files
from operatebench.execution_ledger import read_execution_ledger
from operatebench.providers.cost import CostCapExceededError
from operatebench.runner import run_episode
from tools.aggregate_budget import Budget, SharedGuard, canonical, decimal, money
from tools.diagnose_aggregate_budget import PLACEHOLDER, client_for, module
from tools.track_scorecard import partial_prefix

SOURCE = Path(__file__).resolve().parents[1]
EXAMPLE = SOURCE / "examples/campaign-offline.json"
FIXTURE = SOURCE / "examples/operatebench/maintenance_v0_1.yaml"
SCHEMA = "operatebench.campaign.v2"
CONFIG_SCHEMA = "operatebench.campaign.v1"
PROFILE_KEYS = ("haiku45", "luna56")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path: Path, value: Any) -> None:
    """Exclusive, private, durable publication; no overwrite/resume."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        raw = canonical(value)
        view = memoryview(raw)
        while view:
            n = os.write(fd, view)
            if n <= 0:
                raise OSError("short write")
            view = view[n:]
        os.fsync(fd)
    finally:
        os.close(fd)
    dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def source_identity() -> dict[str, Any]:
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", *args], cwd=SOURCE)

    names = sorted(
        set(
            git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split(
                b"\0"
            )
        )
        - {b""}
    )
    files = {
        os.fsdecode(n): file_digest(SOURCE / os.fsdecode(n))
        for n in names
        if (SOURCE / os.fsdecode(n)).is_file()
    }
    return {
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "clean": not bool(git("status", "--porcelain")),
        "files_sha256": digest(files),
    }


def profiles() -> list[dict[str, Any]]:
    """Pin only reviewed existing profiles, without invoking canary execution."""
    result = []
    for key in PROFILE_KEYS:
        m = module(key)
        result.append(
            {
                "key": key,
                "model": m.CANARY_MODEL,
                "profile": m.CANARY_PROFILE_ID,
                "calls": m.MAX_PROVIDER_CALLS,
                "tokens": m.TOKEN_HARD_CAP,
                "output": m.MAX_OUTPUT_TOKENS,
                "request_seconds": m.TURN_DEADLINE_SECONDS,
                "wall_seconds": m.WALL_CLOCK_DEADLINE_SECONDS,
                "money": str(m.FIXED_CANARY_COST_CAP_USD),
            }
        )
    return result


def make_plan(config: dict[str, Any], root: Path) -> dict[str, Any]:
    if set(config) != {
        "schema",
        "campaign_id",
        "mode",
        "repeats",
        "aggregate_usd",
        "profiles",
        "rate_provenance",
    }:
        raise ValueError("exact campaign config fields required")
    if config["schema"] != CONFIG_SCHEMA or config["mode"] not in ("offline", "live"):
        raise ValueError("unsupported campaign schema/mode")
    if not isinstance(config["campaign_id"], str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9-]{0,63}", config["campaign_id"]
    ):
        raise ValueError("invalid campaign ID")
    if type(config["repeats"]) is not int or not 1 <= config["repeats"] <= 100:
        raise ValueError("explicit bounded repeat count required")
    if money(config["aggregate_usd"]) <= 0:
        raise ValueError("positive aggregate ceiling required")
    if not isinstance(config["rate_provenance"], str) or not config["rate_provenance"]:
        raise ValueError("rate provenance required")
    declared = config["profiles"]
    pinned = profiles()
    if not isinstance(declared, list) or len(declared) != 2:
        raise ValueError("exactly the two reviewed profiles required")
    for given, fixed in zip(declared, pinned, strict=False):
        if set(given) != set(fixed) | {"input_usd_per_mtok", "output_usd_per_mtok"}:
            raise ValueError("exact profile fields required")
        if any(type(given[k]) is not type(v) or given[k] != v for k, v in fixed.items()):
            raise ValueError("reviewed profile/bounds mismatch")
        for field in ("input_usd_per_mtok", "output_usd_per_mtok"):
            if money(given[field]) <= 0:
                raise ValueError("positive explicit rates required")
    config = json.loads(canonical(config))  # freeze caller-owned mutable input
    plan = {
        "schema": SCHEMA,
        "config": config,
        "source": source_identity(),
        "spec": {
            "path": str(FIXTURE.relative_to(SOURCE)),
            "file_sha256": file_digest(FIXTURE),
            "digest": load_spec(FIXTURE).spec_digest_sha256,
            "scenario": "V1",
        },
        "output": str(root),
        "trials": [],
    }
    for repeat in range(config["repeats"]):
        for p in declared:
            ident = f"{config['campaign_id']}-r{repeat + 1:03d}-{p['key']}"
            plan["trials"].append(
                {"trial_id": ident, "profile_key": p["key"], "repeat": repeat + 1}
            )
    return plan


def exposure(rows: list[dict[str, Any]]) -> Fraction:
    return sum(
        (
            Fraction(r["actual"])
            if r["state"] == "settled"
            else Fraction(r["bound"])
            if r["state"] in ("reserved", "sent", "unknown")
            else Fraction(0)
            for r in rows
        ),
        Fraction(0),
    )


class CampaignBudget(Budget):
    """Same durable reducer, with additive per-trial admission constraints.

    Prospective states are checked before the base class appends anything. Both
    money accounts therefore retain the SAME wire-adjusted liability, not two
    charges. Token policy remains conservative cumulative admitted token bounds.
    """

    def __init__(self, *args: Any, limits: dict[str, dict[str, Any]], **kwargs: Any):
        self.limits = limits
        super().__init__(*args, **kwargs)

    def _reduce(self, old: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
        rows = super()._reduce(old, event)
        for cell in {r["cell"] for r in rows.values()}:
            limit = self.limits[cell]
            selected = [r for r in rows.values() if r["cell"] == cell]
            if exposure(selected) > money(limit["money"]):
                raise CostCapExceededError("per-trial money ceiling before HTTP")
            if (
                sum(r["ib"] + r["ob"] for r in selected if r["state"] != "cancelled")
                > limit["tokens"]
            ):
                raise CostCapExceededError("per-trial wire token ceiling before HTTP")
        return rows


class DispatchTransport(httpx.BaseTransport):
    def __init__(
        self, inner: httpx.BaseTransport, budget: Budget, trial: str, deadline: float
    ):
        self.inner, self.budget, self.trial, self.deadline = (
            inner,
            budget,
            trial,
            deadline,
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise CostCapExceededError("per-trial wall deadline before HTTP")
        # SDK/executor can shorten this further; never allow its defaults to
        # extend the remaining whole-trial deadline.
        timeouts = request.extensions.get("timeout", {})
        request.extensions["timeout"] = {
            k: min(float(v), remaining) if v is not None else remaining
            for k, v in timeouts.items()
        }
        self.budget.dispatch(self.trial, json.loads(request.content))
        return self.inner.handle_request(request)

    def close(self) -> None:
        self.inner.close()


class CampaignAgent(EvidenceRecordingModelAgent):
    def __init__(self, *args: Any, deadline: float, **kwargs: Any):
        self.deadline = deadline
        super().__init__(*args, **kwargs)

    def decide(self, observation: Any) -> Any:
        if time.monotonic() >= self.deadline:
            raise TimeoutError("campaign trial deadline")
        return super().decide(observation)


def offline_environment() -> None:
    # Names only: never inspect credential values. Refuse provider routing too.
    prefixes = ("OPENAI_", "ANTHROPIC_", "MISTRAL_", "XAI_", "AWS_", "GOOGLE_")
    if any(
        k.startswith(prefixes) or k.upper() in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
        for k in os.environ
    ):
        raise ValueError("ambient provider credentials/routing forbidden")


def network_tripwire() -> None:
    def deny(event: str, args: tuple[Any, ...]) -> None:
        if event == "socket.connect" and args[0].family in (
            socket.AF_INET,
            socket.AF_INET6,
        ):
            raise RuntimeError("offline internet access forbidden")

    sys.addaudithook(deny)


def failure_result(exc: BaseException) -> dict[str, Any]:
    from operatebench.agents.transport import FAULTS

    fault = getattr(exc, "fault", None)
    reason = fault if type(fault) is str and fault in FAULTS else "infrastructure"
    if isinstance(exc, KeyboardInterrupt):
        reason = "user_interrupt"
    elif isinstance(exc, TimeoutError):
        reason = "provider_deadline"
    elif isinstance(exc, CostCapExceededError):
        reason = "provider_budget"
    # No exception text, arbitrary class name or unreviewed fault string persists.
    return {"status": "non-scored", "reason": reason, "exception_type": None}


def execute_trial(
    root: Path,
    trial: dict[str, Any],
    p: dict[str, Any],
    budget: CampaignBudget,
    *,
    key: str,
    offline: bool,
    handler: Any = None,
) -> dict[str, Any]:
    from operatebench.agents.anthropic_messages import AnthropicMessagesTransport
    from operatebench.agents.openai_responses import OpenAIResponsesTransport

    directory = root / trial["trial_id"]
    directory.mkdir(mode=0o700)
    m = module(p["key"])
    spec = load_spec(FIXTURE)
    policy = LifecyclePricingPolicy(
        policy_id="operator_pinned_v1",
        input_usd_per_mtok=Decimal(p["input_usd_per_mtok"]),
        output_usd_per_mtok=Decimal(p["output_usd_per_mtok"]),
        rate_source=m.FIXED_CANARY_RATE_SOURCE,
    )
    controls = m.controls_for(policy, cap=Decimal(p["money"]))
    guard = SharedGuard(
        budget=budget,
        cell=trial["trial_id"],
        policy=policy,
        max_output_tokens=p["output"],
        token_hard_cap=p["tokens"],
    )
    # SharedGuard owns admission; initialize the inherited reporting fields with
    # this trial's cap rather than the aggregate cap (no new accounting).
    from operatebench.agents.pricing import LifecycleCostGuard

    LifecycleCostGuard.__init__(
        guard,
        policy=policy,
        cap_usd=Decimal(p["money"]),
        max_output_tokens=p["output"],
        token_hard_cap=p["tokens"],
    )
    deadline = time.monotonic() + p["wall_seconds"]
    wire = WireCaptureTransport()
    native = (
        httpx.MockTransport(handler or m.ScriptedProvider(max_calls=p["calls"]))
        if offline
        else httpx.HTTPTransport(retries=0, trust_env=False)
    )
    wire.attach(DispatchTransport(native, budget, trial["trial_id"], deadline))
    client = None
    recorder = None
    result: dict[str, Any] = {
        "status": "non-scored",
        "reason": "infrastructure",
        "exception_type": None,
    }
    try:
        client = client_for(p["key"], key, wire)
        cls = (
            AnthropicMessagesTransport
            if p["key"] == "haiku45"
            else OpenAIResponsesTransport
        )
        transport = cls(
            model=p["model"],
            client=client,
            deadline_seconds=p["request_seconds"],
            cost_guard=guard,
        )
        recorder = ProviderEvidenceRecorder(
            path=directory / "execution_ledger.ndjson",
            identity=EvidenceRunIdentity(
                operation_id=spec.operation_id,
                spec_digest_sha256=spec.spec_digest_sha256,
                scenario_id="V1",
                agent_id=trial["trial_id"],
            ),
            provider=provider_identity_for(transport),
            controls=controls,
            wire=wire,
            guard=guard,
        )
        transport.attach_recorder(recorder)
        agent = CampaignAgent(
            transport,
            model=p["model"],
            agent_id=trial["trial_id"],
            recorder=recorder,
            max_transport_calls=p["calls"],
            max_output_tokens=p["output"],
            deadline=deadline,
        )
        run = run_episode(
            spec,
            "V1",
            trial["trial_id"],
            agent_factory=lambda: agent,
            agent_kind="model",
            evidence_recorder=recorder,
        )
        # Unknown cost stops the campaign even if the business engine scored.
        unknown = any(
            r["state"] in ("unknown", "sent", "reserved") for r in budget.records.values()
        )
        artifact_path = write_artifact(run, directory / "episode_artifact.json")
        bundle = audit_execution_bundle_files(
            artifact_path, directory / "execution_ledger.ndjson"
        )
        replay = replay_artifact(spec, read_artifact(artifact_path))
        if not bundle.ok or not replay.ok:
            raise ValueError("audit/replay failure")
        result = {
            "status": "scored",
            "reason": "unknown_usage" if unknown else "evaluated",
            "exception_type": None,
        }
    except (Exception, KeyboardInterrupt) as exc:
        result = failure_result(exc)
    finally:
        try:
            if recorder is not None:
                recorder.close()
        finally:
            if client is not None:
                client.close()
            else:
                wire.close()
    result["evidence_hashes"] = {
        str(f.relative_to(root)): file_digest(f)
        for f in sorted(directory.iterdir())
        if f.is_file()
    }
    return result


def append_status(
    root: Path, plan_hash: str, trial: str, status: str, **extra: Any
) -> None:
    # One immutable event per transition, sequenced by filename. O_EXCL/fsync
    # prevents overwriting evidence; reconstruction validates the state machine.
    directory = root / "status"
    sequence = len(list(directory.iterdir()))
    previous = file_digest(directory / f"{sequence - 1:06d}.json") if sequence else None
    event = dict(
        plan_sha256=plan_hash,
        previous_sha256=previous,
        trial_id=trial,
        status=status,
        **extra,
    )
    publish(directory / f"{sequence:06d}.json", dict(event, event_sha256=digest(event)))


def consume_authority(
    authority: Path, plan: dict[str, Any], *, mock: bool
) -> dict[str, str]:
    """New one-shot campaign binding; never creates an authorizing record.

    Custody and consumption follow the existing operator's descriptor boundary,
    but its closed scope/episode100 authorization is deliberately NOT reused.
    """
    import stat

    from tools.run_episode100 import private_fd, read_json

    if not plan["source"]["clean"] or source_identity() != plan["source"]:
        raise ValueError("fresh exact clean source required before credentials")
    directory = authority.parent
    if (
        directory != directory.resolve(strict=True)
        or directory.stat().st_uid != os.getuid()
        or stat.S_IMODE(directory.stat().st_mode) != 0o700
    ):
        raise ValueError("private authority directory required")
    fd = private_fd(authority)
    try:
        approval = read_json(fd)
    finally:
        os.close(fd)
    expected = {
        "schema": "operatebench.campaign-authority.v1",
        "plan_sha256": digest(plan),
        "mode": "dummy" if mock else "paid",
        "approved": True,
        "historical_authorities_retired": True,
    }
    if (
        type(approval) is not dict
        or set(approval) != set(expected) | {"credential_file"}
        or any(
            type(approval[k]) is not type(v) or approval[k] != v
            for k, v in expected.items()
        )
    ):
        raise ValueError("authority does not bind exact campaign plan")
    # No credential open (let alone content read) on source/config/plan mismatch.
    credential = private_fd(approval["credential_file"])
    try:
        receipt = {
            "schema": SCHEMA,
            "plan_sha256": digest(plan),
            "authorizing": False,
            "mode": expected["mode"],
            "output": plan["output"],
        }
        publish(directory / "CAMPAIGN-CONSUMED", receipt)
        publish(Path(plan["output"]) / "authority-receipt.json", receipt)
        # Both markers and directory fsyncs are durable before this first read.
        keys = read_json(credential)
        if (
            type(keys) is not dict
            or set(keys) != set(PROFILE_KEYS)
            or any(type(v) is not str or not v.strip() for v in keys.values())
        ):
            raise ValueError("invalid credential mapping; authority stays consumed")
        if mock and any(v != PLACEHOLDER for v in keys.values()):
            keys.clear()
            raise ValueError("dummy authority accepts only public placeholders")
        return keys
    finally:
        os.close(credential)


def run_campaign(
    config: dict[str, Any],
    root: Path,
    *,
    handlers: dict[str, Any] | None = None,
    authority: Path | None = None,
    mock_authority: bool = False,
) -> dict[str, Any]:
    offline_environment()
    offline = config.get("mode") == "offline"
    if not offline and (authority is None or mock_authority or handlers is not None):
        raise ValueError(
            "live requires fresh external campaign authority and no injected handlers"
        )
    if mock_authority and (not offline or authority is None):
        raise ValueError("dummy authority requires offline mode and explicit authority")
    if offline and authority is not None and not mock_authority:
        raise ValueError("offline cannot consume paid authority")
    if offline:
        network_tripwire()
    root = root.absolute()
    if root.parent != root.parent.resolve(strict=True):
        raise ValueError("canonical private output required")
    plan = make_plan(config, root)
    root.mkdir(mode=0o700)
    publish(root / "plan.json", {"plan": plan, "sha256": digest(plan)})
    (root / "status").mkdir(mode=0o700)
    from tools.aggregate_budget import consume_offline_once

    if offline:
        consume_offline_once(root)
    pmap = {p["key"]: p for p in config["profiles"]}
    limits = {t["trial_id"]: pmap[t["profile_key"]] for t in plan["trials"]}
    namespace = (
        ("offline-test-" if offline else "campaign-paid-")
        + "campaign-"
        + config["campaign_id"]
    )
    budget = CampaignBudget(
        root,
        cap=Decimal(config["aggregate_usd"]),
        namespace=namespace,
        create=True,
        limits=limits,
    )
    keys: dict[str, str] = {}
    try:
        keys = (
            consume_authority(authority, plan, mock=mock_authority)
            if authority is not None
            else dict.fromkeys(PROFILE_KEYS, PLACEHOLDER)
        )
        for trial in plan["trials"]:
            profile = pmap[trial["profile_key"]]
            # Even a zero-byte input requires this output reservation. This is
            # only an early impossibility check, never a substitute for actual
            # serialized-wire admission and never a full-trial estimate.
            minimum = money(profile["output_usd_per_mtok"]) * profile["output"] / 1000000
            if budget.cap - budget.exposure < minimum:
                break
            append_status(root, digest(plan), trial["trial_id"], "started")
            try:
                result = execute_trial(
                    root,
                    trial,
                    pmap[trial["profile_key"]],
                    budget,
                    key=keys[trial["profile_key"]],
                    offline=offline,
                    handler=handlers.get(trial["trial_id"]) if handlers else None,
                )
            except (Exception, KeyboardInterrupt) as exc:
                # Setup/close failures are also terminal campaign stops. An
                # unreadable/torn ledger still makes report reconstruction fail
                # closed; never replace it with invented scored evidence.
                result = failure_result(exc)
                directory = root / trial["trial_id"]
                result["evidence_hashes"] = (
                    {
                        str(f.relative_to(root)): file_digest(f)
                        for f in sorted(directory.iterdir())
                        if f.is_file()
                    }
                    if directory.exists()
                    else {}
                )
            result["accounting_checkpoint"] = {
                "seq": budget.seq,
                "sha256": budget.previous,
            }
            append_status(root, digest(plan), trial["trial_id"], **result)
            if result["status"] != "scored" or result["reason"] != "evaluated":
                break
    finally:
        keys.clear()
        budget.close()
    report = rebuild_report(root)
    publish(root / "report.json", report)
    # Same exclusive file writer, plain human-readable Markdown bytes.
    text = render_report(report)
    fd = os.open(root / "report.md", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    return report


def rebuild_report(root: Path) -> dict[str, Any]:
    """Read-only projection. No invented terminal for an unfinished start."""
    envelope = json.loads((root / "plan.json").read_text())
    plan, plan_hash = envelope["plan"], envelope["sha256"]
    if digest(plan) != plan_hash or plan["output"] != str(root.absolute()):
        raise ValueError("plan identity/hash mismatch")
    if plan.get("schema") != SCHEMA:
        raise ValueError(
            "unsupported campaign evidence schema: accounting binding required"
        )
    if source_identity()["files_sha256"] != plan["source"]["files_sha256"]:
        raise ValueError("report requires matching source files")
    if file_digest(FIXTURE) != plan["spec"]["file_sha256"]:
        raise ValueError("report requires matching spec")
    pmap = {p["key"]: p for p in plan["config"]["profiles"]}
    limits = {t["trial_id"]: pmap[t["profile_key"]] for t in plan["trials"]}
    states = {
        t["trial_id"]: {"status": "not_started", "reason": "not_started"}
        for t in plan["trials"]
    }
    events = sorted((root / "status").iterdir())
    for i, path in enumerate(events):
        if path.name != f"{i:06d}.json":
            raise ValueError("status sequence gap")
        e = json.loads(path.read_text())
        claimed = e.pop("event_sha256")
        if digest(e) != claimed or e.pop("previous_sha256") != (
            file_digest(events[i - 1]) if i else None
        ):
            raise ValueError("status hash chain mismatch")
        if e["plan_sha256"] != plan_hash or e["trial_id"] not in states:
            raise ValueError("foreign status identity")
        before = states[e["trial_id"]]["status"]
        if not (
            (before == "not_started" and e["status"] == "started")
            or (before == "started" and e["status"] in ("scored", "non-scored"))
        ):
            raise ValueError("illegal repeated status transition")
        states[e["trial_id"]] = e
    budget = CampaignBudget(
        root,
        cap=Decimal(plan["config"]["aggregate_usd"]),
        namespace=(
            "offline-test-" if plan["config"]["mode"] == "offline" else "campaign-paid-"
        )
        + "campaign-"
        + plan["config"]["campaign_id"],
        limits=limits,
    )
    try:
        # Read the locked, reducer-validated journal descriptor, not another path.
        assert budget.fd is not None
        lines = os.pread(budget.fd, os.fstat(budget.fd).st_size, 0).splitlines(
            keepends=True
        )
        for state in states.values():
            if state["status"] not in ("scored", "non-scored"):
                continue
            checkpoint = state.get("accounting_checkpoint", {})
            if not isinstance(checkpoint, dict):
                raise ValueError("invalid accounting checkpoint")
            seq = checkpoint.get("seq")
            if (
                type(seq) is not int
                or not 1 <= seq <= len(lines)
                or hashlib.sha256(lines[seq - 1]).hexdigest() != checkpoint.get("sha256")
            ):
                raise ValueError("terminal accounting checkpoint mismatch/truncation")
        totals = {k: str(decimal(v)) for k, v in budget.totals().items()}
        trials = []
        for assigned in plan["trials"]:
            t = dict(
                assigned,
                **{
                    k: v
                    for k, v in states[assigned["trial_id"]].items()
                    if k not in ("trial_id", "plan_sha256")
                },
            )
            t.setdefault("reason", "incomplete_start_unknown")
            t.update(
                reliable=None,
                terminal=None,
                dimensions=None,
                counter_status="unknown",
                requests=None,
                actions=None,
                reads=None,
                retries=None,
                bundle_ok=None,
                replay_ok=None,
                evidence={},
            )
            directory = root / t["trial_id"]
            ledger_path = directory / "execution_ledger.ndjson"
            artifact_path = directory / "episode_artifact.json"
            p = pmap[t["profile_key"]]
            if ledger_path.exists():
                if t["status"] == "not_started":
                    raise ValueError("ledger without durable start")
                ledger = read_execution_ledger(
                    ledger_path, require_complete=t["status"] == "scored"
                )
                h = ledger.header
                if (
                    h.agent_id,
                    h.scenario_id,
                    h.spec_digest_sha256,
                    h.provider.model,
                    h.provider.settings["request_profile"],
                ) != (
                    t["trial_id"],
                    "V1",
                    plan["spec"]["digest"],
                    p["model"],
                    p["profile"],
                ):
                    raise ValueError("trial/ledger identity mismatch")
                t.update(
                    execution_run_id=h.execution_run_id,
                    operation_instance_id=h.operation_instance_id,
                    ledger_status=ledger.status,
                    counter_status=(
                        "complete"
                        if ledger.complete and not ledger.truncated_tail
                        else "lower_bound"
                    ),
                    requests=ledger.totals.attempts,
                    actions=sum(
                        c.decision is not None and c.decision.kind == "ACT"
                        for c in ledger.calls
                    ),
                    reads=sum(
                        c.decision is not None and c.decision.kind == "RETRIEVE"
                        for c in ledger.calls
                    ),
                    retries=sum(max(0, len(c.attempts) - 1) for c in ledger.calls),
                )
                if t["status"] == "scored":
                    if ledger.status != "scored":
                        raise ValueError("non-scored ledger cannot become a score")
                    artifact = read_artifact(artifact_path)
                    bundle = audit_execution_bundle_files(artifact_path, ledger_path)
                    replay = replay_artifact(load_spec(FIXTURE), artifact)
                    if not bundle.ok or not replay.ok:
                        raise ValueError("scored artifact failed bundle/replay")
                    t.update(
                        reliable=artifact["evaluation"]["reliable"],
                        terminal=artifact["status"],
                        dimensions=artifact["evaluation"],
                        bundle_ok=True,
                        replay_ok=True,
                    )
                partial = directory / "execution_ledger.partial.ndjson"
                if partial.exists():
                    t["partial"] = partial_prefix(
                        partial,
                        {
                            "operation_instance_id": h.operation_instance_id,
                            "agent_id": t["trial_id"],
                            "scenario_id": "V1",
                            "spec_digest_sha256": plan["spec"]["digest"],
                        },
                    )
            elif t["status"] == "scored":
                raise ValueError("scored status without ledger")
            if directory.exists():
                t["evidence"] = {
                    str(f.relative_to(root)): file_digest(f)
                    for f in sorted(directory.iterdir())
                    if f.is_file()
                }
            if (
                t.get("evidence_hashes") is not None
                and t.pop("evidence_hashes") != t["evidence"]
            ):
                raise ValueError("terminal evidence hash mismatch")
            rows = [r for r in budget.records.values() if r["cell"] == t["trial_id"]]
            t["measured_usd"] = str(
                decimal(
                    sum(
                        (Fraction(r["actual"]) for r in rows if r["state"] == "settled"),
                        Fraction(0),
                    )
                )
            )
            t["unknown_exposure_usd"] = str(
                decimal(
                    sum(
                        (Fraction(r["bound"]) for r in rows if r["state"] == "unknown"),
                        Fraction(0),
                    )
                )
            )
            t["pending_exposure_usd"] = str(
                decimal(
                    sum(
                        (
                            Fraction(r["bound"])
                            for r in rows
                            if r["state"] in ("reserved", "sent")
                        ),
                        Fraction(0),
                    )
                )
            )
            t["budget_exposure_usd"] = str(decimal(exposure(rows)))
            t["unknown_cost"] = any(
                r["state"] in ("reserved", "sent", "unknown") for r in rows
            )
            t["financial_status"] = "bound"
            if t["status"] == "started":
                # No terminal checkpoint: even a valid accounting prefix cannot
                # prove absence of missing dispatch/settlement after a crash.
                # Retain observed liabilities, never publish a zero-cost claim.
                t["financial_status"] = "unknown_incomplete_start"
                t["observed_measured_usd"] = t["measured_usd"]
                t["observed_budget_exposure_usd"] = t["budget_exposure_usd"]
                t["measured_usd"] = t["budget_exposure_usd"] = None
                t["unknown_cost"] = True
            trials.append(t)
    finally:
        budget.close()

    def counts(items: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "assigned": len(items),
            "started": sum(t["status"] != "not_started" for t in items),
            "scored": sum(t["status"] == "scored" for t in items),
            "reliable": sum(t["reliable"] is True for t in items),
        }

    def fractions(c: dict[str, int]) -> dict[str, Any]:
        return {
            k: {
                "numerator": c["reliable"],
                "denominator": c[k],
                "fraction": f"{c['reliable']}/{c[k]}" if c[k] else None,
            }
            for k in ("assigned", "started", "scored")
        }

    def trial_totals(items: list[dict[str, Any]]) -> dict[str, Any]:
        result = {
            k: sum(t[k] or 0 for t in items)
            for k in ("requests", "actions", "reads", "retries")
        }
        result["incomplete_counter_trials"] = sum(
            t["status"] != "not_started" and t["counter_status"] != "complete"
            for t in items
        )
        result["counter_status"] = (
            "lower_bound" if result["incomplete_counter_trials"] else "complete"
        )
        for k in (
            "measured_usd",
            "unknown_exposure_usd",
            "pending_exposure_usd",
            "budget_exposure_usd",
        ):
            result[k] = (
                None
                if any(t[k] is None for t in items)
                else str(decimal(sum((money(t[k]) for t in items), Fraction(0))))
            )
        return result

    financial_complete = all(t["status"] != "started" for t in trials)
    overall = counts(trials)
    return {
        "schema": SCHEMA,
        "plan_sha256": plan_hash,
        "source": plan["source"],
        "cost_label": (
            "SIMULATED usage and prices; not actual invoices"
            if plan["config"]["mode"] == "offline"
            else (
                "Provider-reported usage priced at pinned rates; "
                "not invoice reconciliation"
            )
        ),
        "scope": (
            "Maintenance V1 descriptive case study; "
            "no model ranking or statistical inference"
        ),
        "input_sha256": {
            str(path.relative_to(root)): file_digest(path)
            for path in [root / "plan.json", root / "successor-events.jsonl", *events]
        },
        "units": {
            "requests": "HTTP attempts recorded by ledger (including failures)",
            "actions": "ACT decisions, not accepted actions",
            "reads": "RETRIEVE decisions/batches, not individual documents",
            "retries": "attempts beyond first within each model call",
        },
        "counts": overall,
        "fractions": fractions(overall),
        "financial_complete": financial_complete,
        "observed_accounting": totals,
        "accounting": totals if financial_complete else dict.fromkeys(totals),
        "totals": trial_totals(trials),
        "profiles": {
            key: {
                "counts": counts([t for t in trials if t["profile_key"] == key]),
                "totals": trial_totals([t for t in trials if t["profile_key"] == key]),
                "fractions": fractions(
                    counts([t for t in trials if t["profile_key"] == key])
                ),
            }
            for key in PROFILE_KEYS
        },
        "trials": trials,
    }


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Campaign report",
        "",
        report["scope"],
        report["cost_label"],
        "",
        "Assigned/started/scored/reliable: " + json.dumps(report["counts"]),
        "Success fractions: " + json.dumps(report["fractions"]),
        "Financially complete: " + str(report["financial_complete"]),
        "Observed accounting prefix (retained liabilities, not proof of completeness): "
        + json.dumps(report["observed_accounting"]),
        "Accounting (null when financial completeness is unknown): "
        + json.dumps(report["accounting"]),
        "Units: " + json.dumps(report["units"]),
        "All-attempt totals: " + json.dumps(report["totals"]),
        "Per-profile raw counts/fractions/totals: " + json.dumps(report["profiles"]),
        "",
    ]
    for t in report["trials"]:
        lines.extend([f"## {t['trial_id']}", json.dumps(t, sort_keys=True, indent=2), ""])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="new offline campaign; no resume")
    run.add_argument("--config", type=Path, default=EXAMPLE)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--authority",
        type=Path,
        help="fresh external exact-plan authority, never created here",
    )
    run.add_argument(
        "--mock-authority",
        action="store_true",
        help="offline dummy-only consuming-path rehearsal",
    )
    proposal = sub.add_parser(
        "proposal", help="inert plan to stdout, NOT authorization; no credentials"
    )
    proposal.add_argument("--config", type=Path, required=True)
    proposal.add_argument("--output", type=Path, required=True)
    report = sub.add_parser(
        "report", help="read-only verified report regeneration to stdout"
    )
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "proposal":
            plan = make_plan(json.loads(args.config.read_text()), args.output.absolute())
            print(
                json.dumps(
                    {"plan": plan, "sha256": digest(plan), "authorizing": False}, indent=2
                )
            )
            return 0
        if args.command == "report":
            result = rebuild_report(args.output.absolute())
            print(json.dumps(result, indent=2) if args.json else render_report(result))
        else:
            result = run_campaign(
                json.loads(args.config.read_text()),
                args.output,
                authority=args.authority,
                mock_authority=args.mock_authority,
            )
            print(render_report(result))
        return 0 if result["counts"]["scored"] == result["counts"]["assigned"] else 1
    except Exception:
        # Never stringify provider/credential-bearing exceptions.
        print(
            "campaign refused or evidence incomplete; retained evidence requires audit",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
