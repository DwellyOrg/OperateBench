"""Command line interface.

Seven commands, all of which exit non-zero when a contract is violated so they
can be wired straight into CI:

    boundarybench validate        <card.yaml>
    boundarybench compile         <card.yaml> [--json] [--output FILE]
    boundarybench check-solvers   <card.yaml> [--json]
    boundarybench verify-manifest <card.yaml> <card.manifest.json> [--json]
    boundarybench validate-suite  <suite.yaml> [--json]
    boundarybench check-suite     <suite.yaml> [--json]
    boundarybench run-suite       <suite.yaml> --adapter fake-scripted
                                  --output-dir DIR [--trials N] [--json]
    boundarybench run-suite       <suite.yaml> --adapter anthropic --model ID
                                  --output-dir DIR [--trials N] [--json]

The first three work on a single Cube. The next three work on shipped artefacts:
``verify-manifest`` proves one card still matches its committed fingerprint,
``validate-suite`` proves a collection is the collection it claims to be, and
``check-suite`` additionally runs every Cube's causal CI gate.

``run-suite`` is the only one that executes anything. It runs every variant
through the real environment and the real deterministic evaluator, and records
what happened.

Two adapters, and the difference matters — so much so that they produce runs on
two different *tracks*, with different statuses, different scopes and different
run ids. ``fake-scripted`` executes the released in-process fake: no network, no
credential, and — by construction — no model result at all; its runs are stamped
:data:`~boundarybench.runmanifest.RUN_STATUS_SYNTHETIC_FAKE`. ``anthropic`` calls
the Anthropic Messages API under the same scaffold, evaluator, manifest and
ledger, and needs both an explicit ``--model`` and a credential in
``ANTHROPIC_API_KEY``; its runs are stamped
:data:`~boundarybench.runmanifest.RUN_STATUS_PROVIDER_EXECUTION` and say so in
every report. Nothing here defaults the model: a benchmark run has to name what
it measured. The credential is read from the environment only, and never from
``--setting``.

Neither track produces a score, a ranking or a benchmark result at this n, and
starting either one writes a manifest that records a *plan*: see
:data:`~boundarybench.runmanifest.RUN_PLAN_NOTE`, which every report carries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from boundarybench.adapter import (
    AdapterError,
    PolicyFollowingFakeAdapter,
    fake_identity,
)
from boundarybench.budget import (
    BudgetError,
    CostControls,
    RunCostGuard,
    parse_cost_cap,
    usd_text,
)
from boundarybench.compiler import CompileError, Cube, compile_cube
from boundarybench.jsonsafe import JsonSafetyError
from boundarybench.ledger import LedgerError
from boundarybench.loader import load_card
from boundarybench.manifest import ManifestError, verify_manifest
from boundarybench.pricing import PricingError, price_for, pricing_policy_payload
from boundarybench.providermatrix import (
    EPISODES_PER_MODEL,
    MATRIX_MODELS,
    METHODOLOGY_NOTE,
    STAGE_ORDER,
    STAGE_SMOKE,
    TOTAL_EPISODES,
    TRIALS,
    MatrixError,
    build_matrix_plan,
    check_global_episode_ceiling,
    check_matrix_trials,
    proportional_allocations,
    stage_cumulative_episode_target,
    stage_is_manual,
    stage_new_episodes,
)
from boundarybench.providers.anthropic_messages import (
    API_KEY_VARIABLE,
    DISABLED_THINKING,
    LIVE_CONFIGURATION_VARIABLE,
    build_anthropic_adapter,
)
from boundarybench.providers.openai_responses import (
    REASONING_EFFORT_NONE as REASONING_DISABLED_EFFORT,
)
from boundarybench.providers.registry import PROVIDER_ADAPTERS, provider_choice
from boundarybench.runmanifest import (
    MANIFEST_FILENAME,
    RunLimits,
    RunManifestError,
    build_run_manifest,
    check_run_path_safety,
    open_run_session,
)
from boundarybench.runner import (
    INVOCATION_STAGING_NOTE,
    RunnerError,
    RunReport,
    check_scaffold_contract,
    execute_run,
    parse_stop_after,
)
from boundarybench.scaffold import STANDARD_SCAFFOLD, ScaffoldError, load_scaffold
from boundarybench.schema import SchemaError
from boundarybench.solvers import check_solvers
from boundarybench.suite import (
    CONSTANT_STRATEGY_NOTE,
    SUITE_SCOPE,
    SuiteError,
    SuiteReport,
    check_suite,
    validate_suite,
)

#: The deterministic in-process adapter. Not a model, and never a result.
FAKE_ADAPTER = "fake-scripted"
#: The Anthropic provider adapter. Requires a credential and an explicit model.
#:
#: Kept as its own name although it is now one entry in
#: :data:`~boundarybench.providers.registry.PROVIDER_ADAPTERS`: it is referenced
#: by existing tests and documentation, and the string an operator types has not
#: changed.
ANTHROPIC_ADAPTER = "anthropic"

#: Every provider adapter, derived from the registry rather than restated.
#:
#: Derived on purpose. A hand-maintained tuple beside a registry is two lists
#: that agree until someone adds a provider to one of them, and the failure is
#: quiet: a provider the CLI cannot name, or one it names and cannot build.
PROVIDER_ADAPTER_NAMES: tuple[str, ...] = tuple(sorted(PROVIDER_ADAPTERS))

#: What ``--adapter`` accepts. The fake first, because it is the only one that
#: contacts nothing and the only one a reader should reach for by default.
ADAPTERS: tuple[str, ...] = (FAKE_ADAPTER, *PROVIDER_ADAPTER_NAMES)

DEFAULT_TRIALS = 1
DEFAULT_MAX_TURNS = 12
DEFAULT_MAX_MESSAGES = 64
DEFAULT_TIMEOUT_SECONDS = 30.0

EXPECTED_VARIANTS = 6

#: What a preflight assumes one request's input costs, for planning only.
#:
#: Generous relative to this scaffold's own requests, and deliberately not an
#: enforced ceiling: what is enforced is ``--max-cost-usd``, which the runner
#: checks per request against that request's own measured size. This number
#: exists so a plan's worst case can be stated *before* the run, and it is
#: reported beside the bound it produced so a reader can see what was assumed.
DEFAULT_PLANNING_INPUT_TOKENS = 32_768

PROG = "boundarybench"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Compile and check BoundaryBench Sparse Autonomy Cubes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="check a construct card against the schema and compiler"
    )
    validate.add_argument("card", type=Path)

    compile_cmd = subparsers.add_parser(
        "compile", help="compile a construct card into its six task variants"
    )
    compile_cmd.add_argument("card", type=Path)
    compile_cmd.add_argument("--json", action="store_true", help="emit JSON")
    compile_cmd.add_argument("--output", type=Path, help="write the output to a file")

    solvers = subparsers.add_parser(
        "check-solvers", help="run the causal CI gate over reference and negative solvers"
    )
    solvers.add_argument("card", type=Path)
    solvers.add_argument("--json", action="store_true", help="emit JSON")

    verify = subparsers.add_parser(
        "verify-manifest",
        help="prove a card still matches its committed semantic manifest",
    )
    verify.add_argument("card", type=Path)
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--json", action="store_true", help="emit JSON")

    validate_suite_cmd = subparsers.add_parser(
        "validate-suite",
        help="validate a suite manifest, its artefacts and its declared constraints",
    )
    validate_suite_cmd.add_argument("suite", type=Path)
    validate_suite_cmd.add_argument("--json", action="store_true", help="emit JSON")

    check_suite_cmd = subparsers.add_parser(
        "check-suite",
        help="validate a suite and run every Cube's causal CI gate",
    )
    check_suite_cmd.add_argument("suite", type=Path)
    check_suite_cmd.add_argument("--json", action="store_true", help="emit JSON")

    run_suite_cmd = subparsers.add_parser(
        "run-suite",
        help=(
            "execute every variant against a deterministic fake adapter and record "
            "the run (infrastructure smoke only; produces no model result)"
        ),
    )
    run_suite_cmd.add_argument("suite", type=Path)
    run_suite_cmd.add_argument(
        "--adapter",
        required=True,
        choices=list(ADAPTERS),
        help=(
            f"which adapter to run: {FAKE_ADAPTER} executes the deterministic "
            f"in-process fake and needs no credential; {ANTHROPIC_ADAPTER} calls the "
            f"Anthropic Messages API and needs --model and {API_KEY_VARIABLE}"
        ),
    )
    run_suite_cmd.add_argument(
        "--model",
        default=None,
        help=(
            "the provider model identifier, required by --adapter "
            f"{ANTHROPIC_ADAPTER}. Pin a dated snapshot id: an alias moves, and a "
            "run recorded against one is not comparable with anything"
        ),
    )
    run_suite_cmd.add_argument(
        "--output-dir", required=True, type=Path, help="the run directory"
    )
    run_suite_cmd.add_argument(
        "--scaffold",
        type=Path,
        default=None,
        help=f"scaffold artefact to run under (default: {STANDARD_SCAFFOLD.name})",
    )
    run_suite_cmd.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    run_suite_cmd.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    run_suite_cmd.add_argument(
        "--max-messages",
        type=int,
        default=DEFAULT_MAX_MESSAGES,
        help=("per-episode message budget; one turn costs one request and one response"),
    )
    run_suite_cmd.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="per-episode wall-clock budget in seconds",
    )
    run_suite_cmd.add_argument(
        "--setting",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="a non-secret adapter setting; repeatable",
    )
    run_suite_cmd.add_argument("--json", action="store_true", help="emit JSON")
    _add_limit_arguments(run_suite_cmd)
    _add_staging_argument(run_suite_cmd)

    preflight_cmd = subparsers.add_parser(
        "preflight",
        help=(
            "check a planned provider run without contacting anything: suite, "
            "scaffold, models, settings, pricing, caps, credential presence and "
            "output-directory safety, plus the plan's cost upper bound"
        ),
    )
    preflight_cmd.add_argument("suite", type=Path)
    preflight_cmd.add_argument(
        "--adapter",
        required=True,
        choices=list(PROVIDER_ADAPTER_NAMES),
        help="which provider adapter to plan for",
    )
    preflight_cmd.add_argument("--model", default=None)
    preflight_cmd.add_argument("--output-dir", required=True, type=Path)
    preflight_cmd.add_argument("--scaffold", type=Path, default=None)
    preflight_cmd.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    preflight_cmd.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    preflight_cmd.add_argument("--max-messages", type=int, default=DEFAULT_MAX_MESSAGES)
    preflight_cmd.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    preflight_cmd.add_argument(
        "--planning-input-tokens",
        type=int,
        default=DEFAULT_PLANNING_INPUT_TOKENS,
        help=(
            "the per-request input-token size this plan's cost upper bound assumes. "
            "A planning assumption, not an enforced limit: the enforced limit is "
            "--max-cost-usd, which the runner checks against each request's own "
            "measured size"
        ),
    )
    preflight_cmd.add_argument("--setting", action="append", default=[])
    preflight_cmd.add_argument("--json", action="store_true", help="emit JSON")
    _add_limit_arguments(preflight_cmd)
    _add_staging_argument(preflight_cmd)

    matrix = subparsers.add_parser(
        "provider-matrix",
        help=(
            "plan the private five-model methodology spike offline: contacts no "
            "provider, creates no run directory, and reports the configuration "
            "identity an operator would have to authorise for each cell"
        ),
    )
    matrix.add_argument("suite", type=Path)
    matrix.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="the directory each model's own run directory would be created under",
    )
    matrix.add_argument(
        "--stage",
        default=STAGE_SMOKE,
        help=f"which stage of the approved order to plan for: {list(STAGE_ORDER)}",
    )
    matrix.add_argument(
        "--execute",
        action="store_true",
        help=(
            "refused: this build authorises no live matrix traffic and ships no "
            "pre-approved authorisation"
        ),
    )
    matrix.add_argument("--scaffold", type=Path, default=None)
    matrix.add_argument(
        "--trials",
        type=int,
        default=TRIALS,
        help=(
            f"refused unless it is exactly {TRIALS}. This matrix is a fixed "
            "methodology rather than a parameterised run, and every number it "
            "reports is derived from that count. `run-suite --trials` takes any"
        ),
    )
    matrix.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    matrix.add_argument("--max-messages", type=int, default=DEFAULT_MAX_MESSAGES)
    matrix.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    matrix.add_argument("--json", action="store_true", help="emit JSON")
    # ``--max-cost-usd`` arrives with the shared limit arguments, and it is
    # required here rather than optional: the whole plan is a split of one
    # authorised total, and there is nothing to split without it.
    #
    # ``--max-episodes`` arrives from the same place but means something
    # narrower here, so it is defaulted and re-described rather than accepted as
    # the general ceiling: this command plans one fixed matrix, and the only
    # ceiling that authorises it is the whole of it.
    _add_limit_arguments(
        matrix, require_cost_cap=True, exact_episode_ceiling=TOTAL_EPISODES
    )

    return parser


def _add_limit_arguments(
    command: argparse.ArgumentParser,
    *,
    require_cost_cap: bool = False,
    exact_episode_ceiling: int | None = None,
) -> None:
    """The two hard limits, spelled the same way wherever they are accepted.

    ``require_cost_cap`` exists for the one command whose entire output is a
    split of an authorised total. Everywhere else the cap is optional, because a
    run against the in-process fake buys nothing and a single provider run may
    legitimately be capped by episodes alone.

    ``exact_episode_ceiling`` exists for the same command, and for the same
    reason in the other direction: it plans one fixed matrix, so the only
    episode ceiling that authorises it is the whole of it. The default and the
    help text are set here rather than checked silently later, because an option
    documented as a hard ceiling and then ignored tells an operator their
    ceiling was applied when nothing looked at it.
    """
    command.add_argument(
        "--max-cost-usd",
        default=None,
        required=require_cost_cap,
        metavar="AMOUNT",
        help=(
            "the hard cost ceiling for this run, in USD, as an exact decimal. "
            "Requires a model this build holds a reviewed price for. No provider "
            "request is dispatched whose conservative upper bound would take the "
            "run past it"
        ),
    )
    command.add_argument(
        "--max-episodes",
        type=int,
        default=exact_episode_ceiling,
        metavar="N",
        help=(
            "the hard ceiling on planned and executed episodes. A plan larger than "
            "this is refused before a run directory exists"
        )
        if exact_episode_ceiling is None
        else (
            f"the one global episode ceiling this plan runs under. Exactly "
            f"{exact_episode_ceiling} is accepted and it is the default: this "
            "matrix is a fixed methodology, so a smaller ceiling is one the plan "
            "cannot run under and a larger one authorises episodes nothing would "
            "execute"
        ),
    )


def _add_staging_argument(command: argparse.ArgumentParser) -> None:
    """The invocation limit — deliberately not one of the two hard limits above.

    Kept out of :func:`_add_limit_arguments` because it is a different kind of
    thing and the difference is the whole point: those two are recorded in the
    manifest and bound the run, this one is recorded nowhere and bounds one
    invocation of it. Taken as text rather than ``type=int`` for the same reason
    ``--max-cost-usd`` is: what an operator typed is checked exactly as they
    typed it, and a value this build will not execute is refused with a sentence
    rather than reinterpreted.
    """
    command.add_argument(
        "--stop-after",
        default=None,
        metavar="N",
        help=(
            "execute at most N new episodes in this invocation, then return "
            "normally. For staging a run that must be audited part-way: the plan, "
            "the caps, the pricing and every episode's identity are unchanged, "
            "nothing about the limit is written to the manifest or the ledger, and "
            "resuming the same run directory continues from the next planned "
            "episode. It can only stop this invocation earlier — a cost cap, an "
            "episode ceiling or a run-terminal failure still stops it first"
        ),
    )


def _load_cube(path: Path) -> Cube:
    return compile_cube(load_card(path))


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


# -- commands ---------------------------------------------------------------


def _cmd_validate(args: argparse.Namespace) -> int:
    cube = _load_cube(args.card)
    card = cube.card
    print(f"{card.cube_id}: OK")
    print(f"  workflow          {card.workflow.strip().splitlines()[0]}")
    print(f"  cube type         {card.cube_type}")
    print(f"  fact location     {card.fact_location}")
    print(f"  privacy status    {card.privacy_status}")
    print(f"  variants          {len(cube.variants)}")
    print(f"  live state edge   {[e[1] + ' -> ' + e[2] for e in cube.live_state_edges]}")
    print(f"  live policy edge  {[e[1] + ' -> ' + e[2] for e in cube.live_policy_edges]}")
    print(f"  constant strategy {dict(cube.constant_strategy_scores)} of 6")
    print(f"  disclaimer        {' '.join(card.disclaimer.split())}")
    return 0


def _cmd_compile(args: argparse.Namespace) -> int:
    cube = _load_cube(args.card)
    if len(cube.variants) != EXPECTED_VARIANTS:  # pragma: no cover
        # Defensive: the compiler always emits four cells and two probes, so
        # this can only fire if that contract regresses.
        return _fail(
            f"{cube.cube_id}: expected {EXPECTED_VARIANTS} variants, "
            f"got {len(cube.variants)}"
        )

    if args.json:
        rendered = json.dumps(cube.as_dict(), indent=2, sort_keys=False)
    else:
        lines = [f"{cube.cube_id}: {len(cube.variants)} variants"]
        for variant in cube.variants:
            pressure = variant.pressure_direction or "-"
            lines.append(
                f"  {variant.variant_id}\n"
                f"    disposition {variant.expected_disposition:<4} "
                f"reason {variant.expected_primary_reason}\n"
                f"    pressure    {pressure:<4} "
                f"required evidence {list(variant.required_evidence_ids)}\n"
                f"    digest      {variant.content_digest[:16]}"
            )
        rendered = "\n".join(lines)

    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"{cube.cube_id}: {len(cube.variants)} variants written to {args.output}")
    else:
        print(rendered)
    return 0


def _cmd_check_solvers(args: argparse.Namespace) -> int:
    cube = _load_cube(args.card)
    report = check_solvers(cube)

    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=False))
        return 0 if report.ok else 1

    print(f"{cube.cube_id}: causal CI")
    for outcome in report.outcomes:
        status = "OK" if outcome.satisfied else "FAILED"
        print(
            f"  {outcome.name:<19} {outcome.kind:<9} "
            f"{outcome.passed_variants}/{outcome.total_variants} variants  {status}"
        )
        print(f"      {outcome.description}")
        for failure in outcome.failures:
            print(f"      ! {failure}")
    print(f"causal CI: {'OK' if report.ok else 'FAILED'}")
    return 0 if report.ok else 1


def _cmd_verify_manifest(args: argparse.Namespace) -> int:
    verification = verify_manifest(args.card, args.manifest)
    payload = verification.as_dict()

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=False))
        return 0

    print(f"{payload['cube_id']}: manifest OK")
    print(f"  card              {payload['card']}")
    print(
        f"  manifest          {payload['manifest']} "
        f"(format v{payload['manifest_version']})"
    )
    print(f"  cube type         {payload['cube_type']}")
    print(f"  fact location     {payload['fact_location']}")
    print(f"  privacy status    {payload['privacy_status']}")
    print(f"  card fingerprint  {payload['card_fingerprint_sha256']}")
    print(f"  variants          {payload['variant_count']} pinned content digests match")
    print(f"  scope             {SUITE_SCOPE}")
    return 0


def _print_suite_summary(payload: dict[str, Any]) -> None:
    print(f"{payload['suite_id']}: suite OK")
    print(f"  benchmark version {payload['benchmark_version']}")
    print(f"  status            {payload['status']}")
    print(f"  privacy status    {payload['privacy_status']}")
    print(f"  cubes             {payload['cube_count']}")
    print(f"  variants          {payload['variant_count']}")
    print(f"  cube types        {payload['cube_types']}")
    print(f"  fact locations    {payload['fact_locations']}")
    print(f"  dispositions      {payload['expected_dispositions']}")
    print(
        f"  constant strategy {payload['constant_strategy_scores']} "
        f"of {payload['variant_count']}"
    )
    print(f"                    {CONSTANT_STRATEGY_NOTE}")
    print(f"  suite digest      {payload['suite_content_digest_sha256']}")
    for cube in payload["cubes"]:
        counts = cube["expected_dispositions"]
        print(
            f"    {cube['cube_id']:<38} {cube['cube_type']:<6} "
            f"{cube['fact_location']:<14} {cube['variant_count']} variants  "
            f"ACT {counts['ACT']} / STOP {counts['STOP']}"
        )
    print(f"  scope             {SUITE_SCOPE}")
    print(f"  disclaimer        {payload['disclaimer']}")


def _validated_suite(args: argparse.Namespace) -> SuiteReport:
    return validate_suite(args.suite)


def _cmd_validate_suite(args: argparse.Namespace) -> int:
    payload = _validated_suite(args).as_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=False))
        return 0
    _print_suite_summary(payload)
    return 0


def _cmd_check_suite(args: argparse.Namespace) -> int:
    report = check_suite(_validated_suite(args))
    payload = report.as_dict()

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=False))
        return 0 if report.ok else 1

    _print_suite_summary(payload["suite"])
    for entry in payload["causal_ci"]:
        print(f"{entry['cube_id']}: causal CI")
        for outcome in entry["outcomes"]:
            status = "OK" if outcome["satisfied"] else "FAILED"
            print(
                f"  {outcome['name']:<19} {outcome['kind']:<9} "
                f"{outcome['passed_variants']}/{outcome['total_variants']} "
                f"variants  {status}"
            )
            for failure in outcome["failures"]:
                print(f"      ! {failure}")
    print(f"suite causal CI: {'OK' if report.ok else 'FAILED'}")
    return 0 if report.ok else 1


def _parse_settings(pairs: Sequence[str]) -> dict[str, str]:
    """Read ``key=value`` settings. Values stay strings: no type guessing.

    Inferring that ``0`` means the integer zero would make the recorded settings
    — and so the run id — depend on a guess about what a provider wanted.
    """
    settings: dict[str, str] = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator or not key.strip():
            raise RunManifestError(f"adapter setting {pair!r} must be given as key=value")
        if key in settings:
            raise RunManifestError(f"adapter setting {key!r} was given twice")
        settings[key] = value
    return settings


def _run_report_lines(payload: dict[str, Any]) -> list[str]:
    counts = payload["counts"]
    outcomes = payload["outcome_counts"]
    errors = payload["error_class_counts"]
    adapter = payload["adapter"]
    limits = payload["limits"]
    retries = payload["retry_policy"]
    rows: list[tuple[str, str]] = [
        ("track", payload["track"]),
        ("status", payload["status"]),
        ("suite", payload["suite_id"]),
        ("suite digest", payload["suite_content_digest"]),
        ("scaffold", payload["scaffold_id"]),
        ("scaffold digest", payload["scaffold_content_digest"]),
        (
            "adapter",
            f"{adapter['provider']}/{adapter['model']} "
            f"({adapter['implementation']} {adapter['version']})",
        ),
        ("trials", str(payload["trials"])),
        (
            "limits",
            f"{limits['max_turns']} turn(s), {limits['max_messages']} message(s), "
            f"{limits['episode_timeout_seconds']}s per episode",
        ),
        ("retry policy", f"{retries['retries']} retries ({retries['policy']})"),
        # Two ids and a timestamp, never one id. The configuration is what makes
        # two runs the same experiment; the execution is which run this was.
        ("configuration id", payload["configuration_id"]),
        ("execution id", payload["execution_id"]),
        ("created at", payload["created_at_utc"]),
        ("run directory", payload["run_directory"]),
        ("manifest", payload["manifest_path"]),
        ("ledger", payload["ledger_path"]),
        ("planned", str(counts["planned"])),
        ("completed", str(counts["completed"])),
        ("executed", str(counts["executed"])),
        ("resumed", str(counts["resumed"])),
        ("succeeded", str(counts["succeeded"])),
        ("failed", str(counts["failed"])),
        ("verdict passed", str(counts["verdict_passed"])),
        ("verdict failed", str(counts["verdict_failed"])),
        ("verdict absent", str(counts["verdict_absent"])),
        ("samples", _sample_line(payload)),
        (
            "outcomes",
            " ".join(f"{name} {count}" for name, count in outcomes.items()) or "-",
        ),
        (
            "error classes",
            " ".join(f"{name} {count}" for name, count in errors.items()) or "-",
        ),
        ("cost cap", _cost_line(payload)),
        ("usage totals", _usage_line(payload)),
        ("provider attempts", _attempt_line(payload)),
        ("measurement", _measurement_line(payload)),
    ]
    return [f"  {label:<18} {value}" for label, value in rows]


def _cost_line(payload: dict[str, Any]) -> str:
    """The cap, what was measured, and what is being held as exposure.

    A run with no cap says so rather than printing zeros: nothing was priced, so
    there is nothing to report and a row of noughts would look like a
    measurement.
    """
    accounting = payload["cost_accounting"]
    if accounting is None:
        return "none (this run pinned no price and measured no cost)"
    return (
        f"USD {accounting['max_cost_usd']}; measured {accounting['measured_usd']}, "
        f"exposure {accounting['exposure_usd']}, remaining "
        f"{accounting['remaining_usd']}"
    )


def _sample_line(payload: dict[str, Any]) -> str:
    """The three sample statuses, always all three, always in a fixed order."""
    counts = payload["sample_counts"]
    return " ".join(f"{name} {counts[name]}" for name in sorted(counts))


def _attempt_line(payload: dict[str, Any]) -> str:
    """Provider requests and the faults they met, without inventing a zero."""
    attempts = payload["provider_attempts"]
    faults = payload["provider_fault_counts"]
    total = "-" if attempts["attempts"] is None else attempts["attempts"]
    turns = "-" if attempts["turns"] is None else attempts["turns"]
    met = " ".join(f"{name} {count}" for name, count in faults.items()) or "-"
    return f"{total} attempt(s) over {turns} provider turn(s); faults {met}"


def _measurement_line(payload: dict[str, Any]) -> str:
    coverage = payload["measurement_coverage"]
    episodes = coverage["episodes"]
    attempts = coverage["attempts"]
    return (
        f"episodes full {episodes['full']} partial {episodes['partial']} "
        f"unmeasured {episodes['unmeasured']}; turns "
        f"{coverage['turns']['measured']}/{coverage['turns']['total']}; attempts "
        f"{'-' if attempts['measured'] is None else attempts['measured']}/"
        f"{'-' if attempts['total'] is None else attempts['total']}"
    )


def _usage_line(payload: dict[str, Any]) -> str:
    """Render the totals without turning "not measured" into a number."""
    totals = payload["usage_totals"]
    measured = payload["usage_measured_episodes"]
    return " ".join(
        f"{name} {'-' if totals[name] is None else totals[name]}"
        f"({measured[name]}/{payload['counts']['completed']} measured)"
        for name in totals
    )


def _staging_line(payload: dict[str, Any]) -> str:
    """What this invocation was allowed to execute, and what is left of the plan.

    Both halves, always. The limit alone would not say whether it bound
    anything, and the counts alone would not say why the run stopped where it
    did with episodes still pending.
    """
    staging = payload["invocation_staging"]
    counts = payload["counts"]
    pending = counts["planned"] - counts["completed"]
    reached = "reached" if staging["stopped_early"] else "not reached"
    return (
        f"--stop-after {staging['stop_after']} ({reached}); "
        f"{counts['executed']} new episode(s) this invocation, "
        f"{counts['completed']}/{counts['planned']} of the plan durable, "
        f"{pending} still pending"
    )


def _cost_controls(args: argparse.Namespace) -> CostControls:
    """Resolve this invocation's hard limits, before anything is built.

    Called before the adapter, so before a provider client exists and before a
    run directory could be created. An unpriced model under a cost cap fails
    here: a cap this build cannot measure spend against is not a control, and
    the only alternatives — a neighbouring model's price, a zero, a null — are
    each a different way of claiming a limit that is not being enforced.
    """
    cap = None
    if args.max_cost_usd is not None:
        if args.adapter not in PROVIDER_ADAPTERS:
            raise RunManifestError(
                f"--max-cost-usd applies to a run that spends money, and --adapter "
                f"{FAKE_ADAPTER} contacts no service and buys nothing. A dollar cap "
                "on an in-process fake would record a limit that never bound "
                "anything"
            )
        try:
            cap = parse_cost_cap(args.max_cost_usd)
        except BudgetError as exc:
            raise RunManifestError(str(exc)) from exc
    price = None
    if cap is not None:
        if not args.model:
            raise RunManifestError(
                "--max-cost-usd requires --model: a cost cap is enforced against a "
                "reviewed price for one named model"
            )
        # Priced under *this invocation's* provider, not a build-wide default.
        # A model identifier is only unique within a provider, so pricing by
        # model alone would return a confident rate for a model that never ran.
        try:
            price = price_for(
                provider=provider_choice(args.adapter).provider, model=args.model
            )
        except PricingError as exc:
            raise RunManifestError(str(exc)) from exc
    try:
        return CostControls(max_cost_usd=cap, max_episodes=args.max_episodes, price=price)
    except BudgetError as exc:
        raise RunManifestError(str(exc)) from exc


def _cost_guard(
    controls: CostControls, adapter: str = ANTHROPIC_ADAPTER
) -> RunCostGuard | None:
    """The guard a capped run carries, or ``None`` when nothing is capped.

    The ceiling and the attempt count come from the provider that will actually
    send the requests. Every provider in this build pins the same two numbers
    today, and reading them from the registry rather than from one integration's
    constants is what keeps that a checkable fact instead of a coincidence a
    fifth provider would quietly break.
    """
    if not controls.enforces_cost:
        return None
    choice = provider_choice(adapter)
    return RunCostGuard(
        controls=controls,
        max_output_tokens=choice.max_output_tokens,
        max_attempts_per_turn=choice.retry_attempts(),
    )


def _adapter_factory(choice: Any) -> Any:
    """How this invocation builds its adapter.

    Every provider but one is built straight through the registry. Anthropic is
    built through this module's own ``build_anthropic_adapter`` name, and that is
    a deliberate compatibility seam rather than a leftover: the Anthropic
    integration predates the registry, and the tests that exercise a full
    provider run end to end substitute an in-process transport by rebinding that
    name here. Routing it through the registry instead would rebind nothing, and
    those tests would silently start trying to reach a real service — which is
    the one failure mode this repository cannot allow a test suite to have.

    The behaviour is identical either way: the registry's Anthropic entry calls
    exactly this function.
    """
    if choice.name == ANTHROPIC_ADAPTER:
        return build_anthropic_adapter
    return choice.build


def _select_adapter(
    args: argparse.Namespace, cost_guard: RunCostGuard | None = None
) -> tuple[Any, Any, dict[str, Any]]:
    """Build the adapter this invocation asked for, or refuse the invocation.

    Everything that can fail about *configuration* fails here, which is before
    :func:`open_run_session` creates a directory, takes a lock or writes a
    manifest. A missing credential therefore costs an error message and nothing
    else: there is no run to find afterwards, and so nothing that claims a
    provider was contacted when it was not.
    """
    if args.adapter in PROVIDER_ADAPTERS:
        choice = provider_choice(args.adapter)
        if not args.model:
            raise RunManifestError(
                f"--adapter {choice.name} requires --model. Nothing here picks "
                "a default: a benchmark run has to name the model it measured, and "
                "a build that chose one silently would choose a moving alias"
            )
        if args.setting:
            raise RunManifestError(
                f"--adapter {choice.name} takes no --setting. Every "
                "result-affecting request setting is pinned by this build and "
                "recorded in run identity, and an operator-supplied setting is "
                "free text that lands in the run manifest — which is not somewhere "
                "a credential pasted under an innocuous key may end up"
            )
        # Checks the authorised-configuration gate and then reads this provider's
        # one credential variable and builds its SDK client, so an unauthorised
        # configuration, a missing credential or a redirected endpoint raises
        # here rather than mid-run. The identity comes from the environment
        # because it is the operator's declaration of *which* approved
        # configuration this invocation is; the value it has to equal is checked
        # against the manifest below, once the manifest exists and before any
        # request is sent.
        adapter = _adapter_factory(choice)(
            model=args.model,
            configuration_id=os.environ.get(LIVE_CONFIGURATION_VARIABLE),
            cost_guard=cost_guard,
        )
        return adapter, adapter.identity, dict(adapter.settings)
    if args.model:
        raise RunManifestError(
            f"--adapter {FAKE_ADAPTER} names one fixed model and --model is not the "
            "operator's to set; a fake that could claim a model name would produce "
            "a ledger asserting a provider that was never contacted"
        )
    settings = _parse_settings(args.setting)
    # Its identity and its behaviour are both fixed by released code: there is no
    # way to reach the runner from here with a caller-supplied script or a
    # caller-supplied provider, so a ``fake-scripted`` run always means the same
    # thing.
    return PolicyFollowingFakeAdapter(settings=settings), fake_identity(), settings


def _cmd_run_suite(args: argparse.Namespace) -> int:
    # First of all, before the suite is even read: a malformed staging limit
    # costs an error message and nothing else — no run directory, no client and
    # no request.
    stop_after = parse_stop_after(args.stop_after)
    suite = validate_suite(args.suite)
    scaffold = load_scaffold(args.scaffold or STANDARD_SCAFFOLD)
    # Before the adapter, so an unpriced model under a cap costs an error rather
    # than a client holding a credential.
    controls = _cost_controls(args)
    guard = _cost_guard(controls, args.adapter)
    adapter, identity, settings = _select_adapter(args, guard)

    manifest = build_run_manifest(
        suite=suite,
        scaffold=scaffold,
        provider=identity.provider,
        model=identity.model,
        implementation=identity.implementation,
        adapter_version=identity.version,
        adapter_settings=settings,
        trials=args.trials,
        limits=RunLimits(
            max_turns=args.max_turns,
            max_messages=args.max_messages,
            episode_timeout_seconds=args.timeout,
        ),
        cost_controls=controls,
    )
    # The declared identity is what authorised the client; this is the identity
    # the run actually has. They are checked against each other here — after the
    # manifest exists and before ``open_run_session`` creates a directory, takes
    # a lock or lets a single request out — because an authorisation for one
    # configuration is not an authorisation for another. Skipped when nothing was
    # declared, which for the live path is unreachable: ``build_anthropic_adapter``
    # has already refused an invocation that declared nothing.
    declared = os.environ.get(LIVE_CONFIGURATION_VARIABLE, "").strip()
    if declared and declared != manifest.configuration_id:
        raise RunManifestError(
            f"{LIVE_CONFIGURATION_VARIABLE} authorises configuration {declared}, "
            f"but this invocation is configuration {manifest.configuration_id}. A "
            "configuration identity covers the suite, the scaffold, the model, "
            "every request setting and every limit, so an authorisation for one "
            "of them authorises no other. Re-run `preflight` and authorise what "
            "it reports"
        )
    # The lock is held for the whole run: manifest open, ledger validation,
    # every episode and the final read-back all happen inside this block.
    with open_run_session(args.output_dir, manifest) as session:
        report: RunReport = execute_run(
            suite=suite,
            scaffold=scaffold,
            adapter=adapter,
            session=session,
            cost_guard=guard,
            stop_after=stop_after,
        )
    payload = report.as_dict()

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=False))
    else:
        print(f"{payload['suite_id']}: {args.adapter} run")
        for line in _run_report_lines(payload):
            print(line)
        print(f"  counts             {payload['counts_note']}")
        print(f"  plan               {payload['plan_note']}")
        print(f"  usage              {payload['usage_note']}")
        print(f"  samples            {payload['sample_eligibility_note']}")
        print(f"  measurement        {payload['measurement_note']}")
        if payload["limits_stop"] is not None:
            print(
                f"  limit reached      {payload['limits_stop']['reason']}: "
                f"{payload['limits_stop']['note']}"
            )
        # Printed only when one was asked for, and always with the sentence that
        # says what it is not: an unstaged run has no staging to report, and a
        # staged one must not be read as a run configured to be smaller.
        if payload["invocation_staging"]["stop_after"] is not None:
            print(f"  staging            {_staging_line(payload)}")
            print(f"  staging note       {payload['invocation_staging']['note']}")
        if payload["cost_accounting"] is not None:
            print(f"  cost               {payload['cost_accounting']['note']}")
        if payload["run_terminated"] is not None:
            terminated = payload["run_terminated"]
            print(
                f"  run terminated     at {terminated['episode_id']} with outcome "
                f"{terminated['outcome']}: {terminated['note']}"
            )
        print(f"  ledger integrity   {payload['ledger_integrity_note']}")
        # The run's own scope, read off the manifest the run is executing under,
        # not a module constant: a provider run and a fake run are two different
        # claims and printing one build-wide sentence made them look like one.
        print(f"  scope              {payload['scope']}")
    # A run whose episodes did not all complete is a failed gate, even though
    # each individual failure is recorded rather than raised.
    return 0 if payload["counts"]["failed"] == 0 else 1


# -- preflight ---------------------------------------------------------------
#
# Everything a credentialed operator can check *before* spending anything, and
# nothing that spends. No client is constructed, no request body is built and no
# run directory, manifest, ledger or lock is created — on success or on failure.
# The identity it reports is the identity the run would have, because it is
# produced by the same builder the run uses.

#: Said in every preflight report, because a passing preflight is easy to
#: over-read.
PREFLIGHT_NOTE = (
    "A preflight contacts nothing. It proves that the suite, the scaffold, the "
    "model, the pinned settings, the pricing policy and the caps are the ones "
    "this plan names, that a credential is present, that the output directory is "
    "a safe private place to write evidence, and that the plan's conservative "
    "worst case fits inside the authorised cap. It does not prove that the "
    "credential works, that the model is available to this account, or that the "
    "run will cost what the bound says: the bound is an upper limit under a "
    "stated per-request planning size, and the enforced limit is the cost cap, "
    "which the runner checks against each request's own measured size."
)

#: Said about the credential, because "present" is a much weaker claim than the
#: word invites.
CREDENTIAL_NOTE = (
    "Presence only. The variable was read to see whether it holds a non-empty "
    "value; the value itself is never inspected, logged, hashed or recorded, and "
    "nothing here proves it is a valid credential."
)


def _check_output_directory(path: Path) -> dict[str, Any]:
    """Prove the run's evidence would land somewhere safe and private.

    Two different questions. *Safe*: no component of the path is a symlink, so
    the manifest and ledger cannot be redirected — the same check the run itself
    makes, run early so the answer is known before anything is spent. *Private*:
    if the directory already exists, it may not be readable or writable by group
    or other, because what a provider run writes is unreviewed model output and
    a research directory is not a shared one.
    """
    check_run_path_safety(path)
    exists = path.exists()
    mode: str | None = None
    if exists:
        mode = oct(path.stat().st_mode & 0o777)
        if path.stat().st_mode & 0o077:
            raise RunManifestError(
                f"the output directory {path} is accessible to group or other "
                f"(mode {mode}). A provider run writes unreviewed model output and "
                "run evidence; make it private (chmod 700) or choose a directory "
                "that already is"
            )
    return {
        "path": str(path),
        "exists": exists,
        "mode": mode,
        "holds_a_run": exists and (path / MANIFEST_FILENAME).exists(),
    }


#: What this build says about a reasoning default it could not establish.
#:
#: Said only about a model whose profile actually leaves the field out and whose
#: request contract this build could not verify. A profile that sends an
#: explicit, documented "no reasoning" instruction is in the same position as
#: the Anthropic one — it can state the answer as a fact — and reporting this
#: sentence against it would describe a request that was not made.
REASONING_DEFAULT_UNVERIFIED = (
    "this build sends no reasoning or thinking field to this model and did not "
    "verify, against the provider's published documentation, what the model does "
    "when the field is absent. Whether optional extended reasoning is active is "
    "therefore not established here rather than asserted either way; an operator "
    "authorising live traffic is authorising that uncertainty along with the rest "
    "of the configuration"
)


def _preflight_scope(
    choice: Any, settings: dict[str, Any], scaffold: Any
) -> dict[str, Any]:
    """What the planned request is, and is not, in scope for.

    Assembled from the settings this provider actually records rather than from
    a profile object, because ``settings`` is what the configuration identity
    hashes and what the manifest stores — so a scope statement derived from it
    cannot disagree with the run it describes.

    The reasoning question is answered from the request the plan would actually
    send, and the two possible answers are genuinely different claims.

    A profile that sends an explicit instruction switching the feature *off*,
    against a request contract this build verified against the model's published
    page, lets the plan state as a fact that optional extended reasoning is not
    in scope — Anthropic's ``thinking: {"type": "disabled"}`` block and the
    OpenAI lane's ``reasoning: {"effort": "none"}`` object are the same claim in
    two vendors' vocabularies. Reading the mere presence of either key as the
    feature being *on* would report a scope breach for the request that exists to
    prevent one.

    A profile that sends no such field is the other case: what the model does in
    its absence is unverified, and is reported as unverified rather than guessed
    in either direction. See :data:`REASONING_DEFAULT_UNVERIFIED`.
    """
    scope: dict[str, Any] = {
        "sampling_parameters_sent": (
            {}
            if settings.get("temperature") is None
            else {"temperature": settings["temperature"]}
        ),
        "sampling_parameters_omitted": list(settings["request_fields_omitted"]),
        "external_provider_tools": False,
        "tools": sorted(action.name for action in scaffold.actions),
        "rejected_environment_variables": list(choice.rejected_variables),
    }
    if choice.name == ANTHROPIC_ADAPTER:
        thinking = settings["thinking"]
        scope["optional_extended_thinking"] = (
            thinking is not None and thinking.get("type") != DISABLED_THINKING["type"]
        )
        scope["thinking"] = thinking
        return scope
    # Named per provider because the field is: OpenAI's Responses API takes a
    # ``reasoning`` object and xAI's compatibility surface takes a
    # ``reasoning_effort`` string. Whichever this provider has is reported as
    # sent, including when what it is is ``None``.
    for field in ("reasoning", "reasoning_effort"):
        if field in settings:
            scope[field] = settings[field]
    if settings.get("model_contract_verified") and _disables_reasoning(settings):
        scope["optional_extended_thinking"] = False
        return scope
    scope["optional_extended_thinking"] = None
    scope["optional_extended_thinking_note"] = REASONING_DEFAULT_UNVERIFIED
    return scope


def _disables_reasoning(settings: dict[str, Any]) -> bool:
    """Whether this profile's request switches optional reasoning off explicitly.

    Both spellings, because both are requests that say the same thing: OpenAI's
    Responses API takes a ``reasoning`` object carrying an effort, and xAI's
    compatibility surface takes the effort as a bare string. A profile that
    sends neither answers ``False`` here — which is not "reasoning is on", it is
    "this function is not the thing that can say".
    """
    reasoning = settings.get("reasoning")
    effort = reasoning.get("effort") if isinstance(reasoning, Mapping) else None
    return REASONING_DISABLED_EFFORT in (effort, settings.get("reasoning_effort"))


def _cmd_preflight(args: argparse.Namespace) -> int:
    # Before the credential is even looked for. A preflight that reported a
    # missing credential for an invocation it was going to refuse anyway would
    # send an operator hunting for the wrong problem.
    stop_after = parse_stop_after(args.stop_after)
    choice = provider_choice(args.adapter)
    if not args.model:
        raise RunManifestError(
            f"--adapter {choice.name} requires --model, in a preflight as in a "
            "run: what is being checked is one named model's plan"
        )
    if args.setting:
        raise RunManifestError(
            f"--adapter {choice.name} takes no --setting, in a preflight as in "
            "a run: every result-affecting setting is pinned by this build"
        )
    model = choice.check_model(args.model)
    # Environment first: a variable that would send the request somewhere else
    # invalidates everything below it.
    choice.check_environment()
    credential_present = bool(os.environ.get(choice.api_key_variable, "").strip())
    if not credential_present:
        raise RunManifestError(
            f"{choice.api_key_variable} is not set, so this plan has no credential "
            "to run with. Only its presence is checked here; its value is never "
            "read, recorded or printed"
        )

    output = _check_output_directory(Path(args.output_dir))

    suite = validate_suite(args.suite)
    scaffold = load_scaffold(args.scaffold or STANDARD_SCAFFOLD)
    # The scaffold has to be able to express the protocol the loop speaks, or the
    # run would fail identically on all 36 episodes.
    check_scaffold_contract(scaffold)

    controls = _cost_controls(args)
    identity = choice.identity(model)
    # The request shape is per-model, so the plan is checked under the profile
    # this model would actually be sent rather than under a shared default. The
    # profile's facts reach the report through ``settings``, which is what the
    # configuration identity actually hashes — so the scope statement and the
    # hashed identity cannot describe two different requests.
    settings = choice.settings(model)
    limits = RunLimits(
        max_turns=args.max_turns,
        max_messages=args.max_messages,
        episode_timeout_seconds=args.timeout,
    )
    # Built, not written. This is the same function the run uses, so the plan a
    # reviewer approves and the run that happens can be compared by identity —
    # and the episode ceiling is enforced against the plan here, before anything.
    manifest = build_run_manifest(
        suite=suite,
        scaffold=scaffold,
        provider=identity.provider,
        model=identity.model,
        implementation=identity.implementation,
        adapter_version=identity.version,
        adapter_settings=settings,
        trials=args.trials,
        limits=limits,
        cost_controls=controls,
    )

    planned = len(manifest.episode_plan)
    attempts = choice.retry_attempts()
    ceiling = choice.max_output_tokens
    max_requests = planned * args.max_turns * attempts
    bound: dict[str, Any] | None = None
    if controls.price is not None and controls.max_cost_usd is not None:
        per_request = controls.price.cost(
            input_tokens=args.planning_input_tokens, output_tokens=ceiling
        )
        upper = per_request * max_requests
        fits = upper <= controls.max_cost_usd
        bound = {
            # Named, because two models are sent different bodies and a bound is
            # a bound on one of them: a reader can tell which request shape this
            # plan was priced for without recomputing it.
            "request_profile": settings["request_profile"],
            "planning_input_tokens_per_request": args.planning_input_tokens,
            "max_output_tokens_per_request": ceiling,
            "max_attempts_per_turn": attempts,
            "max_requests": max_requests,
            "per_request_upper_bound_usd": usd_text(per_request),
            "upper_bound_usd": usd_text(upper),
            "max_cost_usd": usd_text(controls.max_cost_usd),
            "fits_within_cap": fits,
        }
        if not fits:
            raise RunManifestError(
                f"this plan's conservative cost upper bound is USD {usd_text(upper)} "
                f"and the authorised cap is USD {usd_text(controls.max_cost_usd)}. "
                f"The bound assumes {max_requests} request(s) — {planned} episode(s) "
                f"x {args.max_turns} turn(s) x {attempts} attempt(s) — each of "
                f"{args.planning_input_tokens} input and {ceiling} output "
                "token(s). Reduce the plan, lower the planning size if it is "
                "unrealistic for this suite, or authorise more"
            )

    payload: dict[str, Any] = {
        "ok": True,
        "command": "preflight",
        "configuration_id": manifest.configuration_id,
        "track": manifest.run_track,
        "status": manifest.run_status,
        "scope_statement": manifest.run_scope,
        "adapter": identity.as_dict(),
        "settings": settings,
        "credentials": {
            "variable": choice.api_key_variable,
            "present": credential_present,
            "note": CREDENTIAL_NOTE,
        },
        "output_directory": output,
        "suite": {
            "suite_id": suite.manifest.suite_id,
            "benchmark_version": suite.manifest.benchmark_version,
            "suite_content_digest": suite.content_digest,
        },
        "scaffold": {
            "scaffold_id": scaffold.scaffold_id,
            "scaffold_version": scaffold.scaffold_version,
            "content_digest": scaffold.content_digest,
        },
        "plan": {
            "planned_episodes": planned,
            "trials": manifest.trials,
            "variants": planned // manifest.trials,
            "limits": limits.as_dict(),
        },
        "cost_controls": controls.as_dict(),
        # Reported next to the plan, never folded into it: the plan above is the
        # 36 episodes this configuration is, whatever a stage asks for, and the
        # cost bound is computed over the whole plan and not over one stage.
        "invocation_staging": {
            "stop_after": stop_after,
            "planned_episodes": planned,
            "note": INVOCATION_STAGING_NOTE,
        },
        "cost_bound": bound,
        # This provider's own policy, never a build-wide default. The block is
        # what an operator audits the plan's money against — its version, its
        # published source, the date this repository read it and the digest a run
        # records — so a default would cite one vendor's page as the provenance
        # of another vendor's rates, and would contradict the ``cost_bound``
        # printed beside it, which is computed from this provider's rate.
        "pricing_policy": pricing_policy_payload(choice.provider),
        "scope": _preflight_scope(choice, settings, scaffold),
        "note": PREFLIGHT_NOTE,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=False))
        return 0
    print(f"{payload['suite']['suite_id']}: preflight OK")
    for label, value in (
        ("configuration id", payload["configuration_id"]),
        ("adapter", f"{identity.provider}/{identity.model}"),
        ("suite digest", payload["suite"]["suite_content_digest"]),
        ("scaffold digest", payload["scaffold"]["content_digest"]),
        ("planned episodes", str(planned)),
        ("trials", str(manifest.trials)),
        ("staging", _preflight_staging_line(payload)),
        # The variable *this plan's* provider reads, taken from the same choice
        # everything else on this report is derived from. It was a module-level
        # constant, so a preflight of an OpenAI, xAI or Mistral plan printed the
        # Anthropic variable while its JSON reported the right one — and the
        # human line is the one an operator acts on.
        (
            "credential",
            f"{choice.api_key_variable} present (value never read)",
        ),
        ("output directory", output["path"]),
        ("cost cap", _preflight_cost_line(payload)),
        ("cost upper bound", _preflight_bound_line(payload)),
        ("extended thinking", str(payload["scope"]["optional_extended_thinking"])),
        ("provider tools", str(payload["scope"]["external_provider_tools"])),
    ):
        print(f"  {label:<18} {value}")
    print(f"  note               {PREFLIGHT_NOTE}")
    print(f"  credential         {CREDENTIAL_NOTE}")
    return 0


def _preflight_cost_line(payload: dict[str, Any]) -> str:
    cap = payload["cost_controls"]["max_cost_usd"]
    episodes = payload["cost_controls"]["max_episodes"]
    return (
        f"USD {cap if cap is not None else '-'}; "
        f"episode ceiling {episodes if episodes is not None else '-'}"
    )


def _preflight_staging_line(payload: dict[str, Any]) -> str:
    """A requested stage, said as what it is: a bound on one invocation."""
    staging = payload["invocation_staging"]
    if staging["stop_after"] is None:
        return (
            "none requested; one invocation would execute the whole plan of "
            f"{staging['planned_episodes']} episode(s)"
        )
    return (
        f"--stop-after {staging['stop_after']} requested: at most "
        f"{staging['stop_after']} of the {staging['planned_episodes']} planned "
        "episode(s) in one invocation. Not part of run identity, and this plan is "
        "the whole plan"
    )


def _preflight_bound_line(payload: dict[str, Any]) -> str:
    bound = payload["cost_bound"]
    if bound is None:
        return "not computed (this plan has no cost cap to check against)"
    return (
        f"USD {bound['upper_bound_usd']} over at most {bound['max_requests']} "
        f"request(s) of {bound['planning_input_tokens_per_request']} input / "
        f"{bound['max_output_tokens_per_request']} output token(s)"
    )


def _stage_execution_instruction(stage: str) -> str:
    """What one cell's operator actually types to execute this stage.

    Two numbers, and the whole point of this function is that they are different
    numbers with different jobs:

    ``--max-episodes`` is the per-model manifest cap. It is
    :data:`~boundarybench.providermatrix.EPISODES_PER_MODEL` at every stage,
    because it is the run's own immutable authority rather than a stage's — a
    run capped lower would be a different configuration identity, and resuming
    it under this plan would be refused.

    ``--stop-after`` bounds the *new* episodes one invocation executes, so the
    number beside it is the stage's delta against what the stage before it
    already made durable. Handing it the cumulative target instead re-runs the
    prefix: 1 for the smoke stage and then 12 for "a trial block of twelve"
    executes thirteen episodes against a twelve-episode block. A resumed run
    still computes its own pending suffix from the ledger, so the delta bounds
    the invocation and never restarts the plan.

    A manual stage gets no command at all, because there is nothing to run.
    """
    if stage_is_manual(stage):
        return (
            f"— for the {stage} stage, nothing: it is a manual gate. A person "
            "reads the evidence the stage before it made durable and decides "
            "whether the run continues; no episodes are executed and the "
            "cumulative target per model does not move from "
            f"{stage_cumulative_episode_target(stage)}"
        )
    new_episodes = stage_new_episodes(stage)
    if not new_episodes:
        return (
            f"— for the {stage} stage, no provider run at all: it contacts "
            "nothing and executes no episodes"
        )
    return (
        f"--max-episodes {EPISODES_PER_MODEL} (the per-model run cap, the same at "
        f"every stage) and --stop-after {new_episodes} (the new episodes this "
        f"stage executes, which takes each model's run to a cumulative "
        f"{stage_cumulative_episode_target(stage)} of {EPISODES_PER_MODEL})"
    )


def _cmd_provider_matrix(args: argparse.Namespace) -> int:
    """Plan the five-model spike offline, or refuse it. Contacts nothing.

    Every cell's configuration identity is computed the same way ``preflight``
    computes one — from the suite, the scaffold, the provider's own settings and
    the run limits — so the identity an operator is asked to authorise is the
    identity the run would execute under, and it is discoverable before a single
    request is paid for.

    ``--execute`` is refused unless an authorisation is supplied, and this build
    ships no way to supply one: an approval binds an exact provider, model,
    configuration identity, episode cap, cost allocation and output directory,
    and it is the operator's to make against a plan they have read. The refusal
    happens before a credential is read or a client is built.
    """
    stage = args.stage
    if stage not in STAGE_ORDER:
        raise MatrixError(
            f"{stage!r} is not a stage this plan defines; the approved order is "
            f"{list(STAGE_ORDER)}"
        )
    # Both of the fixed methodology's own numbers, before the suite is read and
    # before anything is priced: an invocation that authorises a different
    # experiment costs an error message rather than a plan that reports this
    # one's arithmetic under the operator's numbers.
    episode_ceiling = check_global_episode_ceiling(args.max_episodes)
    trials = check_matrix_trials(args.trials)
    try:
        cap = parse_cost_cap(args.max_cost_usd)
    except BudgetError as exc:
        raise MatrixError(str(exc)) from exc
    if cap <= 0:
        raise MatrixError(
            f"the global cap must be a positive amount; got {args.max_cost_usd!r}. "
            "A zero or negative cap authorises nothing, and a plan under one "
            "would describe an experiment that cannot run"
        )

    # Before anything is planned: an environment that would redirect any one of
    # these providers invalidates the whole matrix, because the plan's claim is
    # about where all five would go.
    for entry in MATRIX_MODELS:
        provider_choice(entry.provider).check_environment()

    suite = validate_suite(args.suite)
    scaffold = load_scaffold(args.scaffold or STANDARD_SCAFFOLD)
    check_scaffold_contract(scaffold)
    limits = RunLimits(
        max_turns=args.max_turns,
        max_messages=args.max_messages,
        episode_timeout_seconds=args.timeout,
    )

    plan = build_matrix_plan(
        output_root=Path(args.output_root),
        total_cost_usd=cap,
        allocations=proportional_allocations(cap),
    )

    payload = plan.as_dict()
    payload["ok"] = True
    payload["command"] = "provider-matrix"
    payload["stage"] = stage
    # Named apart wherever they are reported. The target is where a model's run
    # stands once this stage is complete; the delta is what this stage's
    # invocation executes, and it is the only one of the two that belongs beside
    # `--stop-after`.
    payload["stage_cumulative_episode_target_per_model"] = (
        stage_cumulative_episode_target(stage)
    )
    payload["stage_new_episodes_per_model"] = stage_new_episodes(stage)
    payload["stage_is_manual"] = stage_is_manual(stage)
    payload["global_episode_ceiling"] = episode_ceiling
    payload["credentials_ready"] = all(
        entry["credential"]["present"] for entry in payload["models"]
    )
    payload["note"] = PREFLIGHT_NOTE

    # One configuration identity per cell, computed offline from the same
    # builder a run uses. Attached to the cell it belongs to rather than to a
    # separate list, so an approval cannot be read against the wrong row.
    for cell, planned in zip(payload["models"], plan.models, strict=True):
        choice = provider_choice(planned.provider)
        identity = choice.identity(planned.model)
        manifest = build_run_manifest(
            suite=suite,
            scaffold=scaffold,
            provider=identity.provider,
            model=identity.model,
            implementation=identity.implementation,
            adapter_version=identity.version,
            adapter_settings=choice.settings(planned.model),
            trials=trials,
            limits=limits,
            cost_controls=CostControls(
                max_cost_usd=planned.cost_allocation_usd,
                # The per-model manifest cap, and it is this number whatever
                # stage is running and whatever the global ceiling authorises.
                max_episodes=planned.episode_cap,
                price=price_for(provider=planned.provider, model=planned.model),
            ),
        )
        cell["configuration_id"] = manifest.configuration_id
        cell["planned_episodes"] = len(manifest.episode_plan)

    if args.execute:
        # Default-deny, and refused here — after the plan is computed, so the
        # operator is told exactly what they would need to approve, and before
        # any credential is read or client built, so refusing costs nothing.
        raise MatrixError(
            "live execution of the provider matrix is not authorised. This build "
            "enables no live traffic by default and ships no pre-approved "
            "authorisation: an approval binds one exact provider, model, "
            "configuration identity, episode cap, cost allocation and output "
            "directory, and it is the operator's to make against a plan they "
            "have read. Audit the identities this command reports, then run each "
            "cell through `run-suite` with its own authorised configuration, its "
            f"own --max-cost-usd, and {_stage_execution_instruction(stage)}"
        )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=False))
        return 0
    print(
        f"provider-matrix: {payload['total_episodes']} episode(s) planned, stage {stage}"
    )
    for cell in payload["models"]:
        print(
            f"  {cell['provider']}/{cell['model']}: {cell['episode_cap']} episode(s), "
            f"USD {cell['cost_allocation_usd']}, {cell['configuration_id']}"
        )
    # Both stage numbers, in the form most operators read. Printed as one line
    # so neither can be quoted without the other: the target is what the run
    # reaches, the delta is what this invocation would execute.
    print(
        f"  stage {stage}: cumulative target "
        f"{payload['stage_cumulative_episode_target_per_model']} episode(s) per "
        f"model, {payload['stage_new_episodes_per_model']} new episode(s) this "
        f"stage, per-model run cap {payload['per_model_run_episode_cap']}"
    )
    print(
        f"  global episode ceiling: {payload['global_episode_ceiling']} "
        f"({payload['trials']} trials, fixed)"
    )
    print(f"  credentials ready: {payload['credentials_ready']}")
    print(f"  {METHODOLOGY_NOTE}")
    return 0


_COMMANDS = {
    "validate": _cmd_validate,
    "compile": _cmd_compile,
    "check-solvers": _cmd_check_solvers,
    "verify-manifest": _cmd_verify_manifest,
    "validate-suite": _cmd_validate_suite,
    "check-suite": _cmd_check_suite,
    "run-suite": _cmd_run_suite,
    "preflight": _cmd_preflight,
    "provider-matrix": _cmd_provider_matrix,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except (SchemaError, CompileError) as exc:
        # These carry no path of their own, so the card being read is prefixed.
        return _fail(f"{getattr(args, 'card', args.command)}: {exc}")
    except (
        ManifestError,
        SuiteError,
        ScaffoldError,
        RunManifestError,
        LedgerError,
        RunnerError,
        AdapterError,
        MatrixError,
    ) as exc:
        # These already name the artefact that failed.
        return _fail(str(exc))
    except JsonSafetyError as exc:
        # The backstop for representability. Every artefact reader restates this
        # as its own domain error where it happens; reaching here means a walk
        # this build has not wrapped yet — still an input canonical UTF-8 JSON
        # cannot carry, and still not something an operator should meet as a
        # traceback.
        return _fail(f"{args.command}: this input cannot be persisted: {exc}")
    except OSError as exc:
        # The backstop. Every filesystem path a run touches is wrapped in a
        # named domain error where it happens; this catches the ones that are
        # not yet, so an unwritable device or a refused path can never reach the
        # operator as a traceback.
        return _fail(f"{args.command}: the filesystem refused this run: {exc}")
    except RecursionError as exc:
        # The same backstop for depth. Each artefact reader names its own
        # over-nested input, so reaching here means a recursive walk this build
        # has not wrapped — still an input that is too deeply nested to process,
        # and still not something an operator should meet as a traceback.
        return _fail(
            f"{args.command}: an input is nested too deeply to process "
            f"({type(exc).__name__}); excessive nesting is refused rather than "
            "parsed"
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
