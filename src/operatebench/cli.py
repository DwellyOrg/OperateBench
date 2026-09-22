"""Offline CLI for statically registered OperateBench operation packs.

The generic commands dispatch only to reviewed Python objects in the immutable
built-in registry. They do not discover entry points, import a module named by a
spec, or run a scaffold. Legacy Maintenance commands remain aliases over the
same adapter and retain their established result categories. Malformed command
lines use the distinct usage category.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Never

from boundarybench.jsonsafe import JsonSafetyError
from operatebench.core.errors import OperateBenchError
from operatebench.domains.lettings.maintenance.pack import (
    MAINTENANCE_PACK_ID,
    SCOPE_NOTE,
)
from operatebench.sdk.api import (
    CheckRequest,
    CommandResult,
    OperationPack,
    ReplayRequest,
    RunRequest,
    ValidateRequest,
)
from operatebench.sdk.builtins import BUILTIN_PACKS
from operatebench.sdk.company_scaffold import scaffold_company_operation
from operatebench.sdk.registry import OperationPackRegistry
from operatebench.sdk.scaffold import scaffold_operation
from operatebench.version import OPERATEBENCH_VERSION

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONTRACT_FAILED = 2
EXIT_USAGE = 64

PROG = "operatebench"


class _UsageError(ValueError):
    """A malformed invocation that must not impersonate a contract result."""


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        raise _UsageError(message)


def _add_run_arguments(parser: argparse.ArgumentParser, *, with_pack: bool) -> None:
    if with_pack:
        parser.add_argument(
            "--pack",
            required=True,
            help="canonical id or declared alias of a statically registered pack",
        )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-self-check",
        action="store_true",
        help="skip the pack's determinism self-check",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")


def _add_check_arguments(parser: argparse.ArgumentParser, *, with_pack: bool) -> None:
    if with_pack:
        parser.add_argument(
            "--pack",
            required=True,
            help="canonical id or declared alias of a statically registered pack",
        )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument(
        "--agent",
        action="append",
        dest="agents",
        help="restrict the gate to a named agent (repeatable)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")


def _build_parser() -> argparse.ArgumentParser:
    maintenance_agents = BUILTIN_PACKS.resolve(MAINTENANCE_PACK_ID).metadata.agent_ids
    parser = _ArgumentParser(
        prog=PROG,
        description=(
            "Validate, run and replay synthetic longitudinal operation packs "
            f"(engine {OPERATEBENCH_VERSION})."
        ),
        epilog=SCOPE_NOTE,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate",
        help="static spec validation; executes no scenario",
        description=(
            "Static validation only. The explicitly selected pack loads its strict "
            "schema and identity. This does not prove terminal reachability."
        ),
    )
    validate.add_argument("spec", type=Path)
    validate.add_argument(
        "--pack",
        help=("registered pack id; omitted only for legacy Maintenance compatibility"),
    )
    validate.add_argument("--json", action="store_true", help="emit JSON")

    run_cmd = subparsers.add_parser(
        "run", help="execute one scenario through a statically registered pack"
    )
    _add_run_arguments(run_cmd, with_pack=True)

    legacy_run = subparsers.add_parser(
        "run-maintenance",
        help="legacy alias for `run --pack maintenance`",
        description=(
            "Execute the existing Maintenance vertical. Shipped agents: "
            f"{', '.join(maintenance_agents)}"
        ),
    )
    _add_run_arguments(legacy_run, with_pack=False)

    replay = subparsers.add_parser(
        "replay", help="reproduce a pack-owned run and compare its complete result"
    )
    replay.add_argument("--spec", type=Path, required=True)
    replay.add_argument("--run", type=Path, required=True, dest="run_path")
    replay.add_argument(
        "--pack",
        help=("registered pack id; omitted only for legacy Maintenance compatibility"),
    )
    replay.add_argument("--json", action="store_true", help="emit JSON")

    check = subparsers.add_parser(
        "check", help="run a pack's reference and separate negative-control gate"
    )
    _add_check_arguments(check, with_pack=True)

    legacy_check = subparsers.add_parser(
        "check-maintenance", help="legacy alias for `check --pack maintenance`"
    )
    _add_check_arguments(legacy_check, with_pack=False)

    list_packs = subparsers.add_parser(
        "list-packs", help="list immutable built-in operation-pack metadata"
    )
    list_packs.add_argument("--json", action="store_true", help="emit JSON")

    init = subparsers.add_parser(
        "init-operation",
        help="create an unregistered, non-runnable incubator scaffold",
    )
    init.add_argument("--pack-id", required=True)
    init.add_argument("--operation-id", required=True)
    init.add_argument("--owner-id", required=True)
    init.add_argument("--owner-display-name", required=True)
    init.add_argument(
        "--contribution-kind", required=True, choices=("maintainer", "partner")
    )
    init.add_argument("--destination", type=Path, required=True)
    init.add_argument("--json", action="store_true", help="emit JSON")

    company = subparsers.add_parser(
        "init-company-operation",
        help="create a complete owner-scoped, unregistered partner contribution",
    )
    company.add_argument("--repository-root", type=Path, required=True)
    company.add_argument("--owner-id", required=True)
    company.add_argument("--owner-display-name", required=True)
    company.add_argument("--operation-name", required=True)
    company.add_argument("--pack-id", required=True)
    company.add_argument("--operation-id", required=True)
    company.add_argument("--json", action="store_true", help="emit JSON")
    return parser


def _fail(message: str) -> int:
    print(f"{PROG}: {message}", file=sys.stderr)
    return EXIT_ERROR


def _usage_fail(parser: argparse.ArgumentParser, message: str) -> int:
    parser.print_usage(sys.stderr)
    print(f"{PROG}: error: {message}", file=sys.stderr)
    return EXIT_USAGE


def _emit(result: CommandResult, *, as_json: bool) -> int:
    if as_json:
        print(json.dumps(result.payload, indent=2, sort_keys=False))
    else:
        for line in result.text_lines:
            print(line)
    return EXIT_OK if result.contract_passed else EXIT_CONTRACT_FAILED


def _selected(
    registry: OperationPackRegistry, name: str | None, *, legacy_default: bool
) -> OperationPack:
    if name is None:
        if not legacy_default:
            raise _UsageError("a generic operation-pack command requires --pack")
        name = MAINTENANCE_PACK_ID
    return registry.resolve(name)


def _cmd_validate(args: argparse.Namespace, registry: OperationPackRegistry) -> int:
    pack = _selected(registry, args.pack, legacy_default=True)
    return _emit(pack.validate(ValidateRequest(spec_path=args.spec)), as_json=args.json)


def _run_with(
    args: argparse.Namespace,
    registry: OperationPackRegistry,
    *,
    pack_name: str | None,
) -> int:
    pack = _selected(registry, pack_name, legacy_default=False)
    return _emit(
        pack.run(
            RunRequest(
                spec_path=args.spec,
                scenario_id=args.scenario,
                agent_id=args.agent,
                output_path=args.output,
                self_check=not args.no_self_check,
            )
        ),
        as_json=args.json,
    )


def _cmd_run(args: argparse.Namespace, registry: OperationPackRegistry) -> int:
    return _run_with(args, registry, pack_name=args.pack)


def _cmd_run_maintenance(
    args: argparse.Namespace, registry: OperationPackRegistry
) -> int:
    return _run_with(args, registry, pack_name=MAINTENANCE_PACK_ID)


def _cmd_replay(args: argparse.Namespace, registry: OperationPackRegistry) -> int:
    pack = _selected(registry, args.pack, legacy_default=True)
    return _emit(
        pack.replay(ReplayRequest(spec_path=args.spec, run_path=args.run_path)),
        as_json=args.json,
    )


def _check_with(
    args: argparse.Namespace,
    registry: OperationPackRegistry,
    *,
    pack_name: str | None,
) -> int:
    pack = _selected(registry, pack_name, legacy_default=False)
    selected_agents = None if args.agents is None else tuple(args.agents)
    return _emit(
        pack.check(CheckRequest(spec_path=args.spec, agent_ids=selected_agents)),
        as_json=args.json,
    )


def _cmd_check(args: argparse.Namespace, registry: OperationPackRegistry) -> int:
    return _check_with(args, registry, pack_name=args.pack)


def _cmd_check_maintenance(
    args: argparse.Namespace, registry: OperationPackRegistry
) -> int:
    return _check_with(args, registry, pack_name=MAINTENANCE_PACK_ID)


def _cmd_list_packs(args: argparse.Namespace, registry: OperationPackRegistry) -> int:
    metadata = registry.metadata()
    if args.json:
        print(json.dumps([item.as_dict() for item in metadata], indent=2))
    else:
        for item in metadata:
            print(
                f"{item.pack_id}  {item.status}  {item.operation_type}  "
                f"operation_id={item.operation_id}  "
                f"owner={item.owner_id} ({item.owner_display_name}; "
                f"{item.contribution_kind})  evidence_eligible={item.evidence_eligible}"
            )
            print(f"    {item.display_name}")
    return EXIT_OK


def _cmd_init_operation(args: argparse.Namespace, registry: OperationPackRegistry) -> int:
    # Resolve only to detect a collision. Unknown is the expected case and is
    # tested by membership instead of caught as an exception.
    known_names = {
        name
        for item in registry.metadata()
        for name in (
            item.pack_id,
            item.operation_type,
            *item.aliases,
            item.operation_id,
        )
    }
    collision = next(
        (
            identity
            for identity in (args.pack_id, args.operation_id)
            if identity in known_names
        ),
        None,
    )
    if collision is not None:
        raise OperateBenchError(
            f"operation identity {collision!r} is already registered; a scaffold "
            "cannot replace it"
        )
    paths = scaffold_operation(
        pack_id=args.pack_id,
        operation_id=args.operation_id,
        owner_id=args.owner_id,
        owner_display_name=args.owner_display_name,
        contribution_kind=args.contribution_kind,
        destination=args.destination,
    )
    payload = {
        "pack_id": args.pack_id,
        "operation_id": args.operation_id,
        "owner_id": args.owner_id,
        "owner_display_name": args.owner_display_name,
        "contribution_kind": args.contribution_kind,
        "destination": str(args.destination),
        "files": [str(path) for path in paths],
        "registered": False,
        "runnable": False,
        "evidence_eligible": False,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"created unregistered incubator scaffold at {args.destination}")
        for path in paths:
            print(f"  {path}")
        print("not registered; not runnable; not benchmark evidence")
    return EXIT_OK


def _cmd_init_company_operation(
    args: argparse.Namespace, registry: OperationPackRegistry
) -> int:
    del registry
    result = scaffold_company_operation(
        repository_root=args.repository_root,
        owner_id=args.owner_id,
        owner_display_name=args.owner_display_name,
        operation_name=args.operation_name,
        pack_id=args.pack_id,
        operation_id=args.operation_id,
    )
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print("created complete guided partner contribution transaction")
        for path in result.created_paths:
            print(f"  {path}")
        if result.reused_owner_declaration is not None:
            print(f"reused exactly: {result.reused_owner_declaration}")
        print("registered=false; runnable=false; evidence_eligible=false")
    return EXIT_OK


CommandHandler = Callable[[argparse.Namespace, OperationPackRegistry], int]

_COMMANDS: dict[str, CommandHandler] = {
    "validate": _cmd_validate,
    "run": _cmd_run,
    "run-maintenance": _cmd_run_maintenance,
    "replay": _cmd_replay,
    "check": _cmd_check,
    "check-maintenance": _cmd_check_maintenance,
    "list-packs": _cmd_list_packs,
    "init-operation": _cmd_init_operation,
    "init-company-operation": _cmd_init_company_operation,
}


def main(
    argv: Sequence[str] | None = None,
    *,
    registry: OperationPackRegistry = BUILTIN_PACKS,
) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except _UsageError as exc:
        return _usage_fail(parser, str(exc))
    try:
        return _COMMANDS[args.command](args, registry)
    except _UsageError as exc:
        return _usage_fail(parser, str(exc))
    except OperateBenchError as exc:
        return _fail(str(exc))
    except JsonSafetyError as exc:
        return _fail(f"{args.command}: this input cannot be persisted: {exc}")
    except OSError as exc:
        return _fail(f"{args.command}: the filesystem refused this command: {exc}")
    except RecursionError as exc:
        return _fail(
            f"{args.command}: an input is nested too deeply to process "
            f"({type(exc).__name__}); excessive nesting is refused rather than parsed"
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
