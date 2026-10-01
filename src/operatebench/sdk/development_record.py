"""Bounded strict JSON and exclusive creation for development-only records.

A record checksum is corruption detection, not a signature. Semantic acceptance
requires development_runtime.replay with independently supplied trusted factories.
"""

from __future__ import annotations

import json
import math
import os
import stat
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Never, cast

from operatebench._write_once import _write_once_bytes
from operatebench.agents.model import ModelBoundaryError
from operatebench.agents.playback import (
    DECISION_FIELDS,
    PlaybackError,
    decision_record,
    outcome_from_record,
)
from operatebench.agents.transport import content_digest
from operatebench.core.instance import operation_instance_id_problem
from operatebench.sdk.development_runtime import (
    RUN_FORMAT,
    SCOPE,
    DevelopmentRuntimeError,
    _evaluation_fields,
)

MAX_RUN_BYTES = 16 * 1024 * 1024
MAX_DEPTH = 64
MAX_NODES = 500_000
MAX_DECISIONS = 4096


def _json_tree(value: Any) -> None:
    count = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal count
        count += 1
        if count > MAX_NODES or depth > MAX_DEPTH:
            raise DevelopmentRuntimeError("JSON structure exceeds development limit")
        if type(item) is dict:
            if not all(type(key) is str for key in item):
                raise DevelopmentRuntimeError("JSON object keys must be strings")
            for child in item.values():
                visit(child, depth + 1)
        elif type(item) is list:
            for child in item:
                visit(child, depth + 1)
        elif (
            item is None
            or type(item) in (bool, int, str)
            or (type(item) is float and math.isfinite(item))
        ):
            return
        else:
            raise DevelopmentRuntimeError("record is not strict finite JSON")

    visit(value, 0)


def canonical(value: Any) -> bytes:
    _json_tree(value)
    try:
        body = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (ValueError, TypeError, OverflowError) as exc:
        raise DevelopmentRuntimeError("invalid JSON value") from exc
    if len(body) > MAX_RUN_BYTES:
        raise DevelopmentRuntimeError("record exceeds byte limit")
    return body


def _fields(value: Any, fields: Iterable[str], name: str) -> None:
    if type(value) is not dict or set(value) != set(fields):
        raise DevelopmentRuntimeError(f"{name} fields differ from contract")


def _hex(value: Any) -> bool:
    return (
        type(value) is str and len(value) == 64 and set(value) <= set("0123456789abcdef")
    )


def validate_record(record: Any) -> dict[str, Any]:
    canonical(record)
    _fields(
        record,
        (
            "format",
            "scope",
            "binding",
            "identity",
            "agent",
            "episode",
            "decisions",
            "evaluation",
            "record_digest",
        ),
        "record",
    )
    if record["format"] != RUN_FORMAT or record["scope"] != SCOPE:
        raise DevelopmentRuntimeError("not a supported non-evidence development record")
    body = {k: v for k, v in record.items() if k != "record_digest"}
    if not _hex(record["record_digest"]) or record["record_digest"] != content_digest(
        body
    ):
        raise DevelopmentRuntimeError("record checksum mismatch")
    _fields(
        record["binding"],
        (
            "pack_id",
            "pack_version",
            "operation_type",
            "operation_id",
            "spec_digest",
            "scenario_id",
            "profile",
            "implementation_digest",
            "callbacks",
            "runtime_digest",
        ),
        "binding",
    )
    binding = record["binding"]
    for key in (
        "pack_id",
        "pack_version",
        "operation_type",
        "operation_id",
        "scenario_id",
    ):
        if type(binding[key]) is not str or not binding[key]:
            raise DevelopmentRuntimeError("binding identifiers must be nonempty strings")
    for key in ("spec_digest", "implementation_digest", "runtime_digest"):
        if not _hex(binding[key]):
            raise DevelopmentRuntimeError("invalid binding digest")
    _fields(
        binding["profile"],
        ("profile_id", "profile_version", "profile_digest", "jurisdiction"),
        "profile",
    )
    if not all(type(v) is str and v for v in binding["profile"].values()) or not _hex(
        binding["profile"]["profile_digest"]
    ):
        raise DevelopmentRuntimeError("invalid profile identity")
    if type(binding["callbacks"]) is not list or not all(
        type(v) is str and v for v in binding["callbacks"]
    ):
        raise DevelopmentRuntimeError("invalid callback identity")
    _fields(
        record["identity"],
        (
            "operation_id",
            "scenario_id",
            "spec_digest_sha256",
            "agent_id",
            "operation_instance_id",
        ),
        "identity",
    )
    identity = record["identity"]
    if not all(type(v) is str and v for v in identity.values()):
        raise DevelopmentRuntimeError("invalid episode identity")
    if operation_instance_id_problem(identity["operation_instance_id"]) is not None:
        raise DevelopmentRuntimeError("invalid operation instance identity")
    agent = record["agent"]
    if type(agent) is not dict:
        raise DevelopmentRuntimeError("agent must be an object")
    if agent.get("kind") == "fixed":
        _fields(agent, ("kind", "agent_id"), "fixed agent")
    elif agent.get("kind") == "reference_driven_mock_model":
        _fields(
            agent,
            (
                "kind",
                "agent_id",
                "reference_agent_id",
                "model",
                "max_output_tokens",
                "request_digests",
                "mock_calls",
                "protocol_version",
            ),
            "mock agent",
        )
        from operatebench.agents.model import MODEL_PROTOCOL_VERSION

        if (
            agent["model"] != "development-reference-mock-not-a-provider"
            or agent["agent_id"] != "development-reference-mock"
            or agent["protocol_version"] != MODEL_PROTOCOL_VERSION
            or type(agent["max_output_tokens"]) is not int
            or agent["max_output_tokens"] != 4096
            or type(agent["mock_calls"]) is not int
            or type(agent["request_digests"]) is not list
            or not all(_hex(d) for d in agent["request_digests"])
        ):
            raise DevelopmentRuntimeError("invalid mock model identity")
    else:
        raise DevelopmentRuntimeError("unsupported agent source")
    if type(agent["agent_id"]) is not str or not agent["agent_id"]:
        raise DevelopmentRuntimeError("agent identity must be nonempty")
    decisions = record["decisions"]
    if type(decisions) is not list or not 0 < len(decisions) <= MAX_DECISIONS:
        raise DevelopmentRuntimeError("invalid decision count")
    if agent["kind"] == "reference_driven_mock_model" and (
        len(agent["request_digests"]) != len(decisions)
        or agent["mock_calls"] != len(decisions)
        or type(agent["reference_agent_id"]) is not str
        or not agent["reference_agent_id"]
    ):
        raise DevelopmentRuntimeError(
            "mock requests must cover exactly the whole decision tape"
        )
    for decision in decisions:
        _fields(decision, DECISION_FIELDS, "decision")
        if (
            type(decision["invocation_index"]) is not int
            or decision["invocation_index"] < 0
            or type(decision["turn_index"]) is not int
            or decision["turn_index"] < 0
            or not _hex(decision["observation_digest_sha256"])
        ):
            raise DevelopmentRuntimeError("invalid decision identity")
        outcome = decision["outcome"]
        if (
            type(outcome) is not dict
            or type(outcome.get("kind")) is not str
            or outcome["kind"] not in {"ACT", "WAIT", "COMPLETE", "RETRIEVE", "MALFORMED"}
        ):
            raise DevelopmentRuntimeError("unsupported recorded outcome")
        try:
            rebuilt = decision_record(outcome_from_record(outcome))
        except (
            KeyError,
            TypeError,
            ValueError,
            PlaybackError,
            ModelBoundaryError,
        ) as exc:
            raise DevelopmentRuntimeError("malformed recorded outcome") from exc
        if canonical(rebuilt) != canonical(outcome):
            raise DevelopmentRuntimeError(
                "recorded outcome has unknown or noncanonical fields"
            )
    episode = record["episode"]
    _fields(
        episode,
        (
            "status",
            "terminal_outcome",
            "replay_final",
            "started_at",
            "ended_at",
            "simulated_minutes",
            "invocations",
            "final_state",
            "final_state_digest_sha256",
            "trajectory",
            "trajectory_digest_sha256",
            "events",
        ),
        "episode",
    )
    if (
        not all(type(episode[key]) is str for key in ("status", "started_at", "ended_at"))
        or (
            episode["terminal_outcome"] is not None
            and type(episode["terminal_outcome"]) is not str
        )
        or type(episode["replay_final"]) is not bool
        or type(episode["simulated_minutes"]) is not int
        or type(episode["invocations"]) is not int
        or type(episode["final_state"]) is not dict
        or not _hex(episode["final_state_digest_sha256"])
        or not _hex(episode["trajectory_digest_sha256"])
        or any(
            type(episode[key]) is not list
            or not all(type(item) is dict for item in episode[key])
            for key in ("trajectory", "events")
        )
    ):
        raise DevelopmentRuntimeError("invalid episode envelope")
    evaluation = record["evaluation"]
    _fields(
        evaluation,
        _evaluation_fields(evaluation, record["binding"]["pack_id"]),
        "evaluation",
    )
    if (
        type(evaluation["reliable"]) is not bool
        or type(evaluation["dimensions"]) is not dict
        or not evaluation["dimensions"]
        or not all(type(value) is bool for value in evaluation["dimensions"].values())
        or type(evaluation["findings"]) is not list
        or not all(type(finding) is dict for finding in evaluation["findings"])
        or type(evaluation["terminal_outcome"]) is not str
    ):
        raise DevelopmentRuntimeError("invalid evaluation envelope")
    # canonical() already bounds all nested JSON. State, event/trajectory
    # payloads and finding contents belong to domains, not this envelope codec.
    # Recomputed digests, episode semantics and grading remain replay's job.
    return cast(dict[str, Any], record)


def _seal_record(record: dict[str, Any]) -> dict[str, Any]:
    canonical(record)
    sealed = {**record, "record_digest": content_digest(record)}
    validate_record(sealed)
    return sealed


def write_record(path: str | Path, record: Any) -> Path:
    """Shape/checksum checked write-once; does not itself certify semantic replay."""
    validate_record(record)
    return _write_once_bytes(
        canonical(record),
        path,
        error_adapter=lambda failure: DevelopmentRuntimeError(
            f"development output: {failure.reason}"
        ),
    )


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DevelopmentRuntimeError("duplicate JSON key")
        result[key] = value
    return result


def _constant(value: str) -> Never:
    raise DevelopmentRuntimeError("non-finite JSON number")


def _check_depth(body: bytes) -> None:
    # Structural preflight before json.loads allocates recursive containers.
    # The JSON parser still owns syntax; quoted braces and escaped quotes do not
    # count as structure. The bounded raw body makes this scan linear/bounded.
    depth = 0
    quoted = False
    escaped = False
    for byte in body:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            if depth > MAX_DEPTH:
                raise DevelopmentRuntimeError("JSON depth exceeds development limit")
        elif byte in (93, 125):
            depth -= 1


def read_record(path: str | Path) -> dict[str, Any]:
    """Read a bounded regular file; refuse duplicate keys, NaN, depth and size abuse."""
    try:
        descriptor = os.open(Path(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise DevelopmentRuntimeError("record is not a regular file")
            body = handle.read(MAX_RUN_BYTES + 1)
    except OSError as exc:
        raise DevelopmentRuntimeError("cannot read development record") from exc
    if len(body) > MAX_RUN_BYTES:
        raise DevelopmentRuntimeError("record exceeds byte limit")
    _check_depth(body)
    try:
        record = json.loads(
            body.decode("utf-8"), object_pairs_hook=_object, parse_constant=_constant
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise DevelopmentRuntimeError("invalid development JSON") from exc
    return validate_record(record)
