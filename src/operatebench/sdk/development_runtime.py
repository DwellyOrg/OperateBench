"""Trusted, offline DEVELOPMENT execution; not Artifact 8 or model attestation.

Factories are Python capabilities supplied by reviewed integration code, never
module names loaded from records/YAML. Replay executes Core anew and evaluates
its authenticated outcome, not a deserialized state. Checksums detect corruption
and stale bindings, NOT hostile wholesale replacement or authorship.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from operatebench.agents.model import MODEL_PROTOCOL_VERSION, ModelAgent
from operatebench.agents.playback import (
    PlaybackError,
    RecordedModelAgent,
    RecordedOutcomeAgent,
    RecordingAgent,
    decision_record,
    tape_from_records,
)
from operatebench.agents.transport import (
    ModelRequest,
    ModelResponse,
    ToolCall,
    content_digest,
)
from operatebench.core.engine import Engine, EpisodeOutcome
from operatebench.core.instance import (
    new_operation_instance_id,
    require_operation_instance_id,
)
from operatebench.core.outcomes import Ask, Escalate
from operatebench.core.protocol import AgentObservation, OperationAgent, OperationDomain

RUN_FORMAT = "operatebench.development-run.v1"
SCOPE = "DEVELOPMENT_ONLY_NON_EVIDENCE_NON_PROVIDER_NOT_MODEL_ATTESTATION"
OUTCOME_SUPPORT = {
    "supported": ["ACT", "WAIT", "COMPLETE", "RETRIEVE"],
    "unsupported": ["ASK", "ESCALATE"],
    "human_review": "Use an explicit domain ACT review request followed by WAIT. "
    "Generic ASK and ESCALATE lowering is unsupported in this lane.",
}


class DevelopmentRuntimeError(ValueError):
    """A development record or unsupported outcome was refused."""


@dataclass(frozen=True)
class DevelopmentFactories:
    """Reviewed callbacks; implementation_paths cover domain code/resources.

    Callback source files are also hashed. List transitive domain dependencies
    under implementation_paths, not just __init__.py. No sandbox is implied:
    trusted Python callbacks can execute Python. dispatch_failures defaults empty.
    """

    pack_id: str
    pack_version: str
    operation_type: str
    load_spec: Callable[[str | Path], Any]
    build_domain: Callable[[Any, str], OperationDomain]
    build_agent: Callable[[str], OperationAgent]
    evaluate_episode: Callable[[EpisodeOutcome, Any, str], Mapping[str, Any]]
    profile_identity: Callable[[Any, str], Mapping[str, str]]
    implementation_paths: tuple[Path, ...]
    dispatch_failures: Callable[[Any, str], Iterable[str]] = lambda spec, scenario: ()


def _source_digest(paths: Iterable[Path]) -> str:
    rows = []
    for index, raw in enumerate(paths):
        path = Path(raw)
        files = (
            sorted(
                p
                for p in path.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
            )
            if path.is_dir()
            else [path]
        )
        if not files:
            raise DevelopmentRuntimeError("empty implementation source path")
        for file in files:
            name = file.relative_to(path).as_posix() if path.is_dir() else file.name
            rows.append([index, name, hashlib.sha256(file.read_bytes()).hexdigest()])
    return content_digest(rows)


def _bindings(
    factories: DevelopmentFactories, spec: Any, scenario: str
) -> dict[str, Any]:
    if type(scenario) is not str or scenario not in spec.scenarios:
        raise DevelopmentRuntimeError("unknown scenario")
    profile = dict(factories.profile_identity(spec, scenario))
    if set(profile) != {
        "profile_id",
        "profile_version",
        "profile_digest",
        "jurisdiction",
    }:
        raise DevelopmentRuntimeError("profile identity fields differ from contract")
    if not all(isinstance(value, str) and value for value in profile.values()):
        raise DevelopmentRuntimeError("profile identity values must be nonempty strings")
    callbacks: list[Callable[..., Any]] = [
        factories.load_spec,
        factories.build_domain,
        factories.build_agent,
        factories.evaluate_episode,
        factories.profile_identity,
        factories.dispatch_failures,
    ]
    callback_paths = tuple(Path(inspect.getfile(cb)) for cb in callbacks)
    root = Path(__file__).resolve().parents[1]
    return {
        "pack_id": factories.pack_id,
        "pack_version": factories.pack_version,
        "operation_type": factories.operation_type,
        "operation_id": spec.operation_id,
        "spec_digest": spec.content_digest,
        "scenario_id": scenario,
        "profile": profile,
        "implementation_digest": _source_digest(
            (*factories.implementation_paths, *callback_paths)
        ),
        "callbacks": [f"{cb.__module__}:{cb.__qualname__}" for cb in callbacks],
        "runtime_digest": _source_digest(
            (
                root / "core",
                root / "agents",
                Path(__file__),
                root / "sdk" / "development_record.py",
            )
        ),
    }


class _SupportedAgent:
    def __init__(self, inner: Any, agent_identity: Mapping[str, Any]) -> None:
        self.inner = inner
        self.agent_identity = agent_identity
        self.agent_id = inner.agent_id

    def begin_episode(self, identity: Mapping[str, str]) -> None:
        self.inner.begin_episode(identity)

    def decide(self, observation: AgentObservation) -> Any:
        public = replace(
            observation,
            policy={
                **observation.policy,
                "development_outcome_support": deepcopy(OUTCOME_SUPPORT),
                "development_agent_identity": deepcopy(self.agent_identity),
            },
        )
        decision = self.inner.decide(public)
        if isinstance(decision, (Escalate, Ask)):
            raise DevelopmentRuntimeError(
                "unsupported ASK/ESCALATE; use domain ACT review + WAIT"
            )
        return decision


def _execute(
    factories: DevelopmentFactories,
    spec: Any,
    scenario: str,
    agent: OperationAgent,
    identity: Mapping[str, str],
    metadata: Mapping[str, Any],
) -> EpisodeOutcome:
    agent_identity = {
        k: v for k, v in metadata.items() if k not in {"request_digests", "mock_calls"}
    }
    return Engine(
        factories.build_domain(spec, scenario),
        _SupportedAgent(agent, agent_identity),
        identity=identity,
        dispatch_failures=frozenset(factories.dispatch_failures(spec, scenario)),
    ).run()


def _evaluation_fields(grade: Any, pack_id: str) -> set[str]:
    """Accept paired commerce clock metadata without opening other envelopes."""
    fields = {"reliable", "dimensions", "findings", "terminal_outcome"}
    if (
        isinstance(grade, dict)
        and pack_id == "commerce.return_refund.profiles.v1"
        and {"evaluator_version", "diagnostics"} & grade.keys()
    ):
        if (
            not isinstance(grade.get("evaluator_version"), str)
            or not grade["evaluator_version"]
            or not isinstance(grade.get("diagnostics"), list)
            or not all(
                isinstance(item, dict)
                and set(item) == {"dimension", "code"}
                and item["dimension"] == "clock"
                and item["code"] == "REFUND_UNSETTLED_AT_OBSERVATION_END"
                for item in grade["diagnostics"]
            )
        ):
            raise DevelopmentRuntimeError("invalid evaluation clock metadata")
        fields |= {"evaluator_version", "diagnostics"}
    return fields


def _grade(
    factories: DevelopmentFactories, episode: EpisodeOutcome, spec: Any, scenario: str
) -> dict[str, Any]:
    grade = dict(factories.evaluate_episode(episode, spec, scenario))
    if set(grade) != _evaluation_fields(grade, factories.pack_id):
        raise DevelopmentRuntimeError("evaluator fields differ from contract")
    if (
        type(grade["reliable"]) is not bool
        or not isinstance(grade["dimensions"], dict)
        or not grade["dimensions"]
        or not all(
            isinstance(k, str) and type(v) is bool for k, v in grade["dimensions"].items()
        )
        or not isinstance(grade["findings"], list)
        or not all(isinstance(f, dict) for f in grade["findings"])
        or not isinstance(grade["terminal_outcome"], str)
    ):
        raise DevelopmentRuntimeError("invalid evaluator result")
    return grade


def run(
    factories: DevelopmentFactories,
    spec_path: str | Path,
    scenario_id: str,
    agent_id: str = "reference",
    *,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Execute once, record actual decisions, verify playback, optionally write once."""
    return _run(factories, spec_path, scenario_id, agent_id, output=output, mock=False)


class _ReferenceTransport:
    """In-process decisions encoded as tools, through an actual ModelAgent parser."""

    def __init__(self, reference: OperationAgent) -> None:
        self.reference = reference
        self.digests: list[str] = []
        self.reference.begin_episode({})

    def send(self, request: ModelRequest) -> ModelResponse:
        public = json.loads(json.dumps(request.prompt))["observation"]
        decision = self.reference.decide(AgentObservation(**public))
        payload = decision_record(decision)
        kind = payload.pop("kind")
        # Encode all lifecycle tools faithfully; the runtime guard, not a fake
        # transport failure, rejects ASK/ESCALATE before Core lowering.
        if kind == "ASK":
            payload["wait"].pop("kind")
        if kind not in {"ACT", "WAIT", "COMPLETE", "RETRIEVE", "ASK", "ESCALATE"}:
            raise DevelopmentRuntimeError("mock reference produced unsupported outcome")
        self.digests.append(request.request_digest_sha256)
        return ModelResponse(
            model=request.model,
            stop_reason="tool_use",
            tool_calls=(ToolCall(kind.lower(), payload),),
            text="",
            output_tokens=1,
        )


def run_mock_model(
    factories: DevelopmentFactories,
    spec_path: str | Path,
    scenario_id: str,
    reference_agent_id: str = "reference",
    *,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Non-provider/non-official demo, NOT evidence of model ability or authorship.

    No injected transport/provider configuration is accepted. A trusted fixed
    reference sees only fresh JSON-round-tripped model-visible requests and emits
    structured tools, parsed by the real ModelAgent. Max 4096 mock calls.
    """
    return _run(
        factories, spec_path, scenario_id, reference_agent_id, output=output, mock=True
    )


def _run(
    factories: DevelopmentFactories,
    spec_path: str | Path,
    scenario_id: str,
    agent_id: str,
    *,
    output: str | Path | None,
    mock: bool,
) -> dict[str, Any]:
    from operatebench.sdk.development_record import _seal_record, canonical

    spec = factories.load_spec(spec_path)
    binding = _bindings(factories, spec, scenario_id)
    agent: Any = factories.build_agent(agent_id)
    if agent.agent_id != agent_id:
        raise DevelopmentRuntimeError("agent factory returned a different identity")
    agent_metadata: dict[str, Any] = {"kind": "fixed", "agent_id": agent_id}
    transport = None
    if mock:
        transport = _ReferenceTransport(agent)
        agent = ModelAgent(
            transport,
            model="development-reference-mock-not-a-provider",
            agent_id="development-reference-mock",
            max_transport_calls=4096,
        )
        agent_metadata = {
            "kind": "reference_driven_mock_model",
            "agent_id": agent.agent_id,
            "reference_agent_id": agent_id,
            "model": agent.model,
            "max_output_tokens": agent.max_output_tokens,
            "protocol_version": MODEL_PROTOCOL_VERSION,
        }
        agent_id = agent.agent_id
    identity = {
        "operation_id": spec.operation_id,
        "scenario_id": scenario_id,
        "spec_digest_sha256": spec.content_digest,
        "agent_id": agent_id,
        "operation_instance_id": new_operation_instance_id(),
    }
    recorder = RecordingAgent(agent)
    episode = _execute(factories, spec, scenario_id, recorder, identity, agent_metadata)
    if transport is not None:
        agent_metadata.update(
            request_digests=transport.digests, mock_calls=len(transport.digests)
        )
    if canonical(binding) != canonical(_bindings(factories, spec, scenario_id)):
        raise DevelopmentRuntimeError("trusted implementation changed during execution")
    record = {
        "format": RUN_FORMAT,
        "scope": SCOPE,
        "binding": binding,
        "identity": identity,
        "agent": agent_metadata,
        "episode": episode.as_dict(),
        "decisions": recorder.tape().as_list(),
        "evaluation": _grade(factories, episode, spec, scenario_id),
    }
    record = _seal_record(record)
    replay(factories, spec_path, record)
    if output is not None:
        from operatebench.sdk.development_record import write_record

        write_record(output, record)
    return record


def replay(
    factories: DevelopmentFactories, spec_path: str | Path, record: Mapping[str, Any]
) -> dict[str, Any]:
    """Reconstruct without calling build_agent, an original agent, or a provider."""
    from operatebench.sdk.development_record import canonical, validate_record

    record = validate_record(record)
    spec = factories.load_spec(spec_path)
    scenario = record["binding"]["scenario_id"]
    if canonical(record["binding"]) != canonical(_bindings(factories, spec, scenario)):
        raise DevelopmentRuntimeError(
            "trusted spec/profile/implementation/runtime binding mismatch"
        )
    identity = record["identity"]
    require_operation_instance_id(identity["operation_instance_id"], "development run")
    expected_identity = {
        "operation_id": spec.operation_id,
        "scenario_id": scenario,
        "spec_digest_sha256": spec.content_digest,
        "agent_id": record["agent"]["agent_id"],
        "operation_instance_id": identity["operation_instance_id"],
    }
    if identity != expected_identity:
        raise DevelopmentRuntimeError("episode identity mismatch")
    tape = tape_from_records(record["decisions"])
    metadata = record["agent"]
    playback: RecordedModelAgent | RecordedOutcomeAgent
    if metadata["kind"] == "reference_driven_mock_model":
        playback = RecordedModelAgent(
            tape,
            agent_id=identity["agent_id"],
            model=metadata["model"],
            max_output_tokens=metadata["max_output_tokens"],
            request_digests=metadata["request_digests"],
        )
    else:
        playback = RecordedOutcomeAgent(tape, agent_id=identity["agent_id"])
    try:
        episode = _execute(factories, spec, scenario, playback, identity, metadata)
        playback.check_exhausted()
    except PlaybackError as exc:
        raise DevelopmentRuntimeError(
            f"decision playback refused: {type(exc).__name__}"
        ) from exc
    if canonical(episode.as_dict()) != canonical(record["episode"]):
        raise DevelopmentRuntimeError("recomputed full episode differs")
    grade = _grade(factories, episode, spec, scenario)
    if canonical(grade) != canonical(record["evaluation"]):
        raise DevelopmentRuntimeError("recomputed grade differs")
    calls = playback.provider_calls if isinstance(playback, RecordedModelAgent) else 0
    if calls != 0:
        raise DevelopmentRuntimeError("playback reached a provider")
    return {
        "consistent": True,
        "provider_calls": calls,
        "decisions_consumed": playback.cursor,
        "evaluation": grade,
    }
