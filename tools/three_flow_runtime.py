"""Three-flow integration candidate, not Artifact 8 or paid launch authority.

Trusted Python transport injection only. No credential loader, CLI, prices or
implicit output/call/time cap. Provider settings are captured, NOT approved by
this module. Offline SDK mocks are explicitly non-model-performance evidence.
Replay needs the independently retained Trial; the record cannot select a new
flow, scenario, model, campaign or trial by coherently resealing itself.
"""

from __future__ import annotations

import copy
import hashlib
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from operatebench._write_once import _write_once_bytes
from operatebench.agents.evidence import normalized_response_digest
from operatebench.agents.model import ModelAgent
from operatebench.agents.playback import (
    RecordedModelAgent,
    RecordingAgent,
    tape_from_records,
)
from operatebench.agents.transport import (
    ModelRequest,
    ModelResponse,
    ProviderFailure,
    content_digest,
)
from operatebench.core.engine import Engine
from operatebench.core.instance import new_operation_instance_id
from operatebench.partial_evidence import PartialExecutionEvidence
from operatebench.providers.cost import CostCapExceededError
from operatebench.sdk.development_runtime import _bindings, _grade, _source_digest
from operatebench.sdk.profile_packs.pack import COMMERCE_COMMANDS, COMPLIANCE_COMMANDS
from tools.aggregate_budget import canonical

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "operatebench.three-flow-candidate.v1"
SCOPE = "INTEGRATION_CANDIDATE_NOT_PAID_ADMISSION_NOT_ARTIFACT8"
SPECS = {
    "maintenance": "examples/operatebench/maintenance_delivery_recovery_v0_7.yaml",
    "commerce": "examples/operatebench/commerce_return_refund_profiles/operation.yaml",
    "compliance": (
        "examples/contributions/prospire/property_compliance_profiles/operation.yaml"
    ),
}


@dataclass(frozen=True)
class Trial:
    campaign_id: str
    trial_id: str
    flow: str
    scenario: str
    model: str
    max_output_tokens: int
    provider_binding: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not all(
            type(v) is str and v
            for v in (self.campaign_id, self.trial_id, self.scenario, self.model)
        ):
            raise ValueError("nonempty exact trial identities required")
        if self.flow not in SPECS:
            raise ValueError("unknown real Engine flow")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError(
                "explicit positive endpoint-compatible output bound required"
            )


def source_binding() -> str:
    return _source_digest(
        (
            ROOT / "src" / "operatebench",
            *sorted((ROOT / "tools").glob("three_flow_*.py")),
            ROOT / "tools" / "aggregate_budget.py",
            *sorted((ROOT / "tools").glob("three_flow_river_*.json")),
        )
    )


def provider_identity(transport: Any) -> dict[str, Any]:
    return copy.deepcopy(
        {
            "provider": transport.provider,
            "api": transport.api,
            "model": transport.model,
            "request_mapping": transport.request_mapping,
            "settings": transport.settings,
        }
    )


def _load(trial: Trial) -> tuple[Any, Any, dict[str, Any]]:
    if trial.flow == "maintenance":
        from operatebench.domains.lettings.maintenance.spec import load_spec

        spec = load_spec(ROOT / SPECS[trial.flow])
        spec.scenario(trial.scenario)
        return (
            None,
            spec,
            {
                "pack_id": "lettings.maintenance.synthetic",
                "fixture_sha256": hashlib.sha256(
                    (ROOT / SPECS[trial.flow]).read_bytes()
                ).hexdigest(),
                "scenario_id": trial.scenario,
                "spec_digest": spec.spec_digest_sha256,
                "source_digest": source_binding(),
            },
        )
    factories = (
        COMMERCE_COMMANDS if trial.flow == "commerce" else COMPLIANCE_COMMANDS
    ).factories
    spec = factories.load_spec(ROOT / SPECS[trial.flow])
    binding = _bindings(factories, spec, trial.scenario)
    binding["fixture_sha256"] = hashlib.sha256(
        (ROOT / SPECS[trial.flow]).read_bytes()
    ).hexdigest()
    binding["campaign_runtime_digest"] = source_binding()
    return factories, spec, binding


class _Capture:
    preserve_internal_errors = True

    def __init__(self, inner: Any, partial: PartialExecutionEvidence) -> None:
        self.inner = inner
        self.partial = partial
        self.requests: list[str] = []
        self.request_outputs: list[int] = []
        self.expected_provider = provider_identity(inner)

    def release(self) -> None:
        """End capture ownership without closing the caller's transport.

        Engine/agent cycles may outlive the trial. They must not retain its
        native transport resources once evidence and records are finalized.
        """
        self.inner = None

    @property
    def closed(self) -> bool:
        return self.inner is None

    def send(self, request: ModelRequest) -> ModelResponse:
        if self.closed:
            raise RuntimeError("trial capture is closed")
        if provider_identity(self.inner) != self.expected_provider:
            raise ValueError("provider settings changed during trial")
        self.partial.append(
            {"kind": "request", "request_digest": request.request_digest_sha256}
        )
        self.requests.append(request.request_digest_sha256)
        self.request_outputs.append(request.max_output_tokens)
        self.partial.append(
            {
                "kind": "request_payload",
                "identity": request.identity(),
                "prompt": dict(request.prompt),
            }
        )
        try:
            response: ModelResponse = self.inner.send(request)
        finally:
            self.partial.append(
                {"kind": "provider_turn", "telemetry": self.inner.last_turn}
            )
        self.partial.append(
            {
                "kind": "response",
                "response_digest": normalized_response_digest(
                    model=response.model,
                    stop_classification="max_tokens"
                    if response.stop_reason == "max_tokens"
                    else "completed",
                    tool_calls=response.tool_calls,
                ),
            }
        )
        return response


class _ContextModelAgent(ModelAgent):
    """Bind the chosen per-request output BEFORE recording request identity."""

    def build_request(self, observation: Any) -> ModelRequest:
        request = super().build_request(observation)
        if not isinstance(self._transport, _Capture):
            raise TypeError("context agent requires recording transport")
        if self._transport.inner is None:
            raise TypeError("context agent requires live transport")
        try:
            prepared: ModelRequest = self._transport.inner.prepare_request(request)
        except CostCapExceededError:
            from operatebench.agents.transport import FAULT_BUDGET

            raise ProviderFailure(
                self._excluded(FAULT_BUDGET), "local cost admission refused; run excluded"
            ) from None
        self.max_output_tokens = prepared.max_output_tokens
        return prepared


class _ContextPlayback(RecordedModelAgent):
    def __init__(self, *args: Any, request_outputs: list[int], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.request_outputs = request_outputs

    def decide(self, observation: Any) -> Any:
        self._rebuilder.max_output_tokens = self.request_outputs[self.cursor]
        return super().decide(observation)


def _identity(spec: Any, trial: Trial, instance: str) -> dict[str, str]:
    return {
        "operation_id": spec.operation_id,
        "scenario_id": trial.scenario,
        "spec_digest_sha256": spec.spec_digest_sha256
        if trial.flow == "maintenance"
        else spec.content_digest,
        "agent_id": "campaign-model",
        "operation_instance_id": instance,
    }


def _execute(
    factories: Any,
    spec: Any,
    trial: Trial,
    agent: Any,
    identity: dict[str, str],
    partial: PartialExecutionEvidence | None = None,
) -> Any:
    if trial.flow == "maintenance":
        from operatebench.runner import SELF_CHECK_PLAYBACK, _run_and_evaluate

        run = _run_and_evaluate(
            spec,
            trial.scenario,
            identity["agent_id"],
            operation_instance_id=identity["operation_instance_id"],
            agent_factory=lambda: agent,
            self_check_mode=SELF_CHECK_PLAYBACK,
            agent_kind="model",
            partial_evidence=partial,
        )
        return run.outcome, run.evaluation.as_dict()
    episode = Engine(
        factories.build_domain(spec, trial.scenario),
        agent,
        identity=identity,
        dispatch_failures=frozenset(factories.dispatch_failures(spec, trial.scenario)),
        partial_evidence=partial,
    ).run()
    return episode, _grade(factories, episode, spec, trial.scenario)


def run_trial(trial: Trial, transport: Any, *, output: Path) -> dict[str, Any]:
    """Execute through injected SDK transport; this primitive grants no authority.

    The caller owns transport closure and global guard attachment. Each output
    directory is created once before any send. No automatic retry/rerun.
    """
    factories, spec, binding = _load(trial)
    if transport.model != trial.model:
        raise ValueError("transport model differs from assigned trial")
    provider = provider_identity(transport)
    if trial.provider_binding != provider:
        raise ValueError("provider/settings differ from independently assigned binding")
    parent = output.parent
    if (
        parent != parent.resolve(strict=True)
        or stat.S_IMODE(parent.stat().st_mode) != 0o700
    ):
        raise ValueError("canonical private output parent required")
    output.mkdir(mode=0o700)
    directory = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
        parentfd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(parentfd)
        finally:
            os.close(parentfd)
        identity = _identity(spec, trial, new_operation_instance_id())
        header = {
            "trial": asdict(trial),
            "identity": identity,
            "binding": binding,
            "provider": provider,
            "execution_id": str(uuid4()),
        }
        partial = PartialExecutionEvidence(
            header, path=output / "partial.ndjson", dir_fd=directory
        )
        os.fsync(directory)
    finally:
        os.close(directory)
    previous_observer = transport.wire.observer
    capture = _Capture(transport, partial)
    failure: BaseException | None = None
    try:
        transport.wire.observer = partial.append
        diagnostic = provider["settings"].get("bounds_policy") is not None
        agent = (_ContextModelAgent if diagnostic else ModelAgent)(
            capture,
            model=trial.model,
            agent_id=identity["agent_id"],
            max_output_tokens=trial.max_output_tokens,
            max_transport_calls=None,
        )
        recorder = RecordingAgent(agent, observer=partial.decision)
        episode, grade = _execute(factories, spec, trial, recorder, identity, partial)
        record = {
            "format": FORMAT,
            "scope": SCOPE,
            **header,
            "episode": episode.as_dict(),
            "decisions": recorder.tape().as_list(),
            "request_digests": capture.requests,
            "evaluation": grade,
            "provider_turns": [
                r["telemetry"] for r in partial.rows if r["kind"] == "provider_turn"
            ],
            "evidence_origin": provider["settings"]["evidence_origin"],
        }
        record["record_digest"] = content_digest(record)
        if diagnostic:
            record["request_output_bounds"] = capture.request_outputs
            record["record_digest"] = content_digest(
                {k: v for k, v in record.items() if k != "record_digest"}
            )
        replay_trial(trial, record, expected_record_digest=record["record_digest"])
        _write_once_bytes(
            canonical(record),
            output / "record.json",
            error_adapter=lambda failure: ValueError(
                f"record write refused: {failure.reason}"
            ),
        )
        partial.finish("scored")
        return record
    except BaseException as exc:
        failure = exc
        local_failure = transport.local_failure
        if local_failure is not None:
            failure = RuntimeError("local campaign infrastructure failed")
        try:
            if local_failure is not None:
                # Trusted local seam, regardless of whether the SDK wrapped it.
                partial.append(
                    {"kind": "infrastructure_failure", "local_failure": local_failure}
                )
                partial.finish("aborted")
            elif isinstance(exc, ProviderFailure):
                partial.finish("excluded", exc.fault)
            else:
                partial.finish("aborted")
        except BaseException:
            failure.add_note("Trial evidence finalization also failed")
        if failure is not exc:
            raise failure from exc
        raise
    finally:
        # Ambient caller exception state is not a failure of this invocation.
        try:
            try:
                transport.wire.observer = previous_observer
            finally:
                partial.close()
        except BaseException:
            if failure is None:
                raise
            failure.add_note("Trial evidence cleanup also failed")
        finally:
            capture.release()
            # Do not introduce an exception -> traceback -> frame -> exception cycle.
            del failure


def replay_trial(
    trial: Trial, record: dict[str, Any], *, expected_record_digest: str
) -> dict[str, Any]:
    """Replay all decisions in the original domain and rerun its original grader."""
    if record.get("record_digest") != expected_record_digest:
        raise ValueError("record differs from independently retained closure digest")
    if record.get("format") != FORMAT or record.get("scope") != SCOPE:
        raise ValueError("not a three-flow integration record")
    if record.get("record_digest") != content_digest(
        {k: v for k, v in record.items() if k != "record_digest"}
    ):
        raise ValueError("record checksum mismatch")
    if record["trial"] != asdict(trial):
        raise ValueError("record differs from independently assigned trial")
    if record["provider"] != trial.provider_binding:
        raise ValueError("provider/settings binding differs")
    if record["evidence_origin"] != record["provider"]["settings"]["evidence_origin"]:
        raise ValueError("evidence origin differs")
    factories, spec, binding = _load(trial)
    if record["binding"] != binding:
        raise ValueError("source/spec/profile/evaluator binding differs")
    identity = record["identity"]
    if identity != _identity(spec, trial, identity["operation_instance_id"]):
        raise ValueError("episode identity differs")
    diagnostic = record["provider"]["settings"].get("bounds_policy") is not None
    options = {}
    if diagnostic:
        outputs = record.get("request_output_bounds")
        if (
            not isinstance(outputs, list)
            or len(outputs) != len(record["request_digests"])
            or any(
                type(o) is not int or not 0 < o <= trial.max_output_tokens
                for o in outputs
            )
        ):
            raise ValueError("complete per-request output identity required")
        for output, turn in zip(outputs, record["provider_turns"], strict=True):
            if turn["requested_output_bound"] != output or (
                type(turn["input_bound"]) is not int
                or turn["input_bound"] < 0
                or turn["input_bound"] + output
                > record["provider"]["settings"]["context_request_bound"]
            ):
                raise ValueError("per-request context evidence differs")
            wire = turn["wire"]
            if isinstance(wire, dict):
                import json

                payload = json.loads(wire["request_utf8"])
                if payload.get("max_output_tokens", payload.get("max_tokens")) != output:
                    raise ValueError("wire output differs from request identity")
            elif isinstance(wire, list):
                import base64

                from tools.three_flow_river import sdk

                pb, _ = sdk()
                submissions = [row for row in wire if row["model_submission"]]
                if len(submissions) != 1:
                    raise ValueError("exactly one native submission required")
                raw = base64.b64decode(submissions[0]["request_base64"], validate=True)
                if hashlib.sha256(raw).hexdigest() != submissions[0]["request_sha256"]:
                    raise ValueError("native request digest differs")
                proto = pb.InferenceGenerateRequest.FromString(raw)
                if (
                    len(proto.prompts) != 1
                    or proto.prompts[0].max_tokens != output
                    or len(proto.prompts[0].input_ids) != turn["input_bound"]
                    or proto.base_model != trial.model
                ):
                    raise ValueError(
                        "native wire output/input differs from request identity"
                    )
            else:
                raise ValueError("unknown diagnostic wire evidence")
        options["request_outputs"] = outputs
    elif "request_output_bounds" in record:
        raise ValueError("historical static policy cannot select dynamic output")
    playback = (_ContextPlayback if diagnostic else RecordedModelAgent)(
        tape_from_records(record["decisions"]),
        agent_id=identity["agent_id"],
        model=trial.model,
        max_output_tokens=trial.max_output_tokens,
        request_digests=record["request_digests"],
        **options,
    )
    if len(record["request_digests"]) != len(record["decisions"]):
        raise ValueError("full request/decision coverage required")
    episode, grade = _execute(factories, spec, trial, playback, identity)
    playback.check_exhausted()
    if canonical(episode.as_dict()) != canonical(record["episode"]):
        raise ValueError("full replay episode differs")
    if canonical(grade) != canonical(record["evaluation"]):
        raise ValueError("original evaluator result differs")
    return {
        "provider_calls": playback.provider_calls,
        "decisions_consumed": playback.cursor,
        "evaluation": grade,
    }
