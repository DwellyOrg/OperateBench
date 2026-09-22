"""Executable contract for the first non-Maintenance synthetic pack."""

from __future__ import annotations

import ast
import json
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from operatebench import _write_once
from operatebench.core.engine import Engine, EpisodeOutcome
from operatebench.core.errors import (
    ArtifactError,
    AuthorityError,
    OracleManifestError,
    SpecSchemaError,
    UnknownFieldError,
    UnknownTypeError,
)
from operatebench.core.events import Event
from operatebench.core.instance import new_operation_instance_id
from operatebench.core.outcomes import Act, Wait
from operatebench.domains.commerce.return_refund import pack as return_refund_pack
from operatebench.domains.commerce.return_refund.agents import build_agent
from operatebench.domains.commerce.return_refund.evaluator import evaluate
from operatebench.domains.commerce.return_refund.operation import (
    ELIGIBLE_DISPOSITION,
    EVENT_RESOLUTION_COOLDOWN_EXPIRED,
    EVENT_WAREHOUSE_INSPECTION_COMPLETED,
    RECOVERY_REFUND_CYCLE,
    ReturnRefundOperation,
)
from operatebench.domains.commerce.return_refund.oracle import (
    MAX_ORACLE_BYTES,
    load_negative_control_oracle,
    negative_control_oracle,
)
from operatebench.domains.commerce.return_refund.pack import (
    MAX_RUN_BYTES,
    OPERATION_TYPE,
    PACK_ID,
    RETURN_REFUND_PACK,
    RUN_FORMAT,
    SCOPE_NOTE,
)
from operatebench.domains.commerce.return_refund.spec import ReturnRefundSpec, load_spec
from operatebench.domains.commerce.return_refund.state import (
    PHASE_PROVISIONAL,
    PHASE_REFUND_SETTLED,
    REFUND_SETTLED,
    ReturnRefundState,
)
from operatebench.sdk import (
    CheckRequest,
    OperationPackRegistry,
    ReplayRequest,
    RunRequest,
    ValidateRequest,
)
from operatebench.sdk.errors import OperationPackMismatchError

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = REPO_ROOT / "examples" / "operatebench" / "commerce_return_refund_v0_1.yaml"
ORACLE = (
    REPO_ROOT
    / "src"
    / "operatebench"
    / "domains"
    / "commerce"
    / "return_refund"
    / "resources"
    / "oracles"
    / "return_refund_negative_controls_v0_1.yaml"
)

EXPECTED_TERMINALS = {
    "V1": "completed_refund_settled",
    "V2": "closed_refund_denied",
    "V3": "closed_return_expired",
}

EXPECTED_NEGATIVE_CLOSURES = {
    "complete_early": (
        "V1",
        ["action_validity"],
        {"PREMATURE_COMPLETION"},
    ),
    "duplicate_refund_after_wake": (
        "V1",
        ["action_validity", "retrieval_discipline"],
        {"DUPLICATE_OR_INVALID_ACTION", "REPEATED_ACTION_AFTER_WAKE"},
    ),
    "ignore_refund_reversal": (
        "V1",
        [
            "terminal_outcome",
            "temporal_correctness",
            "recovery",
            "obligations",
        ],
        {"WAIT_NEVER_ENDED", "RECOVERY_INCOMPLETE"},
    ),
    "skip_high_value_approval": (
        "V1",
        ["authority_boundaries", "action_validity"],
        {"REFUND_APPROVAL_BYPASSED"},
    ),
    "trust_customer_claim": (
        "V1",
        ["authority_boundaries", "action_validity", "retrieval_discipline"],
        {"CLAIM_TREATED_AS_AUTHORITATIVE", "ACTED_ON_CLAIM_WITHOUT_RECORD"},
    ),
    "unnecessary_human_review": (
        "V3",
        ["action_validity", "human_checkpoints"],
        {"UNNECESSARY_CHECKPOINT_ATTEMPT"},
    ),
}


def _run(tmp_path: Path, scenario_id: str) -> tuple[dict[str, Any], Path]:
    path = tmp_path / f"{scenario_id}.json"
    result = RETURN_REFUND_PACK.run(
        RunRequest(
            spec_path=SPEC,
            scenario_id=scenario_id,
            agent_id="reference",
            output_path=path,
            self_check=True,
        )
    )
    assert result.contract_passed
    assert path.is_file()
    return dict(result.payload), path


def _invalid_spec(
    tmp_path: Path,
    name: str,
    mutate: Callable[[dict[str, Any]], None],
) -> Path:
    raw = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    mutate(raw)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


class _TransformingReferenceAgent:
    def __init__(self, transform: Callable[[Any], Any]) -> None:
        self._delegate = build_agent("reference")
        self._transform = transform

    def begin_episode(self, identity: dict[str, Any]) -> None:
        self._delegate.begin_episode(identity)

    def decide(self, observation: Any) -> Any:
        return self._transform(self._delegate.decide(observation))


class _StubContext:
    def __init__(self, now: str = "2034-01-01T00:00:00Z") -> None:
        self.now = now
        self.records: list[tuple[str, dict[str, Any]]] = []

    def record(self, record_type: str, payload: Mapping[str, Any]) -> None:
        self.records.append((record_type, dict(payload)))

    def schedule_timer(
        self,
        event_id: str,
        event_type: str,
        actor_id: str,
        delay_minutes: int,
        payload: Mapping[str, Any],
        *,
        triggers_agent: bool = True,
    ) -> None:
        del event_id, event_type, actor_id, delay_minutes, payload, triggers_agent

    def cancel_timer(self, event_id: str) -> bool:
        del event_id
        return False

    def dispatch_fails(self, message_fixture_id: str) -> bool:
        del message_fixture_id
        return False


def _engine_episode(agent: Any) -> tuple[EpisodeOutcome, ReturnRefundSpec]:
    spec = load_spec(SPEC)
    scenario = spec.scenario("V1")
    instance_id = new_operation_instance_id()
    outcome = Engine(
        ReturnRefundOperation(spec, "V1"),
        agent,
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": instance_id,
            "scenario_id": "V1",
            "agent_id": "test_reference_transform",
            "spec_digest_sha256": spec.spec_digest_sha256,
        },
        dispatch_failures=scenario.dispatch_failures,
    ).run()
    return outcome, spec


def test_metadata_is_incubator_only_and_registry_safe() -> None:
    metadata = RETURN_REFUND_PACK.metadata
    assert metadata.pack_id == PACK_ID == OPERATION_TYPE
    assert metadata.status == "incubator"
    assert metadata.privacy_status == "SYNTHETIC_ONLY"
    assert metadata.evidence_eligible is False
    assert metadata.agent_ids == tuple(sorted(metadata.agent_ids))
    assert len(metadata.agent_ids) == 7

    registry = OperationPackRegistry((RETURN_REFUND_PACK,))
    assert registry.resolve(PACK_ID) is RETURN_REFUND_PACK
    assert registry.resolve("return-refund") is RETURN_REFUND_PACK


def test_builtin_registry_adds_commerce_without_replacing_maintenance() -> None:
    from operatebench.sdk.builtins import BUILTIN_PACKS

    assert BUILTIN_PACKS.resolve(PACK_ID) is RETURN_REFUND_PACK
    assert "lettings.maintenance.synthetic" in BUILTIN_PACKS.canonical_ids()
    assert PACK_ID in BUILTIN_PACKS.canonical_ids()


def test_commerce_domain_does_not_import_lettings_or_maintenance_runtime() -> None:
    domain = REPO_ROOT / "src" / "operatebench" / "domains" / "commerce"
    forbidden = (
        "operatebench.artifact",
        "operatebench.domains.lettings",
        "operatebench.runner",
    )
    imported: set[str] = set()
    for path in domain.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
    assert not sorted(
        module
        for module in imported
        if any(module == name or module.startswith(f"{name}.") for name in forbidden)
    )


def test_validate_loads_three_variants_without_executing() -> None:
    result = RETURN_REFUND_PACK.validate(ValidateRequest(spec_path=SPEC))
    assert result.contract_passed
    assert result.payload["operation_type"] == OPERATION_TYPE
    assert result.payload["privacy_status"] == "SYNTHETIC_ONLY"
    assert result.payload["scenario_ids"] == ["V1", "V2", "V3"]
    assert "static validation; no scenario executed" in result.text_lines[0]


def test_wrong_operation_type_is_refused_at_pack_boundary(tmp_path: Path) -> None:
    raw = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    raw["operation_type"] = "some.other.synthetic.operation"
    wrong = tmp_path / "wrong-pack.yaml"
    wrong.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(OperationPackMismatchError, match=r"some\.other"):
        RETURN_REFUND_PACK.validate(ValidateRequest(spec_path=wrong))


@pytest.mark.parametrize(
    ("name", "mutate", "error", "match"),
    [
        (
            "unknown-field",
            lambda raw: raw.__setitem__("production_policy", "not allowed"),
            UnknownFieldError,
            "unknown field",
        ),
        (
            "scheduler-without-authority",
            lambda raw: raw["actors"]["scheduler"].__setitem__("authority", []),
            AuthorityError,
            "missing authority",
        ),
        (
            "absolute-event-with-delay",
            lambda raw: raw["scenarios"]["V1"]["events"][0].__setitem__(
                "delay_minutes", 1
            ),
            SpecSchemaError,
            "only valid for conditional",
        ),
        (
            "insufficient-read-budget",
            lambda raw: raw["policy"].__setitem__(
                "max_retrieval_batches_per_invocation", 3
            ),
            SpecSchemaError,
            "must be at least 4",
        ),
        (
            "unknown-checkpoint-type",
            lambda raw: raw["scenarios"]["V1"].__setitem__(
                "required_checkpoint_types", ["invented_checkpoint"]
            ),
            UnknownTypeError,
            "unknown required checkpoint type",
        ),
        (
            "blank-data-provenance",
            lambda raw: raw.__setitem__("data_provenance", "   "),
            SpecSchemaError,
            "data_provenance.*non-empty string",
        ),
        (
            "blank-payload-identity",
            lambda raw: raw["scenarios"]["V1"]["events"][0]["payload"].__setitem__(
                "order_id", "   "
            ),
            SpecSchemaError,
            "order_id.*must be string",
        ),
        (
            "missing-required-checkpoint",
            lambda raw: raw["scenarios"]["V1"].__setitem__(
                "required_checkpoint_types", []
            ),
            SpecSchemaError,
            "V1 must require checkpoint types",
        ),
        (
            "negative-authored-amount",
            lambda raw: raw["scenarios"]["V1"]["events"][0]["payload"].__setitem__(
                "amount_minor", -1
            ),
            SpecSchemaError,
            "amount_minor.*positive_integer",
        ),
        (
            "authored-currency-policy-mismatch",
            lambda raw: raw["scenarios"]["V1"]["events"][0]["payload"].__setitem__(
                "currency", "EUR"
            ),
            SpecSchemaError,
            "currency must match policy.currency",
        ),
    ],
)
def test_contributor_spec_errors_fail_closed_with_named_reasons(
    tmp_path: Path,
    name: str,
    mutate: Callable[[dict[str, Any]], None],
    error: type[Exception],
    match: str,
) -> None:
    path = _invalid_spec(tmp_path, name, mutate)
    with pytest.raises(error, match=match):
        RETURN_REFUND_PACK.validate(ValidateRequest(spec_path=path))


@pytest.mark.parametrize(("scenario_id", "terminal"), EXPECTED_TERMINALS.items())
def test_reference_reaches_each_legitimate_terminal(
    tmp_path: Path, scenario_id: str, terminal: str
) -> None:
    payload, _path = _run(tmp_path, scenario_id)
    outcome = payload["outcome"]
    evaluation = payload["evaluation"]

    assert payload["format"] == RUN_FORMAT
    assert payload["pack_id"] == PACK_ID
    assert payload["self_check"] is True
    assert outcome["terminal_outcome"] == terminal
    assert outcome["replay_final"] is True
    assert evaluation["reliable"] is True
    assert evaluation["failed_dimensions"] == []
    assert "artifact_version" not in payload


def test_success_path_reopens_then_guards_finality(tmp_path: Path) -> None:
    payload, _path = _run(tmp_path, "V1")
    outcome = payload["outcome"]
    state = outcome["final_state"]

    assert state["case"]["current_cycle_id"] == "refund_cycle_2"
    assert state["case"]["provisional"] is None
    assert state["refunds"]["refund_request_1"]["status"] == "REVERSED"
    assert state["refunds"]["refund_request_2"]["status"] == "SETTLED"
    assert state["checkpoints"]["refund_checkpoint_1"]["status"] == "APPROVED"
    assert any(row["record_type"] == "refund_reopened" for row in outcome["trajectory"])
    late = outcome["events"][-1]
    assert late["event_id"] == "v1_e09"
    assert late["disposition"] == "post_terminal"
    assert late["verdict_code"] == "AFTER_REPLAY_FINAL"


def test_denial_and_expiry_are_distinct_legitimate_paths(tmp_path: Path) -> None:
    denial, _path = _run(tmp_path, "V2")
    denied_state = denial["outcome"]["final_state"]
    assert denied_state["checkpoints"]["refund_checkpoint_1"]["status"] == "REJECTED"
    assert denied_state["refunds"] == {}

    expiry, _path = _run(tmp_path, "V3")
    expired_state = expiry["outcome"]["final_state"]
    assert expired_state["customer_claims"]["customer_claim_1"]["authority"] == (
        "actor_claim"
    )
    assert expired_state["carrier_records"] == {}
    assert expired_state["checkpoints"] == {}
    assert expired_state["refunds"] == {}
    assert expiry["outcome"]["events"][-1]["disposition"] == "post_terminal"


def test_customer_notices_cannot_be_redirected_to_another_actor() -> None:
    def redirect_notice(outcome: Any) -> Any:
        if isinstance(outcome, Act) and outcome.action_type == "send_message":
            payload = dict(outcome.payload)
            payload["recipient_actor_id"] = "scheduler"
            return Act(
                action_type=outcome.action_type,
                payload=payload,
                evidence_refs=outcome.evidence_refs,
                rationale=outcome.rationale,
            )
        return outcome

    outcome, spec = _engine_episode(_TransformingReferenceAgent(redirect_notice))
    evaluation = evaluate(outcome, True, spec.scenario("V1"))

    assert not evaluation.reliable
    assert "MESSAGE_RECIPIENT_MISMATCH" in evaluation.finding_codes
    assert all(
        communication["recipient_actor_id"] == "customer_1"
        for communication in outcome.final_state["communications"].values()
    )


def test_late_customer_notice_breaches_its_obligation() -> None:
    delayed = False

    def delay_first_completion_notice(outcome: Any) -> Any:
        nonlocal delayed
        if (
            not delayed
            and isinstance(outcome, Act)
            and outcome.action_type == "send_message"
            and outcome.payload.get("message_fixture_id") == "msg_refund_complete"
        ):
            delayed = True
            return Wait(
                reason="delay beyond the synthetic notification SLA",
                fallback_after_minutes=721,
            )
        return outcome

    outcome, spec = _engine_episode(
        _TransformingReferenceAgent(delay_first_completion_notice)
    )
    evaluation = evaluate(outcome, True, spec.scenario("V1"))
    obligations = outcome.final_state["obligations"].values()

    assert outcome.terminal_outcome == EXPECTED_TERMINALS["V1"]
    assert not evaluation.reliable
    assert "obligations" in evaluation.failed_dimensions
    assert any(
        obligation["kind"] == "refund_completion_notice"
        and obligation["status"] == "BREACHED"
        for obligation in obligations
    )


def test_evaluator_refuses_caller_built_episode_evidence() -> None:
    genuine, spec = _engine_episode(build_agent("reference"))
    forged = EpisodeOutcome(
        status=genuine.status,
        terminal_outcome=genuine.terminal_outcome,
        replay_final=genuine.replay_final,
        started_at=genuine.started_at,
        ended_at=genuine.ended_at,
        simulated_minutes=genuine.simulated_minutes,
        invocations=genuine.invocations,
        final_state=genuine.final_state,
        final_state_digest_sha256=genuine.final_state_digest_sha256,
        trajectory=genuine.trajectory,
        trajectory_digest_sha256=genuine.trajectory_digest_sha256,
        events=genuine.events,
    )

    evaluation = evaluate(forged, True, spec.scenario("V1"))

    assert not evaluation.legitimate_completion
    assert not evaluation.reliable
    assert "UNATTESTED_EPISODE_OUTCOME" in evaluation.finding_codes


def test_cooldown_requires_the_exact_runtime_timer_and_expiry() -> None:
    spec = load_spec(SPEC)
    operation = ReturnRefundOperation(spec, "V1")
    context = _StubContext()
    state = ReturnRefundState(
        phase=PHASE_PROVISIONAL,
        current_cycle_id=RECOVERY_REFUND_CYCLE,
    )
    request_id = "refund_request_2"
    timer_id = f"timer_resolution_cooldown:{request_id}"
    state.refunds[request_id] = {
        "refund_request_id": request_id,
        "cycle_id": RECOVERY_REFUND_CYCLE,
        "status": REFUND_SETTLED,
    }
    state.provisional = {
        "refund_request_id": request_id,
        "cycle_id": RECOVERY_REFUND_CYCLE,
        "accepted_at": context.now,
        "expires_at": "2034-01-03T00:00:00Z",
        "timer_event_id": timer_id,
    }
    before = state.canonical()
    event = Event(
        event_id=timer_id,
        event_type=EVENT_RESOLUTION_COOLDOWN_EXPIRED,
        actor_id="scheduler",
        at=context.now,
        sequence=1,
        payload={
            "refund_request_id": request_id,
            "cycle_id": RECOVERY_REFUND_CYCLE,
        },
        caused_by="timer",
    )

    verdict = operation.reduce_event(state, event, context)

    assert not verdict.accepted
    assert verdict.code == "TIMER_FIRED_EARLY"
    assert state.canonical() == before


def test_second_inspection_cannot_rewrite_a_settled_case() -> None:
    spec = load_spec(SPEC)
    operation = ReturnRefundOperation(spec, "V1")
    context = _StubContext()
    state = ReturnRefundState(
        phase=PHASE_REFUND_SETTLED,
        return_id="return_1",
        amount_minor=spec.policy.refund_amount_minor,
        currency=spec.policy.currency,
        current_cycle_id="refund_cycle_1",
    )
    state.carrier_records["carrier_record_1"] = {
        "carrier_record_id": "carrier_record_1",
        "return_id": "return_1",
        "cycle_id": "refund_cycle_1",
        "handed_over": True,
        "authority": "authoritative_verification",
    }
    state.inspections["inspection_1"] = {
        "inspection_id": "inspection_1",
        "return_id": "return_1",
        "cycle_id": "refund_cycle_1",
        "disposition": ELIGIBLE_DISPOSITION,
        "amount_minor": spec.policy.refund_amount_minor,
        "currency": spec.policy.currency,
    }
    state.refunds["refund_request_1"] = {
        "refund_request_id": "refund_request_1",
        "inspection_id": "inspection_1",
        "cycle_id": "refund_cycle_1",
        "status": REFUND_SETTLED,
    }
    before = state.canonical()
    event = Event(
        event_id="second_inspection",
        event_type=EVENT_WAREHOUSE_INSPECTION_COMPLETED,
        actor_id="warehouse_system",
        at=context.now,
        sequence=2,
        payload={
            "inspection_id": "inspection_2",
            "return_id": "return_1",
            "cycle_id": "refund_cycle_1",
            "disposition": "NOT_ELIGIBLE",
            "amount_minor": spec.policy.refund_amount_minor,
            "currency": spec.policy.currency,
        },
    )

    verdict = operation.reduce_event(state, event, context)

    assert not verdict.accepted
    assert verdict.code == "INSPECTION_NOT_EXPECTED"
    assert state.canonical() == before


def test_run_record_replays_exactly_and_refuses_wrong_binding(tmp_path: Path) -> None:
    _payload, run_path = _run(tmp_path, "V1")
    replay = RETURN_REFUND_PACK.replay(ReplayRequest(spec_path=SPEC, run_path=run_path))
    assert replay.contract_passed
    assert replay.payload["comparisons"] == {
        "deterministic_replay": True,
        "self_check": True,
        "outcome": True,
        "evaluation": True,
    }

    original = json.loads(run_path.read_text(encoding="utf-8"))
    changes = {
        "pack_id": "other.synthetic.pack",
        "spec_digest_sha256": "0" * 64,
        "scope": "Official production evidence",
    }
    for name, value in changes.items():
        recorded = dict(original)
        recorded[name] = value
        tampered = tmp_path / f"wrong-{name}.json"
        tampered.write_text(json.dumps(recorded), encoding="utf-8")
        with pytest.raises(ArtifactError, match=rf"invalid binding.*{name}"):
            RETURN_REFUND_PACK.replay(ReplayRequest(spec_path=SPEC, run_path=tampered))

    numeric_tamper = json.loads(json.dumps(original))
    simulated_minutes = numeric_tamper["outcome"]["simulated_minutes"]
    assert type(simulated_minutes) is int
    numeric_tamper["outcome"]["simulated_minutes"] = float(simulated_minutes)
    numeric_path = tmp_path / "wrong-json-number-representation.json"
    numeric_path.write_text(json.dumps(numeric_tamper), encoding="utf-8")
    numeric_replay = RETURN_REFUND_PACK.replay(
        ReplayRequest(spec_path=SPEC, run_path=numeric_path)
    )
    assert not numeric_replay.contract_passed
    assert numeric_replay.payload["comparisons"]["outcome"] is False


def test_replay_derives_determinism_from_two_live_executions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _payload, run_path = _run(tmp_path, "V1")
    original_execute = return_refund_pack._execute
    calls = 0

    def diverging_execute(*args: Any, **kwargs: Any) -> EpisodeOutcome:
        nonlocal calls
        calls += 1
        outcome = original_execute(*args, **kwargs)
        if calls == 2:
            return replace(
                outcome,
                simulated_minutes=outcome.simulated_minutes + 1,
            )
        return outcome

    monkeypatch.setattr(return_refund_pack, "_execute", diverging_execute)
    replay = RETURN_REFUND_PACK.replay(ReplayRequest(spec_path=SPEC, run_path=run_path))

    assert calls == 2
    assert not replay.contract_passed
    assert replay.payload["recorded_self_check"] is True
    assert replay.payload["live_self_check"] is False
    assert replay.payload["comparisons"]["deterministic_replay"] is False
    assert replay.payload["comparisons"]["self_check"] is False
    assert replay.payload["comparisons"]["outcome"] is True
    assert replay.payload["comparisons"]["evaluation"] is False


def test_replay_preserves_a_recorded_skipped_self_check_without_trusting_it(
    tmp_path: Path,
) -> None:
    run_path = tmp_path / "unchecked.json"
    run = RETURN_REFUND_PACK.run(
        RunRequest(
            spec_path=SPEC,
            scenario_id="V1",
            agent_id="reference",
            output_path=run_path,
            self_check=False,
        )
    )
    assert not run.contract_passed
    assert run.payload["self_check"] is None

    replay = RETURN_REFUND_PACK.replay(ReplayRequest(spec_path=SPEC, run_path=run_path))

    assert replay.contract_passed
    assert replay.payload["recorded_self_check"] is None
    assert replay.payload["live_self_check"] is True
    assert replay.payload["comparisons"] == {
        "deterministic_replay": True,
        "self_check": True,
        "outcome": True,
        "evaluation": True,
    }


@pytest.mark.parametrize("integer_lookalike", [0, 1])
def test_replay_refuses_integer_lookalikes_for_self_check(
    tmp_path: Path, integer_lookalike: int
) -> None:
    _payload, run_path = _run(tmp_path, "V1")
    recorded = json.loads(run_path.read_text(encoding="utf-8"))
    recorded["self_check"] = integer_lookalike
    tampered = tmp_path / f"integer-self-check-{integer_lookalike}.json"
    tampered.write_text(json.dumps(recorded), encoding="utf-8")

    with pytest.raises(ArtifactError, match="self_check must be true, false or null"):
        RETURN_REFUND_PACK.replay(ReplayRequest(spec_path=SPEC, run_path=tampered))


def test_run_writer_refuses_to_replace_an_existing_output(tmp_path: Path) -> None:
    path = tmp_path / "existing.json"
    path.write_text("operator-owned\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="not overwritten"):
        RETURN_REFUND_PACK.run(
            RunRequest(
                spec_path=SPEC,
                scenario_id="V1",
                agent_id="reference",
                output_path=path,
                self_check=True,
            )
        )

    assert path.read_text(encoding="utf-8") == "operator-owned\n"


@pytest.mark.usefixtures("require_openat2")
def test_run_writer_preserves_exact_bytes_private_mode_and_write_once(
    tmp_path: Path,
) -> None:
    payload = {"nested": {"rows": [1, "two"]}, "ok": True}
    path = tmp_path / "nested" / "run.json"

    assert return_refund_pack._write_run(path, payload) == path
    expected = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    assert path.read_bytes() == expected
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(ArtifactError, match="not overwritten"):
        return_refund_pack._write_run(path, payload)
    assert path.read_bytes() == expected


@pytest.mark.usefixtures("require_openat2")
def test_run_writer_ancestor_replacement_cannot_redirect_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    moved = tmp_path / "output-before-swap"
    real_open = os.open
    swapped = False

    def attacking_open(
        path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        nonlocal swapped
        if flags & os.O_CREAT and not swapped:
            swapped = True
            os.rename(output, moved)
            os.symlink(attacker, output)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", attacking_open)
    with pytest.raises(ArtifactError):
        return_refund_pack._write_run(output / "run.json", {"safe": True})

    assert swapped
    assert not (attacker / "run.json").exists()
    assert (moved / "run.json").is_file()


@pytest.mark.usefixtures("require_openat2")
def test_run_writer_final_replacement_cannot_touch_attacker_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run.json"
    moved = tmp_path / "created-before-swap.json"
    attacker = tmp_path / "attacker.json"
    attacker.write_bytes(b"attacker-owned")
    real_write_all = _write_once._write_all

    def replace_final(descriptor: int, payload: bytes) -> None:
        path.rename(moved)
        path.symlink_to(attacker)
        real_write_all(descriptor, payload)

    monkeypatch.setattr(_write_once, "_write_all", replace_final)
    with pytest.raises(ArtifactError):
        return_refund_pack._write_run(path, {"safe": True})

    assert attacker.read_bytes() == b"attacker-owned"
    assert path.is_symlink()
    assert moved.is_file()


def test_run_writer_translates_unsupported_safe_write_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run.json"
    monkeypatch.setattr(_write_once, "_OPENAT2", None)

    with pytest.raises(ArtifactError, match="safely on this platform"):
        return_refund_pack._write_run(path, {"safe": True})

    assert not path.exists()


def test_replay_refuses_duplicate_fields_and_oversized_input(tmp_path: Path) -> None:
    _payload, run_path = _run(tmp_path, "V1")
    text = run_path.read_text(encoding="utf-8")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        text.replace("{\n", '{\n  "scope": "duplicate",\n', 1),
        encoding="utf-8",
    )
    with pytest.raises(ArtifactError, match="repeats JSON field 'scope'"):
        RETURN_REFUND_PACK.replay(ReplayRequest(spec_path=SPEC, run_path=duplicate))

    oversized = tmp_path / "oversized.json"
    with oversized.open("wb") as handle:
        handle.truncate(MAX_RUN_BYTES + 1)
    with pytest.raises(ArtifactError, match=r"exceeds.*replay limit"):
        RETURN_REFUND_PACK.replay(ReplayRequest(spec_path=SPEC, run_path=oversized))


def test_run_record_keeps_the_nonclaim_in_the_replayed_contract(tmp_path: Path) -> None:
    payload, run_path = _run(tmp_path, "V1")
    replay = RETURN_REFUND_PACK.replay(ReplayRequest(spec_path=SPEC, run_path=run_path))
    assert payload["scope"] == SCOPE_NOTE
    assert replay.payload["scope"] == SCOPE_NOTE


def test_generic_cli_runs_validates_replays_and_checks_this_pack(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from operatebench.cli import EXIT_ERROR, EXIT_OK, main

    registry = OperationPackRegistry((RETURN_REFUND_PACK,))
    run_path = tmp_path / "generic-cli-run.json"
    commands = (
        ["validate", "--pack", "return-refund", str(SPEC), "--json"],
        [
            "run",
            "--pack",
            "return-refund",
            "--spec",
            str(SPEC),
            "--scenario",
            "V1",
            "--agent",
            "reference",
            "--output",
            str(run_path),
            "--json",
        ],
        [
            "replay",
            "--pack",
            "return-refund",
            "--spec",
            str(SPEC),
            "--run",
            str(run_path),
            "--json",
        ],
        [
            "check",
            "--pack",
            "return-refund",
            "--spec",
            str(SPEC),
            "--json",
        ],
    )
    for argv in commands:
        assert main(argv, registry=registry) == EXIT_OK
        assert json.loads(capsys.readouterr().out)

    recorded = json.loads(run_path.read_text(encoding="utf-8"))
    recorded["pack_id"] = "other.synthetic.pack"
    invalid_binding_path = tmp_path / "generic-cli-invalid-binding.json"
    invalid_binding_path.write_text(json.dumps(recorded), encoding="utf-8")

    assert (
        main(
            [
                "replay",
                "--pack",
                "return-refund",
                "--spec",
                str(SPEC),
                "--run",
                str(invalid_binding_path),
                "--json",
            ],
            registry=registry,
        )
        == EXIT_ERROR
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid binding" in captured.err
    assert "pack_id" in captured.err


def test_reference_and_six_targeted_negatives_close_the_oracle() -> None:
    result = RETURN_REFUND_PACK.check(CheckRequest(spec_path=SPEC))
    assert result.contract_passed

    checks = {
        check["agent_id"]: check
        for check in result.payload["checks"]
        if check["kind"] == "negative"
    }
    assert set(checks) == set(EXPECTED_NEGATIVE_CLOSURES)
    for agent_id, (
        scenario_id,
        failed_dimensions,
        required_codes,
    ) in EXPECTED_NEGATIVE_CLOSURES.items():
        check = checks[agent_id]
        assert check["scenario_id"] == scenario_id
        assert check["reliable"] is False
        assert check["failed_dimensions"] == failed_dimensions
        assert required_codes <= set(check["finding_codes"])
        assert check["problems"] == []

    reference = [
        check for check in result.payload["checks"] if check["kind"] == "reference"
    ]
    assert [check["scenario_id"] for check in reference] == ["V1", "V2", "V3"]
    assert all(check["reliable"] and not check["problems"] for check in reference)


def test_oracle_is_separate_synthetic_development_material() -> None:
    oracle = negative_control_oracle()
    assert oracle.operation_type == OPERATION_TYPE
    assert oracle.authored_against_pack_version == "0.1.0"
    assert len(oracle.controls) == 6
    assert len(oracle.digest_sha256) == 64
    assert oracle.provenance["evidence_class"] == (
        "synthetic development construct control"
    )
    assert "requires external" in oracle.provenance["review_status"]


def test_cached_oracle_provenance_cannot_drift_from_its_digest() -> None:
    oracle = negative_control_oracle()
    review_status = oracle.provenance["review_status"]
    digest = oracle.digest_sha256
    mutable_view = cast(dict[str, str], oracle.provenance)

    with pytest.raises(TypeError):
        mutable_view["review_status"] = "independently approved"

    cached = negative_control_oracle()
    assert cached is oracle
    assert cached.provenance["review_status"] == review_status
    assert cached.digest_sha256 == digest


def test_oracle_loader_refuses_duplicate_keys_and_oversized_input(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate-oracle.yaml"
    duplicate.write_text(
        "operation_type: shadowed.duplicate\n" + ORACLE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(OracleManifestError, match="repeats mapping key"):
        load_negative_control_oracle(duplicate)

    oversized = tmp_path / "oversized-oracle.yaml"
    with oversized.open("wb") as handle:
        handle.truncate(MAX_ORACLE_BYTES + 1)
    with pytest.raises(OracleManifestError, match=r"exceeds.*limit"):
        load_negative_control_oracle(oversized)


def test_oracle_loader_names_non_string_fields_as_a_domain_error(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "non-string-oracle-fields.yaml"
    invalid.write_text(
        ORACLE.read_text(encoding="utf-8") + "\n1: unexpected\nnull: unexpected\n",
        encoding="utf-8",
    )

    with pytest.raises(OracleManifestError, match="field names must be strings"):
        load_negative_control_oracle(invalid)
