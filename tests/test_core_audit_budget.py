# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import copy
import dataclasses
import gc
import json
import os
import pickle
import subprocess
import sys
import threading
import weakref
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import operatebench.core as core_package
import operatebench.core.audit_budget as audit_budget_module
from operatebench.core.audit_budget import (
    AuditBudgetDeclaration,
    AuditBudgetRuntime,
    AuditBudgetSource,
    AuditIntegrityHalt,
    AuditIntent,
    AuditTransitionScope,
)


def _runtime(*, occurrences: int = 1, fanout: int = 2) -> AuditBudgetRuntime:
    source = AuditBudgetSource(
        "test.audit", "scheduled_audit_event_intent", occurrences, fanout
    )
    return AuditBudgetRuntime(
        AuditBudgetDeclaration(
            "operatebench.domain-generated-audit-budget.v1",
            "1" * 64,
            "opinst_test",
            occurrences * fanout,
            (source,),
        )
    )


def _inject_state(
    runtime: AuditBudgetRuntime, state: object, *, coherent: bool = False
) -> None:
    control = runtime._control
    object.__setattr__(
        runtime,
        "_control",
        dataclasses.replace(
            control,
            state=state,
            trusted_state=state if coherent else control.trusted_state,
        ),
    )


@pytest.mark.parametrize(
    "counted_kind",
    ["scheduled_audit_event_intent", "immediate_audit_event_intent"],
)
def test_exact_two_intent_batch_commits_contiguous_immutable_evidence(
    counted_kind: str,
) -> None:
    source = AuditBudgetSource("test.audit", counted_kind, 1, 2)
    declaration = AuditBudgetDeclaration(
        "operatebench.domain-generated-audit-budget.v1",
        "1" * 64,
        "opinst_test",
        2,
        (source,),
    )
    runtime = AuditBudgetRuntime(declaration)

    transition = runtime.begin_transition("test.transition", 3)
    reservation = transition.reserve_batch(
        "test.audit", counted_kind, ("test.audit.intent-1", "test.audit.intent-2")
    )
    reservation.consume("test.audit.intent-1")
    reservation.consume("test.audit.intent-2")
    transition.commit()

    snapshot = runtime.snapshot()
    assert snapshot.operation_committed_count == 2
    assert snapshot.source_committed_counts == (("test.audit", 2),)
    assert snapshot.source_occurrence_counts == (("test.audit", 1),)
    assert [row.batch_ordinal for row in snapshot.committed_intents] == [0, 1]
    assert [row.operation_committed_count for row in snapshot.committed_intents] == [1, 2]
    assert len({row.intent_digest_sha256 for row in snapshot.committed_intents}) == 2
    assert dataclasses.is_dataclass(source) and source.__dataclass_params__.frozen
    assert (
        dataclasses.is_dataclass(declaration) and declaration.__dataclass_params__.frozen
    )
    assert dataclasses.is_dataclass(snapshot) and snapshot.__dataclass_params__.frozen


def test_zero_declaration_is_exact_and_accepts_no_sources() -> None:
    declaration = AuditBudgetDeclaration(
        "operatebench.domain-generated-audit-budget.v1",
        "0" * 64,
        "opinst_zero",
        0,
        (),
    )
    assert AuditBudgetRuntime(declaration).snapshot().operation_committed_count == 0


@pytest.mark.parametrize(
    "declaration",
    [
        AuditBudgetDeclaration.__new__(AuditBudgetDeclaration),
        ("wrong", "0" * 64, "opinst_test", 0, ()),
        (
            "operatebench.domain-generated-audit-budget.v1",
            "z" * 64,
            "opinst_test",
            0,
            (),
        ),
        (
            "operatebench.domain-generated-audit-budget.v1",
            "0" * 64,
            "bad instance!",
            0,
            (),
        ),
        (
            "operatebench.domain-generated-audit-budget.v1",
            "0" * 64,
            "opinst_test",
            True,
            (),
        ),
        (
            "operatebench.domain-generated-audit-budget.v1",
            "0" * 64,
            "opinst_test",
            1,
            (),
        ),
    ],
)
def test_declaration_rejects_wrong_contract_identity_type_and_aggregate(
    declaration: object,
) -> None:
    if type(declaration) is tuple:
        with pytest.raises((TypeError, ValueError)):
            AuditBudgetDeclaration(*declaration)
    else:
        with pytest.raises((TypeError, ValueError)):
            AuditBudgetRuntime(declaration)  # type: ignore[arg-type]


def test_declaration_rejects_unsorted_duplicate_sources_and_checked_product() -> None:
    one = AuditBudgetSource("z.audit", "scheduled_audit_event_intent", 1, 1)
    two = AuditBudgetSource("a.audit", "immediate_audit_event_intent", 1, 1)
    with pytest.raises(ValueError):
        AuditBudgetDeclaration(
            "operatebench.domain-generated-audit-budget.v1",
            "0" * 64,
            "opinst_test",
            2,
            (one, two),
        )
    with pytest.raises(ValueError):
        AuditBudgetDeclaration(
            "operatebench.domain-generated-audit-budget.v1",
            "0" * 64,
            "opinst_test",
            2,
            (one, one),
        )
    with pytest.raises(ValueError):
        AuditBudgetSource(
            "test.audit", "scheduled_audit_event_intent", 9_223_372_036_854_775_807, 2
        )


def test_abort_after_partial_consumption_discards_batch_and_runtime_remains_usable() -> (
    None
):
    runtime = _runtime()
    transition = runtime.begin_transition("test.first", 0)
    reservation = transition.reserve_batch(
        "test.audit",
        "scheduled_audit_event_intent",
        ("test.audit.one", "test.audit.two"),
    )
    reservation.consume("test.audit.one")
    transition.abort()
    assert runtime.snapshot().operation_committed_count == 0

    retry = runtime.begin_transition("test.retry", 1)
    retry_reservation = retry.reserve_batch(
        "test.audit", "scheduled_audit_event_intent", ("test.audit.one",)
    )
    retry_reservation.consume("test.audit.one")
    retry.commit()
    assert runtime.snapshot().operation_committed_count == 1


def test_invalid_second_batch_poisons_transition_and_publishes_none() -> None:
    runtime = _runtime(occurrences=2)
    transition = runtime.begin_transition("test.poison", 7)
    reservation = transition.reserve_batch(
        "test.audit", "scheduled_audit_event_intent", ("test.audit.valid",)
    )
    reservation.consume("test.audit.valid")

    with pytest.raises(AuditIntegrityHalt) as caught:
        transition.reserve_batch(
            "unknown.audit",
            "scheduled_audit_event_intent",
            ("test.audit.invalid",),
        )

    halt = caught.value
    assert halt.contract_id == "operatebench.domain-generated-audit-budget.v1"
    assert halt.integrity_code == "DOMAIN_AUDIT_CONTRACT_VIOLATION"
    assert halt.refusal_class == "unknown_source_or_kind"
    assert halt.source_id == "unknown.audit"
    assert halt.requested_count == 1
    assert halt.operation_committed_before == 0
    assert halt.source_committed_before is None
    assert halt.declared_max_events == 4
    assert halt.transition_id == "test.poison"
    assert halt.invocation_index == 7
    assert runtime.snapshot().operation_committed_count == 0
    with pytest.raises(AuditIntegrityHalt) as frozen:
        runtime.begin_transition("test.later", 8)
    assert frozen.value is not halt
    assert dataclasses.astuple(frozen.value) == dataclasses.astuple(halt)


def test_reservation_capability_is_bound_single_use_and_non_transferable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    transition = runtime.begin_transition("test.capability", 0)
    reservation = transition.reserve_batch(
        "test.audit", "scheduled_audit_event_intent", ("test.audit.one",)
    )
    for transfer in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError):
            transfer(reservation)

    original_pid = reservation.process_id
    monkeypatch.setattr(audit_budget_module.os, "getpid", lambda: original_pid + 1)
    with pytest.raises(RuntimeError, match="another process"):
        reservation.consume("test.audit.one")


def test_source_occurrence_overrun_is_atomic_budget_exceeded() -> None:
    runtime = _runtime(occurrences=1)
    first = runtime.begin_transition("test.first", 0)
    token = first.reserve_batch(
        "test.audit", "scheduled_audit_event_intent", ("test.audit.first",)
    )
    token.consume("test.audit.first")
    first.commit()

    with pytest.raises(AuditIntegrityHalt) as caught:
        runtime.begin_transition("test.second", 1).reserve_batch(
            "test.audit", "scheduled_audit_event_intent", ("test.audit.second",)
        )
    assert caught.value.integrity_code == "DOMAIN_AUDIT_BUDGET_EXCEEDED"
    assert caught.value.refusal_class == "valid_reservation_above_declared_source"
    assert runtime.snapshot().operation_committed_count == 1


@pytest.mark.parametrize(
    "intent_ids",
    [(), ("test.audit.same", "test.audit.same"), ("test.audit.1",) * 3],
)
def test_invalid_batch_shape_halts_without_staging(intent_ids: tuple[str, ...]) -> None:
    runtime = _runtime()
    transition = runtime.begin_transition("test.invalid", 0)
    with pytest.raises(AuditIntegrityHalt) as caught:
        transition.reserve_batch("test.audit", "scheduled_audit_event_intent", intent_ids)
    assert caught.value.integrity_code == "DOMAIN_AUDIT_CONTRACT_VIOLATION"
    assert caught.value.refusal_class == "invalid_batch_or_identity"
    assert runtime.snapshot().committed_intents == ()


def test_unconsumed_commit_and_malformed_counter_have_exact_classes() -> None:
    runtime = _runtime()
    transition = runtime.begin_transition("test.unconsumed", 0)
    transition.reserve_batch(
        "test.audit", "scheduled_audit_event_intent", ("test.audit.one",)
    )
    with pytest.raises(AuditIntegrityHalt) as unconsumed:
        transition.commit()
    assert unconsumed.value.refusal_class == "reservation_lifecycle_violation"

    corrupted = _runtime()
    _inject_state(
        corrupted,
        dataclasses.replace(corrupted._state, source_counts=(("test.audit", True),)),
    )
    with pytest.raises(AuditIntegrityHalt) as counter:
        corrupted.begin_transition("test.counter", 0)
    assert counter.value.integrity_code == "DOMAIN_AUDIT_COUNTER_INTEGRITY"
    assert counter.value.refusal_class == "malformed_runtime_counter"
    assert counter.value.operation_committed_before == 0


def test_duplicate_committed_intent_persists_across_invocation_labels() -> None:
    runtime = _runtime(occurrences=2)
    first = runtime.begin_transition("test.invoke-one", 1)
    reservation = first.reserve_batch(
        "test.audit", "scheduled_audit_event_intent", ("test.audit.same",)
    )
    reservation.consume("test.audit.same")
    first.commit()
    with pytest.raises(AuditIntegrityHalt) as duplicate:
        runtime.begin_transition("test.reopen", 99).reserve_batch(
            "test.audit", "scheduled_audit_event_intent", ("test.audit.same",)
        )
    assert duplicate.value.refusal_class == "invalid_batch_or_identity"
    assert runtime.snapshot().operation_committed_count == 1


def test_finality_produces_exact_success_or_finding_then_error_summary() -> None:
    clean = _runtime()
    clean.mark_finality()
    clean_records = clean.terminal_evidence()
    assert [row.record_type for row in clean_records] == ["terminal_summary"]
    assert clean_records[-1].result == "SUCCESS"
    assert clean_records[-1].integrity_code is None
    assert clean.terminal_evidence() == clean_records

    halted = _runtime()
    halted.mark_finality()
    before = halted.snapshot()
    with pytest.raises(AuditIntegrityHalt) as caught:
        halted.begin_transition("test.after-finality", 4)
    assert caught.value.refusal_class == "post_finality_attempt"
    assert halted.snapshot().operation_committed_count == before.operation_committed_count
    records = halted.terminal_evidence()
    assert [row.record_type for row in records] == [
        "integrity_finding",
        "terminal_summary",
    ]
    assert records[-1].result == "ERROR"
    assert records[-1].integrity_code == "DOMAIN_AUDIT_CONTRACT_VIOLATION"


@pytest.mark.parametrize(
    ("transition_id", "invocation_index"), [("bad id!", 0), ("ok", True)]
)
def test_malformed_transition_scope_is_a_programmer_boundary_error(
    transition_id: str, invocation_index: object
) -> None:
    runtime = _runtime()
    with pytest.raises((TypeError, ValueError)):
        runtime.begin_transition(transition_id, invocation_index)  # type: ignore[arg-type]
    assert not runtime.snapshot().halted


def test_business_exception_aborts_partial_consumption_without_halting() -> None:
    runtime = _runtime()
    with (
        pytest.raises(RuntimeError, match="business refusal"),
        runtime.begin_transition("test.exception", 0) as transition,
    ):
        reservation = transition.reserve_batch(
            "test.audit",
            "scheduled_audit_event_intent",
            ("test.audit.one", "test.audit.two"),
        )
        reservation.consume("test.audit.one")
        raise RuntimeError("business refusal")
    assert runtime.snapshot().operation_committed_count == 0
    assert not runtime.snapshot().halted
    runtime.begin_transition("test.still-usable", 1).abort()


def test_foreign_missing_out_of_order_and_double_consumption_halt() -> None:
    for attack in ("foreign", "missing", "out-of-order", "double"):
        runtime = _runtime()
        transition = runtime.begin_transition(f"test.{attack}", 0)
        reservation = transition.reserve_batch(
            "test.audit",
            "scheduled_audit_event_intent",
            ("test.audit.one", "test.audit.two"),
        )
        with pytest.raises(AuditIntegrityHalt) as caught:
            if attack == "foreign":
                foreign = object.__new__(AuditIntent)
                transition.consume(foreign, "test.audit.one")  # type: ignore[arg-type]
            elif attack == "missing":
                transition.consume(None, "test.audit.one")  # type: ignore[arg-type]
            elif attack == "out-of-order":
                reservation.consume("test.audit.two")
            else:
                reservation.consume("test.audit.one")
                reservation.consume("test.audit.two")
                reservation.consume("test.audit.two")
        assert caught.value.refusal_class == "reservation_lifecycle_violation"
        assert runtime.snapshot().operation_committed_count == 0


def test_private_module_does_not_export_or_import_maintenance(tmp_path: Path) -> None:
    assert not hasattr(core_package, "AuditBudgetRuntime")

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (
            str(Path(__file__).resolve().parents[1] / "src"),
            inherited_pythonpath,
        )
        if path
    )
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

maintenance = "operatebench.domains.lettings.maintenance"

class MaintenancePoison:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == maintenance or fullname.startswith(maintenance + "."):
            raise AssertionError(f"audit budget imported {fullname}")
        return None

sys.meta_path.insert(0, MaintenancePoison())

from operatebench.core.audit_budget import AuditBudgetDeclaration, AuditBudgetRuntime

declaration = AuditBudgetDeclaration(
    "operatebench.domain-generated-audit-budget.v1",
    "0" * 64,
    "opinst_import_probe",
    0,
    (),
)
AuditBudgetRuntime(declaration)
assert not any(
    name == maintenance or name.startswith(maintenance + ".") for name in sys.modules
)
""",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr


def test_full_state_rederivation_is_constant_per_exact_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = AuditBudgetSource("test.audit", "immediate_audit_event_intent", 1, 128)
    runtime = AuditBudgetRuntime(
        AuditBudgetDeclaration(
            "operatebench.domain-generated-audit-budget.v1",
            "e" * 64,
            "opinst_complexity",
            128,
            (source,),
        )
    )
    full_validations = 0
    original = AuditBudgetRuntime._integrity_valid_locked

    def counted(
        self: AuditBudgetRuntime,
        state: object = None,
        *,
        deep_active: bool = True,
    ) -> bool:
        nonlocal full_validations
        full_validations += 1
        return original(self, state, deep_active=deep_active)  # type: ignore[arg-type]

    monkeypatch.setattr(AuditBudgetRuntime, "_integrity_valid_locked", counted)
    intent_ids = tuple(f"test.complexity.intent-{index}" for index in range(128))
    transition = runtime.begin_transition("test.complexity", 0)
    reservation = transition.reserve_batch(
        "test.audit", "immediate_audit_event_intent", intent_ids
    )
    before_consumes = full_validations
    for intent_id in intent_ids:
        reservation.consume(intent_id)
    consume_validations = full_validations - before_consumes
    transition.commit()

    # Full committed-state rederivation belongs to batch/transition boundaries,
    # never the individual-intent hot path.
    assert consume_validations == 0
    assert full_validations <= 8
    assert runtime.snapshot().operation_committed_count == len(intent_ids)


def test_protocol_capacity_retains_every_row_without_truncation() -> None:
    source = AuditBudgetSource("test.audit", "immediate_audit_event_intent", 1, 99_998)
    runtime = AuditBudgetRuntime(
        AuditBudgetDeclaration(
            "operatebench.domain-generated-audit-budget.v1",
            "f" * 64,
            "opinst_capacity",
            99_998,
            (source,),
        )
    )
    intent_ids = tuple(f"test.audit.intent-{index}" for index in range(99_998))
    transition = runtime.begin_transition("test.capacity", 0)
    reservation = transition.reserve_batch(
        "test.audit", "immediate_audit_event_intent", intent_ids
    )
    for intent_id in intent_ids:
        reservation.consume(intent_id)
    transition.commit()
    runtime.mark_finality()
    assert len(runtime.terminal_evidence()) == 99_999

    with pytest.raises(AuditIntegrityHalt):
        runtime.begin_transition("test.after-capacity-finality", 1)
    records = runtime.terminal_evidence()
    assert len(records) == 100_000
    assert records[-2].record_type == "integrity_finding"
    assert records[-1].record_type == "terminal_summary"


def test_qodo_reexecution_is_noop_until_representable_reservation_halts() -> None:
    runtime = _runtime(occurrences=2, fanout=1)
    first = runtime.begin_transition("qodo.retry", 3)
    reservation = first.reserve_batch(
        "test.audit", "scheduled_audit_event_intent", ("qodo.original",)
    )
    reservation.consume("qodo.original")
    first.commit()

    before = runtime.snapshot()
    retry = runtime.begin_transition("qodo.retry", 4)
    retry.commit()
    assert runtime.snapshot() == before

    retry = runtime.begin_transition("qodo.retry", 5)
    with pytest.raises(AuditIntegrityHalt) as caught:
        retry.reserve_batch(
            "test.audit", "scheduled_audit_event_intent", ("qodo.reexecuted",)
        )
    assert dataclasses.astuple(caught.value)[1:] == (
        "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "invalid_batch_or_identity",
        "test.audit",
        0,
        1,
        1,
        2,
        "qodo.retry",
        5,
    )


def test_qodo_many_one_event_transitions_do_not_fully_rederive_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = AuditBudgetSource("test.audit", "scheduled_audit_event_intent", 1_024, 1)
    runtime = AuditBudgetRuntime(
        AuditBudgetDeclaration(
            "operatebench.domain-generated-audit-budget.v1",
            "d" * 64,
            "opinst_qodo_many",
            1_024,
            (source,),
        )
    )
    validations = 0
    inserts = 0
    visits = 0
    original = AuditBudgetRuntime._integrity_valid_locked
    original_insert = audit_budget_module._treap_insert

    def counted(self: AuditBudgetRuntime, *args: object, **kwargs: object) -> bool:
        nonlocal validations
        validations += 1
        return original(self, *args, **kwargs)  # type: ignore[arg-type]

    def counted_insert(*args: object, **kwargs: object) -> object:
        nonlocal inserts
        inserts += 1
        return original_insert(*args, **kwargs)

    def counted_visit() -> None:
        nonlocal visits
        visits += 1

    monkeypatch.setattr(AuditBudgetRuntime, "_integrity_valid_locked", counted)
    monkeypatch.setattr(audit_budget_module, "_treap_insert", counted_insert)
    monkeypatch.setattr(audit_budget_module, "_treap_visit", counted_visit)
    for index in range(1_024):
        scope = runtime.begin_transition(f"qodo.transition-{index}", index)
        intent_id = f"qodo.intent-{index}"
        reservation = scope.reserve_batch(
            "test.audit", "scheduled_audit_event_intent", (intent_id,)
        )
        reservation.consume(intent_id)
        scope.commit()
    assert validations == 0
    assert inserts == 10 * 1_024
    assert visits <= 256 * 1_024
    assert runtime.snapshot().operation_committed_count == 1_024


def test_qodo_invalid_nonce_publication_is_rollback_proof_and_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    scope = runtime.begin_transition("qodo.atomic", 0)
    original = runtime._state
    monkeypatch.setattr(audit_budget_module.secrets, "token_hex", lambda _: "invalid")
    with pytest.raises(ValueError, match="internal reservation nonce"):
        scope.reserve_batch(
            "test.audit", "scheduled_audit_event_intent", ("qodo.atomic.one",)
        )
    assert runtime._state is original
    monkeypatch.undo()
    reservation = scope.reserve_batch(
        "test.audit", "scheduled_audit_event_intent", ("qodo.atomic.one",)
    )
    reservation.consume("qodo.atomic.one")
    scope.commit()
    assert runtime.snapshot().operation_committed_count == 1


def test_qodo_duplicate_active_nonce_is_atomic_and_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = (
        AuditBudgetSource("a.audit", "scheduled_audit_event_intent", 1, 1),
        AuditBudgetSource("b.audit", "scheduled_audit_event_intent", 1, 1),
    )
    runtime = AuditBudgetRuntime(
        AuditBudgetDeclaration(
            "operatebench.domain-generated-audit-budget.v1",
            "a" * 64,
            "opinst_nonce_collision",
            2,
            sources,
        )
    )
    scope = runtime.begin_transition("qodo.nonce-collision", 0)
    first = scope.reserve_batch("a.audit", sources[0].counted_kind, ("qodo.a",))
    original = runtime._control
    monkeypatch.setattr(audit_budget_module.secrets, "token_hex", lambda _: first._nonce)
    with pytest.raises(ValueError, match="duplicate internal reservation nonce"):
        scope.reserve_batch("b.audit", sources[1].counted_kind, ("qodo.b",))
    assert runtime._control is original
    monkeypatch.undo()
    second = scope.reserve_batch("b.audit", sources[1].counted_kind, ("qodo.b",))
    first.consume("qodo.a")
    second.consume("qodo.b")
    scope.commit()
    assert runtime.snapshot().operation_committed_count == 2


def test_qodo_persistent_index_failure_leaves_exact_control_and_commit_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(fanout=1)
    scope = runtime.begin_transition("qodo.index-failure", 0)
    reservation = scope.reserve_batch(
        "test.audit", "scheduled_audit_event_intent", ("qodo.index",)
    )
    reservation.consume("qodo.index")
    original = runtime._control

    def fail_insert(*args: object, **kwargs: object) -> object:
        raise RuntimeError("injected persistent-index failure")

    monkeypatch.setattr(audit_budget_module, "_treap_insert", fail_insert)
    with pytest.raises(RuntimeError, match="persistent-index failure"):
        scope.commit()
    assert runtime._control is original
    monkeypatch.undo()
    scope.commit()
    assert runtime.snapshot().operation_committed_count == 1


@pytest.mark.parametrize("attack", ["index", "history-order", "history-count"])
def test_qodo_full_oracle_rejects_persistent_history_corruption(attack: str) -> None:
    runtime = _runtime(occurrences=1, fanout=2)
    scope = runtime.begin_transition("qodo.corruption", 0)
    reservation = scope.reserve_batch(
        "test.audit",
        "scheduled_audit_event_intent",
        ("qodo.corruption-a", "qodo.corruption-b"),
    )
    reservation.consume("qodo.corruption-a")
    reservation.consume("qodo.corruption-b")
    scope.commit()
    state = runtime._state
    assert state.committed is not None
    if attack == "index":
        replacement = dataclasses.replace(
            state, committed_ids=state.committed_ids.set("qodo.forged", True)
        )
    elif attack == "history-order":
        replacement = dataclasses.replace(
            state,
            committed=dataclasses.replace(
                state.committed, rows=tuple(reversed(state.committed.rows))
            ),
        )
    else:
        replacement = dataclasses.replace(
            state,
            committed=dataclasses.replace(
                state.committed, count=state.committed.count + 1
            ),
        )
    _inject_state(runtime, replacement, coherent=True)
    with pytest.raises(AuditIntegrityHalt) as caught:
        runtime.snapshot()
    assert caught.value.refusal_class == "malformed_runtime_counter"


def test_finality_and_transition_admission_are_one_atomic_cut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    entered = threading.Event()
    release = threading.Event()
    original = AuditTransitionScope.__init__

    def paused_init(self: AuditTransitionScope, *args: object) -> None:
        entered.set()
        assert release.wait(2)
        original(self, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(AuditTransitionScope, "__init__", paused_init)
    result: list[object] = []
    finality_errors: list[BaseException] = []
    admission = threading.Thread(
        target=lambda: result.append(runtime.begin_transition("test.atomic", 0))
    )
    admission.start()
    assert entered.wait(2)

    def finalize() -> None:
        try:
            runtime.mark_finality()
        except BaseException as exc:
            finality_errors.append(exc)

    finality = threading.Thread(target=finalize)
    finality.start()
    finality.join(0.1)
    release.set()
    admission.join(2)
    finality.join(2)
    assert not (runtime.snapshot().final and result)
    assert len(result) == 1
    assert len(finality_errors) == 1


def test_reservation_is_frozen_opaque_authority_and_digests_are_runtime_owned() -> None:
    runtime = _runtime()
    scope = runtime.begin_transition("test.capability", 0)
    reservation = scope.reserve_batch(
        "test.audit",
        "scheduled_audit_event_intent",
        ("test.audit.one", "test.audit.two"),
    )
    assert dataclasses.is_dataclass(reservation)
    assert reservation.__dataclass_params__.frozen
    assert tuple(field.name for field in dataclasses.fields(reservation)) == (
        "_runtime",
        "_transition_identity",
        "_nonce",
        "_creator_pid",
    )
    for name in ("_next", "_digests", "_intents"):
        with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
            setattr(reservation, name, 2)
    reservation.consume("test.audit.one")
    reservation.consume("test.audit.two")
    scope.commit()
    assert [row.intent_digest_sha256 for row in runtime.snapshot().committed_intents] != [
        "0" * 64,
        "f" * 64,
    ]


def test_runtime_scope_and_reservation_refuse_copy_deepcopy_and_pickle() -> None:
    runtime = _runtime()
    scope = runtime.begin_transition("test.transfer", 0)
    reservation = scope.reserve_batch(
        "test.audit", "scheduled_audit_event_intent", ("test.audit.one",)
    )
    for authority in (runtime, scope, reservation):
        for transfer in (copy.copy, copy.deepcopy, pickle.dumps):
            with pytest.raises(TypeError):
                transfer(authority)


def test_runtime_slots_are_sealed_before_any_ordinary_mutation() -> None:
    runtime = _runtime(occurrences=2, fanout=1)
    original_state = runtime._state
    replacement = dataclasses.replace(original_state, final=True)
    for name, value in (
        ("_creator_pid", os.getpid() + 1),
        ("_declaration", _runtime()._declaration),
        ("_sources", ()),
        ("_control", dataclasses.replace(runtime._control, state=replacement)),
        ("_state", replacement),
        ("_lock", threading.RLock()),
        ("_identity", "0" * 64),
        ("_trusted_state", replacement),
        ("_halt_authority", object()),
    ):
        with pytest.raises(AttributeError, match="sealed"):
            setattr(runtime, name, value)
        assert runtime._state is original_state
    for name in (
        "_creator_pid",
        "_declaration",
        "_sources",
        "_control",
        "_state",
        "_lock",
        "_identity",
        "_trusted_state",
        "_halt_authority",
    ):
        with pytest.raises(AttributeError, match="sealed"):
            delattr(runtime, name)
        assert runtime._state is original_state

    scope = runtime.begin_transition("seal.parent", 0)
    reservation = scope.reserve_batch(
        "test.audit", "scheduled_audit_event_intent", ("seal.parent.intent",)
    )
    reservation.consume("seal.parent.intent")
    scope.commit()
    assert runtime.snapshot().operation_committed_count == 1


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is unavailable")
def test_child_cannot_reset_runtime_pid_and_parent_remains_usable() -> None:
    runtime = _runtime(occurrences=1, fanout=1)
    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:
        os.close(read_fd)
        result = ""
        try:
            try:
                runtime._creator_pid = os.getpid()
            except AttributeError as exc:
                result += f"sealed:{exc};"
            try:
                runtime.begin_transition("seal.child", 0)
            except RuntimeError as exc:
                result += f"pid:{exc}"
            os.write(write_fd, result.encode("ascii"))
        finally:
            os.close(write_fd)
            os._exit(0)
    os.close(write_fd)
    child_result = os.read(read_fd, 4096).decode("ascii")
    os.close(read_fd)
    waited, status = os.waitpid(child_pid, 0)
    assert waited == child_pid and os.waitstatus_to_exitcode(status) == 0
    assert child_result == (
        "sealed:audit budget runtime is sealed;"
        "pid:audit budget runtime belongs to another process"
    )

    scope = runtime.begin_transition("seal.parent", 0)
    reservation = scope.reserve_batch(
        "test.audit", "scheduled_audit_event_intent", ("seal.parent.intent",)
    )
    reservation.consume("seal.parent.intent")
    scope.commit()
    assert runtime.snapshot().operation_committed_count == 1


def test_committed_transition_identity_cannot_reopen_or_erase_evidence() -> None:
    runtime = _runtime(occurrences=2, fanout=1)
    first = runtime.begin_transition("transition.once", 4)
    first_reservation = first.reserve_batch(
        "test.audit", "scheduled_audit_event_intent", ("transition.first",)
    )
    first_reservation.consume("transition.first")
    first.commit()
    before = runtime.snapshot()

    retry = runtime.begin_transition("transition.once", 99)
    retry.commit()
    assert runtime.snapshot() == before
    assert not runtime.snapshot().halted

    later = runtime.begin_transition("transition.fresh", 5)
    later_reservation = later.reserve_batch(
        "test.audit", "scheduled_audit_event_intent", ("transition.second",)
    )
    later_reservation.consume("transition.second")
    later.commit()
    assert tuple(row.intent_id for row in runtime.snapshot().committed_intents) == (
        "transition.first",
        "transition.second",
    )


def test_explicit_active_cursor_corruption_cannot_forge_consumption() -> None:
    runtime = _runtime(fanout=1)
    scope = runtime.begin_transition("cursor.forge", 0)
    scope.reserve_batch(
        "test.audit", "scheduled_audit_event_intent", ("cursor.unconsumed",)
    )
    active = runtime._state.active
    assert active is not None
    forged_batch = dataclasses.replace(active.batches[0], next_index=1)
    forged_active = dataclasses.replace(active, batches=(forged_batch,))
    _inject_state(runtime, dataclasses.replace(runtime._state, active=forged_active))

    with pytest.raises(AuditIntegrityHalt) as caught:
        scope.commit()
    assert caught.value.refusal_class == "malformed_runtime_counter"
    assert runtime.snapshot().committed_intents == ()
    assert [row.record_type for row in runtime.terminal_evidence()] == [
        "integrity_finding",
        "terminal_summary",
    ]


def test_occurrence_corruption_halts_with_frozen_zero_null_counter_row() -> None:
    runtime = _runtime()
    _inject_state(
        runtime,
        dataclasses.replace(runtime._state, source_occurrences=(("test.audit", 1),)),
    )
    with pytest.raises(AuditIntegrityHalt) as caught:
        runtime.snapshot()
    assert dataclasses.astuple(caught.value)[1:] == (
        "DOMAIN_AUDIT_COUNTER_INTEGRITY",
        "malformed_runtime_counter",
        None,
        0,
        0,
        None,
        2,
        None,
        None,
    )


def test_commit_publication_failure_cannot_partially_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    scope = runtime.begin_transition("test.publish", 0)
    reservation = scope.reserve_batch(
        "test.audit",
        "scheduled_audit_event_intent",
        ("test.audit.one", "test.audit.two"),
    )
    reservation.consume("test.audit.one")
    reservation.consume("test.audit.two")

    def fail_publication(state: object) -> object:
        raise RuntimeError("injected publication failure")

    monkeypatch.setattr(audit_budget_module, "_publish_state", fail_publication)
    with pytest.raises(RuntimeError, match="injected publication failure"):
        scope.commit()
    assert runtime.snapshot().operation_committed_count == 0
    assert runtime.snapshot().committed_intents == ()
    monkeypatch.undo()
    scope.commit()
    assert runtime.snapshot().operation_committed_count == 2


def test_halt_publication_failure_cannot_orphan_authority_or_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    scope = runtime.begin_transition("test.halt-publish", 0)
    original = runtime._state

    def fail_publication(authority: object, state: object) -> object:
        raise RuntimeError("injected halt publication failure")

    monkeypatch.setattr(audit_budget_module, "_publish_halt", fail_publication)
    with pytest.raises(RuntimeError, match="injected halt publication failure"):
        scope.reserve_batch(
            "unknown.audit",
            "scheduled_audit_event_intent",
            ("test.audit.one",),
        )
    assert runtime._state is original
    assert runtime._trusted_state is original
    assert runtime._halt_authority is None


def test_halt_record_does_not_retain_rejected_payload_or_traceback() -> None:
    class Payload:
        pass

    runtime = _runtime()
    scope = runtime.begin_transition("test.payload", 0)
    payload = Payload()
    reference = weakref.ref(payload)
    attempted = (payload,)
    with pytest.raises((TypeError, ValueError)):
        scope.reserve_batch(
            "test.audit",
            "scheduled_audit_event_intent",
            attempted,  # type: ignore[arg-type]
        )
    del attempted, payload
    gc.collect()
    assert reference() is None
    assert not runtime.snapshot().halted

    def assigned_halt() -> tuple[AuditBudgetRuntime, weakref.ReferenceType[Payload]]:
        poisoned = _runtime()
        transition = poisoned.begin_transition("test.assigned-payload", 0)
        hostile = Payload()
        retained = weakref.ref(hostile)
        attempted_list = [hostile]
        with contextlib.suppress(AuditIntegrityHalt):
            transition.reserve_batch(
                "test.audit",
                "scheduled_audit_event_intent",
                attempted_list,  # type: ignore[arg-type]
            )
        return poisoned, retained

    poisoned, retained = assigned_halt()
    gc.collect()
    assert retained() is None
    assert poisoned.snapshot().halted
    assert [row.record_type for row in poisoned.terminal_evidence()] == [
        "integrity_finding",
        "terminal_summary",
    ]


def test_aggregate_overrun_precedes_source_overrun_exactly_as_frozen_golden() -> None:
    sources = (
        AuditBudgetSource("approval.audit", "scheduled_audit_event_intent", 1, 2),
        AuditBudgetSource("note.audit", "immediate_audit_event_intent", 1, 1),
    )
    runtime = AuditBudgetRuntime(
        AuditBudgetDeclaration(
            "operatebench.domain-generated-audit-budget.v1",
            "2" * 64,
            "opinst_one_over",
            3,
            sources,
        )
    )
    for index, (source, kind, ids) in enumerate(
        (
            ("approval.audit", sources[0].counted_kind, ("approval.one", "approval.two")),
            ("note.audit", sources[1].counted_kind, ("note.one",)),
        )
    ):
        scope = runtime.begin_transition(f"transition.{index}", index)
        reservation = scope.reserve_batch(source, kind, ids)
        for intent_id in ids:
            reservation.consume(intent_id)
        scope.commit()
    with pytest.raises(AuditIntegrityHalt) as caught:
        runtime.begin_transition("transition.over", 2).reserve_batch(
            "note.audit", sources[1].counted_kind, ("note.two",)
        )
    assert caught.value.refusal_class == "valid_reservation_above_declared_aggregate"
    assert caught.value.requested_count == 1
    assert caught.value.operation_committed_before == 3
    assert caught.value.source_committed_before == 1


def test_assigned_refusal_rows_pin_registry_matrix_and_programmer_boundary() -> None:
    runtime = _runtime()
    scope = runtime.begin_transition("test.matrix", 7)
    with pytest.raises(AuditIntegrityHalt) as invalid:
        scope.reserve_batch(
            "test.audit",
            "scheduled_audit_event_intent",
            ("test.audit.same", "test.audit.same"),
        )
    assert dataclasses.astuple(invalid.value)[2:] == (
        "invalid_batch_or_identity",
        "test.audit",
        0,
        0,
        0,
        2,
        "test.matrix",
        7,
    )

    final = _runtime()
    final.mark_finality()
    with pytest.raises(AuditIntegrityHalt) as post:
        final.begin_transition("test.after", 9)
    assert post.value.source_id is None
    assert post.value.transition_id is None
    assert post.value.invocation_index is None
    assert post.value.requested_count == 0

    programmer = _runtime()
    with pytest.raises(ValueError):
        programmer.begin_transition("Bad:Transition", 0)
    assert not programmer.snapshot().halted


def test_terminal_evidence_validates_frozen_schema_and_pins_canonical_digest() -> None:
    runtime = _runtime(fanout=1)
    scope = runtime.begin_transition("transition.exact", 0)
    reservation = scope.reserve_batch(
        "test.audit", "scheduled_audit_event_intent", ("test.audit.one",)
    )
    reservation.consume("test.audit.one")
    scope.commit()
    runtime.mark_finality()
    records = runtime.terminal_evidence()
    assert (
        records[0].intent_digest_sha256
        == "ca1b9624173d665a40b9debd5bebec7969b340d4461c7757e975038e09ddc0e4"
    )
    schema_path = (
        Path(__file__).parents[1]
        / "docs/schemas/domain-generated-audit-budget-evidence-v1.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    envelope = {
        "schema": "operatebench.domain_generated_audit_budget_evidence.v1",
        "schema_version": 1,
        "contract_id": "operatebench.domain-generated-audit-budget.v1",
        "operation_core_content_digest_sha256": "1" * 64,
        "operation_instance_id": "opinst_test",
        "declared_max_events": 1,
        "records": [dataclasses.asdict(row) for row in records],
    }
    Draft202012Validator(schema).validate(envelope)


def test_multi_source_and_repeated_source_transition_occurrences_rederive() -> None:
    sources = (
        AuditBudgetSource("a.audit", "scheduled_audit_event_intent", 2, 1),
        AuditBudgetSource("b.audit", "immediate_audit_event_intent", 1, 2),
    )
    runtime = AuditBudgetRuntime(
        AuditBudgetDeclaration(
            "operatebench.domain-generated-audit-budget.v1",
            "3" * 64,
            "opinst_multi",
            4,
            sources,
        )
    )
    first = runtime.begin_transition("transition.first", 0)
    a = first.reserve_batch("a.audit", sources[0].counted_kind, ("a.one",))
    b = first.reserve_batch("b.audit", sources[1].counted_kind, ("b.one", "b.two"))
    for reservation, ids in ((a, ("a.one",)), (b, ("b.one", "b.two"))):
        for intent_id in ids:
            reservation.consume(intent_id)
    first.commit()
    second = runtime.begin_transition("transition.second", 1)
    a2 = second.reserve_batch("a.audit", sources[0].counted_kind, ("a.two",))
    a2.consume("a.two")
    second.commit()
    assert runtime.snapshot().source_occurrence_counts == (("a.audit", 2), ("b.audit", 1))


def test_registry_runtime_refusal_subset_is_independently_literal_pinned() -> None:
    root = Path(__file__).parents[1]
    docs = json.loads(
        (root / "docs/schemas/domain-generated-audit-budget-registry-v1.json").read_text()
    )
    package = json.loads(
        (
            root
            / "src/operatebench/resources/identity"
            / "domain-generated-audit-budget-registry-v1.json"
        ).read_text()
    )
    assert docs == package
    assigned = {
        row["refusal_class"]: (
            row["integrity_code"],
            row["source_id_rule"],
            row["requested_count_rule"],
            row["transition_id_rule"],
            row["pre_operation_count_rule"],
            row["pre_source_count_rule"],
        )
        for row in docs["refusal_mapping"]
        if row["stage"].startswith("assigned")
        and row["refusal_class"]
        not in {"source_or_aggregate_mismatch", "v1_capacity_exceeded"}
    }
    attempted = (
        "exact_attempted_source",
        "positive_exact_attempted_batch_size",
        "exact_attempted_transition",
        "exact_authoritative_committed_before",
        "exact_authoritative_source_committed_before",
    )
    assert assigned == {
        "unknown_source_or_kind": (
            "DOMAIN_AUDIT_CONTRACT_VIOLATION",
            attempted[0],
            attempted[1],
            attempted[2],
            attempted[3],
            "exact_authoritative_source_committed_before_or_null_if_unknown",
        ),
        "invalid_batch_or_identity": (
            "DOMAIN_AUDIT_CONTRACT_VIOLATION",
            attempted[0],
            "zero",
            attempted[2],
            attempted[3],
            attempted[4],
        ),
        "reservation_lifecycle_violation": (
            "DOMAIN_AUDIT_CONTRACT_VIOLATION",
            "exact_reservation_source",
            "positive_exact_reservation_size",
            "exact_reservation_transition",
            attempted[3],
            attempted[4],
        ),
        "post_finality_attempt": (
            "DOMAIN_AUDIT_CONTRACT_VIOLATION",
            "null",
            "zero",
            "null",
            attempted[3],
            "null",
        ),
        "checked_signed_int64_arithmetic_overflow": (
            "DOMAIN_AUDIT_COUNTER_INTEGRITY",
            *attempted,
        ),
        "malformed_runtime_counter": (
            "DOMAIN_AUDIT_COUNTER_INTEGRITY",
            "null",
            "zero",
            "null",
            "zero_untrusted_counter",
            "null",
        ),
        "valid_reservation_above_declared_aggregate": (
            "DOMAIN_AUDIT_BUDGET_EXCEEDED",
            *attempted,
        ),
        "valid_reservation_above_declared_source": (
            "DOMAIN_AUDIT_BUDGET_EXCEEDED",
            *attempted,
        ),
    }
    assert audit_budget_module._checked_add(1, 2) == 3
    assert audit_budget_module._checked_multiply(3, 4) == 12
    with pytest.raises(OverflowError):
        audit_budget_module._checked_add(9_223_372_036_854_775_807, 1)
    with pytest.raises(OverflowError):
        audit_budget_module._checked_multiply(9_223_372_036_854_775_807, 2)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda state: dataclasses.replace(state, committed_ids=frozenset({"forged"})),
        lambda state: dataclasses.replace(state, committed_digests=frozenset()),
        lambda state: dataclasses.replace(state, source_counts=(("test.audit", 0),)),
        lambda state: dataclasses.replace(state, source_occurrences=(("test.audit", 0),)),
        lambda state: dataclasses.replace(state, final=1),
        lambda state: dataclasses.replace(
            state, committed=list(audit_budget_module._flatten_history(state.committed))
        ),
        lambda state: dataclasses.replace(
            state,
            committed=dataclasses.replace(
                state.committed,
                rows=(
                    dataclasses.replace(
                        state.committed.rows[0], intent_digest_sha256="0" * 64
                    ),
                ),
            ),
        ),
        lambda state: dataclasses.replace(
            state,
            committed=dataclasses.replace(
                state.committed,
                rows=(
                    dataclasses.replace(
                        state.committed.rows[0], operation_committed_count=True
                    ),
                ),
            ),
        ),
    ],
)
def test_every_supported_entry_rejects_coherent_runtime_state_corruption(
    corrupt: object,
) -> None:
    runtime = _runtime(fanout=1)
    scope = runtime.begin_transition("test.integrity", 0)
    reservation = scope.reserve_batch(
        "test.audit", "scheduled_audit_event_intent", ("test.audit.one",)
    )
    reservation.consume("test.audit.one")
    scope.commit()
    _inject_state(runtime, corrupt(runtime._state))  # type: ignore[operator]
    with pytest.raises(AuditIntegrityHalt) as caught:
        runtime.snapshot()
    assert dataclasses.astuple(caught.value)[1:] == (
        "DOMAIN_AUDIT_COUNTER_INTEGRITY",
        "malformed_runtime_counter",
        None,
        0,
        0,
        None,
        1,
        None,
        None,
    )
    records = runtime.terminal_evidence()
    assert [row.record_type for row in records] == [
        "integrity_finding",
        "terminal_summary",
    ]
    # The frozen malformed-counter row says the authoritative pre-count is zero;
    # rows from the untrusted envelope cannot be published beside that claim.
    assert records[-1].committed_count == 0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda halt: dataclasses.replace(
            halt, integrity_code="DOMAIN_AUDIT_COUNTER_INTEGRITY"
        ),
        lambda halt: dataclasses.replace(halt, refusal_class="unknown_source_or_kind"),
        lambda halt: dataclasses.replace(halt, source_id="other.audit"),
        lambda halt: dataclasses.replace(halt, requested_count=2),
        lambda halt: dataclasses.replace(halt, operation_committed_before=0),
        lambda halt: dataclasses.replace(halt, source_committed_before=0),
        lambda halt: dataclasses.replace(halt, transition_id="other.transition"),
        lambda halt: dataclasses.replace(halt, invocation_index=9),
        lambda halt: dataclasses.replace(halt, source_id="a" * 256),
        lambda halt: dataclasses.replace(halt, transition_id="a" * 256),
        lambda halt: dataclasses.replace(
            halt,
            witness=dataclasses.replace(halt.witness, cause="aggregate_limit"),
        ),
        lambda halt: dataclasses.replace(
            halt,
            integrity_code="DOMAIN_AUDIT_BUDGET_EXCEEDED",
            refusal_class="valid_reservation_above_declared_aggregate",
            witness=audit_budget_module._HaltWitness("aggregate_limit", (1, 0, 1, 2)),
        ),
        lambda halt: dataclasses.replace(
            halt,
            integrity_code="DOMAIN_AUDIT_COUNTER_INTEGRITY",
            refusal_class="checked_signed_int64_arithmetic_overflow",
            witness=audit_budget_module._HaltWitness(
                "checked_overflow", ("add", 9_223_372_036_854_775_807, 1)
            ),
        ),
        lambda halt: dataclasses.replace(
            halt,
            integrity_code="DOMAIN_AUDIT_CONTRACT_VIOLATION",
            refusal_class="reservation_lifecycle_violation",
            witness=audit_budget_module._HaltWitness(
                "reservation_lifecycle",
                (
                    "incomplete_commit",
                    "0" * 64,
                    "1" * 64,
                    "test.audit",
                    1,
                    "matrix.source-over",
                    1,
                ),
            ),
        ),
        lambda halt: dataclasses.replace(
            halt,
            integrity_code="DOMAIN_AUDIT_CONTRACT_VIOLATION",
            refusal_class="unknown_source_or_kind",
            witness=audit_budget_module._HaltWitness(
                "unknown_source_or_kind",
                ("immediate_audit_event_intent", True),
            ),
        ),
    ],
)
def test_halt_mutation_matrix_canonicalizes_without_untrusted_committed_evidence(
    mutate: object,
) -> None:
    runtime = _runtime(occurrences=1, fanout=2)
    first = runtime.begin_transition("matrix.source-first", 0)
    reservation = first.reserve_batch(
        "test.audit", "scheduled_audit_event_intent", ("matrix.source-one",)
    )
    reservation.consume("matrix.source-one")
    first.commit()
    second = runtime.begin_transition("matrix.source-over", 1)
    with pytest.raises(AuditIntegrityHalt):
        second.reserve_batch(
            "test.audit", "scheduled_audit_event_intent", ("matrix.source-two",)
        )
    assert runtime._state.halt is not None
    _inject_state(
        runtime,
        dataclasses.replace(runtime._state, halt=mutate(runtime._state.halt)),  # type: ignore[operator]
        coherent=True,
    )

    with pytest.raises(AuditIntegrityHalt) as caught:
        runtime.snapshot()
    assert dataclasses.astuple(caught.value)[1:] == (
        "DOMAIN_AUDIT_COUNTER_INTEGRITY",
        "malformed_runtime_counter",
        None,
        0,
        0,
        None,
        2,
        None,
        None,
    )
    records = runtime.terminal_evidence()
    assert [row.record_type for row in records] == [
        "integrity_finding",
        "terminal_summary",
    ]
    assert dataclasses.astuple(records[0])[2:] == (
        "DOMAIN_AUDIT_COUNTER_INTEGRITY",
        "malformed_runtime_counter",
        None,
        0,
        0,
        None,
        2,
        None,
        None,
    )
    assert records[-1].committed_count == 0

    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/domain-generated-audit-budget-evidence-v1.schema.json"
        ).read_text()
    )
    Draft202012Validator(schema).validate(
        {
            "schema": "operatebench.domain_generated_audit_budget_evidence.v1",
            "schema_version": 1,
            "contract_id": "operatebench.domain-generated-audit-budget.v1",
            "operation_core_content_digest_sha256": "1" * 64,
            "operation_instance_id": "opinst_test",
            "declared_max_events": 2,
            "records": [dataclasses.asdict(row) for row in records],
        }
    )


@pytest.mark.parametrize(
    "attack",
    [
        "invalid-cause",
        "unknown-kind",
        "lifecycle-nonce-cause",
        "aggregate-fields-source",
    ],
)
def test_halt_authority_rejects_coherent_reviewer_causal_laundering(
    attack: str,
) -> None:
    if attack == "invalid-cause":
        runtime = _runtime()
        scope = runtime.begin_transition("review.invalid", 0)
        with pytest.raises(AuditIntegrityHalt):
            scope.reserve_batch(
                "test.audit",
                "scheduled_audit_event_intent",
                ("review.same", "review.same"),
            )
        assert runtime._state.halt is not None
        substituted = dataclasses.replace(
            runtime._state.halt,
            witness=dataclasses.replace(
                runtime._state.halt.witness,
                facts=(
                    "duplicate_source_batch",
                    *runtime._state.halt.witness.facts[1:],
                ),
            ),
        )
    elif attack == "unknown-kind":
        runtime = _runtime()
        scope = runtime.begin_transition("review.unknown", 1)
        with pytest.raises(AuditIntegrityHalt):
            scope.reserve_batch(
                "test.audit",
                "immediate_audit_event_intent",
                ("review.unknown.one",),
            )
        assert runtime._state.halt is not None
        substituted = dataclasses.replace(
            runtime._state.halt,
            witness=dataclasses.replace(
                runtime._state.halt.witness,
                facts=("invented.kind", *runtime._state.halt.witness.facts[1:]),
            ),
        )
    elif attack == "lifecycle-nonce-cause":
        sources = (
            AuditBudgetSource("a.audit", "scheduled_audit_event_intent", 1, 1),
            AuditBudgetSource("b.audit", "scheduled_audit_event_intent", 1, 1),
        )
        runtime = AuditBudgetRuntime(
            AuditBudgetDeclaration(
                "operatebench.domain-generated-audit-budget.v1",
                "8" * 64,
                "opinst_reviewer_lifecycle",
                2,
                sources,
            )
        )
        first = runtime.begin_transition("review.lifecycle.first", 0)
        first_reservation = first.reserve_batch(
            "a.audit", sources[0].counted_kind, ("review.lifecycle.one",)
        )
        first_nonce = first_reservation._nonce
        first_reservation.consume("review.lifecycle.one")
        first.commit()
        second = runtime.begin_transition("review.lifecycle.second", 1)
        second.reserve_batch(
            "b.audit", sources[1].counted_kind, ("review.lifecycle.two",)
        )
        with pytest.raises(AuditIntegrityHalt):
            second.commit()
        assert runtime._state.halt is not None
        substituted = dataclasses.replace(
            runtime._state.halt,
            witness=dataclasses.replace(
                runtime._state.halt.witness,
                facts=(
                    "consume_order",
                    first_nonce,
                    "f" * 64,
                    *runtime._state.halt.witness.facts[3:],
                ),
            ),
        )
    else:
        sources = (
            AuditBudgetSource("a.audit", "scheduled_audit_event_intent", 1, 1),
            AuditBudgetSource("b.audit", "immediate_audit_event_intent", 1, 1),
        )
        runtime = AuditBudgetRuntime(
            AuditBudgetDeclaration(
                "operatebench.domain-generated-audit-budget.v1",
                "9" * 64,
                "opinst_reviewer_aggregate",
                2,
                sources,
            )
        )
        for index, source in enumerate(sources):
            scope = runtime.begin_transition(f"review.aggregate.{index}", index)
            reservation = scope.reserve_batch(
                source.source_id,
                source.counted_kind,
                (f"review.aggregate.intent-{index}",),
            )
            reservation.consume(f"review.aggregate.intent-{index}")
            scope.commit()
        with pytest.raises(AuditIntegrityHalt):
            runtime.begin_transition("review.aggregate.over", 2).reserve_batch(
                "b.audit", sources[1].counted_kind, ("review.aggregate.over",)
            )
        assert runtime._state.halt is not None
        substituted = dataclasses.replace(
            runtime._state.halt,
            source_id="a.audit",
            transition_id="forged.transition",
            invocation_index=99,
            witness=dataclasses.replace(
                runtime._state.halt.witness,
                facts=(
                    *runtime._state.halt.witness.facts[:4],
                    "a.audit",
                    "forged.transition",
                    99,
                ),
            ),
        )

    authority = runtime._halt_authority
    assert authority is not None
    assert runtime._state.halt is authority.record
    replacement = dataclasses.replace(runtime._state, halt=substituted)
    # Replacing every independent authority field too is arbitrary in-process
    # code execution and intentionally outside this private threat boundary.
    _inject_state(runtime, replacement, coherent=True)
    with pytest.raises(AuditIntegrityHalt) as caught:
        runtime.snapshot()
    assert caught.value.refusal_class == "malformed_runtime_counter"
    assert runtime._halt_authority is authority
    assert runtime._state.halt is authority.malformed_record


def test_feasible_runtime_refusal_matrix_emits_schema_valid_exact_rows() -> None:
    halted: list[AuditBudgetRuntime] = []

    def capture(runtime: AuditBudgetRuntime, action: object) -> None:
        with pytest.raises(AuditIntegrityHalt):
            action()  # type: ignore[operator]
        halted.append(runtime)

    unknown = _runtime()
    unknown_scope = unknown.begin_transition("matrix.unknown", 0)
    capture(
        unknown,
        lambda: unknown_scope.reserve_batch(
            "unknown.audit",
            "scheduled_audit_event_intent",
            ("test.audit.one",),
        ),
    )
    invalid = _runtime()
    invalid_scope = invalid.begin_transition("matrix.invalid", 1)
    capture(
        invalid,
        lambda: invalid_scope.reserve_batch(
            "test.audit",
            "scheduled_audit_event_intent",
            ("test.audit.same", "test.audit.same"),
        ),
    )
    lifecycle = _runtime()
    lifecycle_scope = lifecycle.begin_transition("matrix.lifecycle", 2)
    lifecycle_scope.reserve_batch(
        "test.audit", "scheduled_audit_event_intent", ("test.audit.one",)
    )
    capture(lifecycle, lifecycle_scope.commit)
    post = _runtime()
    post.mark_finality()
    capture(post, lambda: post.begin_transition("matrix.post", 3))
    malformed = _runtime()
    _inject_state(
        malformed,
        dataclasses.replace(malformed._state, source_counts=(("test.audit", True),)),
    )
    capture(malformed, malformed.snapshot)

    sources = (
        AuditBudgetSource("a.audit", "scheduled_audit_event_intent", 1, 1),
        AuditBudgetSource("b.audit", "immediate_audit_event_intent", 1, 1),
    )
    aggregate = AuditBudgetRuntime(
        AuditBudgetDeclaration(
            "operatebench.domain-generated-audit-budget.v1",
            "4" * 64,
            "opinst_matrix_aggregate",
            2,
            sources,
        )
    )
    for index, source in enumerate(sources):
        scope = aggregate.begin_transition(f"matrix.fill-{index}", index)
        reservation = scope.reserve_batch(
            source.source_id, source.counted_kind, (f"matrix.intent-{index}",)
        )
        reservation.consume(f"matrix.intent-{index}")
        scope.commit()
    aggregate_scope = aggregate.begin_transition("matrix.aggregate", 4)
    capture(
        aggregate,
        lambda: aggregate_scope.reserve_batch(
            "b.audit", sources[1].counted_kind, ("matrix.intent-over",)
        ),
    )

    source_overrun = _runtime(occurrences=1, fanout=2)
    first = source_overrun.begin_transition("matrix.source-first", 0)
    first_reservation = first.reserve_batch(
        "test.audit", "scheduled_audit_event_intent", ("matrix.source-one",)
    )
    first_reservation.consume("matrix.source-one")
    first.commit()
    second = source_overrun.begin_transition("matrix.source-over", 1)
    capture(
        source_overrun,
        lambda: second.reserve_batch(
            "test.audit", "scheduled_audit_event_intent", ("matrix.source-two",)
        ),
    )

    expected = {
        "unknown_source_or_kind",
        "invalid_batch_or_identity",
        "reservation_lifecycle_violation",
        "post_finality_attempt",
        "malformed_runtime_counter",
        "valid_reservation_above_declared_aggregate",
        "valid_reservation_above_declared_source",
    }
    expected_rows = {
        "unknown_source_or_kind": (
            "DOMAIN_AUDIT_CONTRACT_VIOLATION",
            "unknown.audit",
            1,
            0,
            None,
            2,
            "matrix.unknown",
            0,
        ),
        "invalid_batch_or_identity": (
            "DOMAIN_AUDIT_CONTRACT_VIOLATION",
            "test.audit",
            0,
            0,
            0,
            2,
            "matrix.invalid",
            1,
        ),
        "reservation_lifecycle_violation": (
            "DOMAIN_AUDIT_CONTRACT_VIOLATION",
            "test.audit",
            1,
            0,
            0,
            2,
            "matrix.lifecycle",
            2,
        ),
        "post_finality_attempt": (
            "DOMAIN_AUDIT_CONTRACT_VIOLATION",
            None,
            0,
            0,
            None,
            2,
            None,
            None,
        ),
        "malformed_runtime_counter": (
            "DOMAIN_AUDIT_COUNTER_INTEGRITY",
            None,
            0,
            0,
            None,
            2,
            None,
            None,
        ),
        "valid_reservation_above_declared_aggregate": (
            "DOMAIN_AUDIT_BUDGET_EXCEEDED",
            "b.audit",
            1,
            2,
            1,
            2,
            "matrix.aggregate",
            4,
        ),
        "valid_reservation_above_declared_source": (
            "DOMAIN_AUDIT_BUDGET_EXCEEDED",
            "test.audit",
            1,
            1,
            1,
            2,
            "matrix.source-over",
            1,
        ),
    }
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/domain-generated-audit-budget-evidence-v1.schema.json"
        ).read_text()
    )
    observed: set[str] = set()
    for index, runtime in enumerate(halted):
        snapshot = runtime.snapshot()
        records = runtime.terminal_evidence()
        finding = records[-2]
        observed.add(finding.refusal_class)
        assert (
            finding.integrity_code,
            finding.source_id,
            finding.requested_count,
            finding.operation_committed_before,
            finding.source_committed_before,
            finding.declared_max_events,
            finding.transition_id,
            finding.invocation_index,
        ) == expected_rows[finding.refusal_class]
        assert [row.record_type for row in records[-2:]] == [
            "integrity_finding",
            "terminal_summary",
        ]
        envelope = {
            "schema": "operatebench.domain_generated_audit_budget_evidence.v1",
            "schema_version": 1,
            "contract_id": "operatebench.domain-generated-audit-budget.v1",
            "operation_core_content_digest_sha256": "1" * 64,
            "operation_instance_id": f"opinst_matrix_{index}",
            "declared_max_events": records[-1].declared_max_events,
            "records": [dataclasses.asdict(row) for row in records],
        }
        assert snapshot.halted
        Draft202012Validator(schema).validate(envelope)
    assert observed == expected
