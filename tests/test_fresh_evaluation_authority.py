# SPDX-FileCopyrightText: 2026 OperateBench contributors
# SPDX-License-Identifier: Apache-2.0
"""Fresh execution/evaluation authority stays private and process-local."""

from __future__ import annotations

import ast
import copy
import gc
import inspect
import pickle
import subprocess
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

import operatebench.runner as runner
from operatebench.core.engine import has_canonical_episode_outcome_provenance
from operatebench.domains.lettings.maintenance import evaluator
from operatebench.domains.lettings.maintenance.spec import load_spec

FIXTURE = Path("examples/operatebench/maintenance_v0_1.yaml")


def _mint(*, scenario_id: str = "V1", discriminator: int = 0) -> Any:
    return runner._execute_and_self_check(  # type: ignore[attr-defined]
        load_spec(FIXTURE),
        scenario_id,
        "reference",
        operation_instance_id=f"opinst_{discriminator:032x}",
    )


def _changed_value(name: str, value: Any) -> Any:
    if name == "status":
        return "FORGED_STATUS"
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is str:
        return f"{value}_FORGED"
    if value is None:
        return "FORGED_ABSENCE"
    if type(value) is tuple:
        return value[1:] if value else ("FORGED_ITEM",)
    if type(value) is frozenset:
        return value | {"FORGED_ITEM"}
    if type(value) is MappingProxyType:
        return MappingProxyType({**value, "FORGED_KEY": "FORGED_VALUE"})
    raise AssertionError(f"test has no mutation for {name}: {type(value)!r}")


def test_runtime_only_execution_mints_fresh_input_without_grading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = load_spec(FIXTURE)
    calls: list[str] = []

    def grading_must_not_run(*args: Any, **kwargs: Any) -> Any:
        calls.append("grade")
        raise AssertionError("runtime-only execution invoked the maintenance evaluator")

    monkeypatch.setattr(evaluator, "evaluate", grading_must_not_run)
    monkeypatch.setattr(runner, "_evaluate_fresh", grading_must_not_run, raising=False)

    fresh = runner._execute_and_self_check(  # type: ignore[attr-defined]
        spec,
        "V1",
        "reference",
        operation_instance_id="opinst_0123456789abcdef0123456789abcdef",
    )

    assert type(fresh).__name__ == "_FreshEvaluationInput"
    assert type(fresh).__module__ == "operatebench.runner"
    assert has_canonical_episode_outcome_provenance(fresh.outcome)
    assert calls == []


def test_fresh_authority_grade_matches_the_compatibility_computation() -> None:
    spec = load_spec(FIXTURE)
    fresh = runner._execute_and_self_check(  # type: ignore[attr-defined]
        spec,
        "V1",
        "reference",
        operation_instance_id="opinst_1123456789abcdef0123456789abcdef",
    )
    outcome = fresh.outcome
    baseline = evaluator.evaluate(
        status=outcome.status,
        replay_final=outcome.replay_final,
        simulated_minutes=outcome.simulated_minutes,
        invocations=outcome.invocations,
        final_state=outcome.final_state,
        trajectory=outcome.trajectory,
        events=outcome.events,
        scenario=spec.scenario("V1"),
        replay_ok=True,
    )

    authoritative = evaluator._evaluate_fresh(fresh)  # type: ignore[attr-defined]

    assert authoritative.as_dict() == baseline.as_dict()


def test_authority_grade_does_not_call_the_public_raw_computation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh = runner._execute_and_self_check(  # type: ignore[attr-defined]
        load_spec(FIXTURE),
        "V1",
        "reference",
        operation_instance_id="opinst_6123456789abcdef0123456789abcdef",
    )

    def raw_computation_must_not_run(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("authoritative grading called public raw evaluate")

    monkeypatch.setattr(evaluator, "evaluate", raw_computation_must_not_run)

    assert evaluator._evaluate_fresh(fresh).reliable  # type: ignore[attr-defined]


def test_fresh_authority_refuses_a_caller_forged_wrapper() -> None:
    spec = load_spec(FIXTURE)
    fresh = runner._execute_and_self_check(  # type: ignore[attr-defined]
        spec,
        "V1",
        "reference",
        operation_instance_id="opinst_2123456789abcdef0123456789abcdef",
    )
    forged = SimpleNamespace(
        outcome=fresh.outcome,
        scenario=spec.scenario("V1"),
        replay_ok=True,
    )

    with pytest.raises(ValueError, match="fresh evaluation authority"):
        evaluator._evaluate_fresh(forged)  # type: ignore[attr-defined]


def test_direct_private_wrapper_construction_does_not_mint_authority() -> None:
    fresh = runner._execute_and_self_check(  # type: ignore[attr-defined]
        load_spec(FIXTURE),
        "V1",
        "reference",
        operation_instance_id="opinst_d123456789abcdef0123456789abcdef",
    )
    caller_built = type(fresh)(
        fresh.outcome,
        fresh.replay_ok,
        fresh.scenario,
        fresh._executed,
        fresh._snapshot,
    )

    with pytest.raises(ValueError, match="fresh evaluation authority"):
        evaluator._evaluate_fresh(caller_built)  # type: ignore[attr-defined]


def test_caller_built_and_replaced_outcomes_are_refused_before_grading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = load_spec(FIXTURE)
    fresh = runner._execute_and_self_check(  # type: ignore[attr-defined]
        spec,
        "V1",
        "reference",
        operation_instance_id="opinst_3123456789abcdef0123456789abcdef",
    )
    projection = fresh.outcome.as_dict()
    caller_outcomes = (
        type(fresh.outcome)(**projection),
        replace(fresh.outcome),
    )

    def grading_must_not_run(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an unauthenticated outcome reached grading")

    monkeypatch.setattr(evaluator, "evaluate", grading_must_not_run)
    for outcome in caller_outcomes:
        forged = SimpleNamespace(
            outcome=outcome,
            replay_ok=fresh.replay_ok,
            scenario=fresh.scenario,
            _executed=fresh._executed,
        )
        with pytest.raises(ValueError, match="fresh evaluation authority"):
            evaluator._evaluate_fresh(forged)  # type: ignore[attr-defined]


def test_copy_deepcopy_and_pickle_cannot_transfer_fresh_authority() -> None:
    fresh = runner._execute_and_self_check(  # type: ignore[attr-defined]
        load_spec(FIXTURE),
        "V1",
        "reference",
        operation_instance_id="opinst_4123456789abcdef0123456789abcdef",
    )

    for transfer in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError, match="fresh evaluation authority"):
            transfer(fresh)


def test_evaluator_admission_uses_a_recursively_frozen_detached_snapshot() -> None:
    spec = load_spec(FIXTURE)
    scenario = spec.scenario("V1")
    fresh = runner._execute_and_self_check(  # type: ignore[attr-defined]
        spec,
        "V1",
        "reference",
        operation_instance_id="opinst_5123456789abcdef0123456789abcdef",
    )

    snapshot, sealed_scenario, replay_ok = runner._admit_fresh_evaluation(fresh)  # type: ignore[attr-defined]

    assert snapshot is not fresh.outcome
    assert sealed_scenario is not scenario
    assert replay_ok is True
    with pytest.raises(TypeError):
        snapshot.final_state["forged"] = True
    with pytest.raises(TypeError):
        snapshot.trajectory[0]["forged"] = True
    with pytest.raises(TypeError):
        snapshot.events[0]["forged"] = True
    with pytest.raises(TypeError):
        sealed_scenario.events[0].payload["forged"] = True


def test_scenario_and_self_check_are_sealed_inside_the_fresh_input() -> None:
    spec = load_spec(FIXTURE)
    for field_name, replacement in (
        ("scenario", spec.scenario("V2")),
        ("replay_ok", False),
    ):
        fresh = runner._execute_and_self_check(  # type: ignore[attr-defined]
            spec,
            "V1",
            "reference",
            operation_instance_id=(
                "opinst_7123456789abcdef0123456789abcdef"
                if field_name == "scenario"
                else "opinst_8123456789abcdef0123456789abcdef"
            ),
        )
        object.__setattr__(fresh, field_name, replacement)

        with pytest.raises(ValueError, match="fresh evaluation authority"):
            evaluator._evaluate_fresh(fresh)  # type: ignore[attr-defined]


def test_fresh_authority_refuses_a_pid_mismatch_without_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh = runner._execute_and_self_check(  # type: ignore[attr-defined]
        load_spec(FIXTURE),
        "V1",
        "reference",
        operation_instance_id="opinst_e123456789abcdef0123456789abcdef",
    )
    creator_pid = runner.os.getpid()
    monkeypatch.setattr(runner.os, "getpid", lambda: creator_pid + 1)

    with pytest.raises(ValueError, match="fresh evaluation authority"):
        evaluator._evaluate_fresh(fresh)  # type: ignore[attr-defined]


def test_nested_outcome_mutation_is_refused_before_grading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = load_spec(FIXTURE)
    mutations = (
        lambda fresh: fresh.outcome.final_state.__setitem__("phase", "FORGED"),
        lambda fresh: fresh.outcome.trajectory[0].__setitem__("record_type", "forged"),
        lambda fresh: fresh.outcome.events[0].__setitem__("disposition", "forged"),
    )
    grade_calls: list[str] = []

    def grading_must_not_run(*args: Any, **kwargs: Any) -> Any:
        grade_calls.append("grade")
        raise AssertionError("mutated outcome reached deterministic grading")

    monkeypatch.setattr(evaluator, "_evaluate_fields", grading_must_not_run)
    for index, mutate in enumerate(mutations):
        fresh = runner._execute_and_self_check(  # type: ignore[attr-defined]
            spec,
            "V1",
            "reference",
            operation_instance_id=f"opinst_{index + 9:032x}",
        )
        mutate(fresh)
        with pytest.raises(ValueError, match="fresh evaluation authority"):
            evaluator._evaluate_fresh(fresh)  # type: ignore[attr-defined]
    assert grade_calls == []


@pytest.mark.parametrize(
    "field_name",
    (
        "status",
        "replay_final",
        "simulated_minutes",
        "invocations",
        "final_state",
        "trajectory",
        "events",
    ),
)
def test_every_evaluator_snapshot_field_is_sealed_after_mint(field_name: str) -> None:
    fresh = _mint(discriminator=0xA0)
    assert {item.name for item in fields(fresh._snapshot)} == {
        "status",
        "replay_final",
        "simulated_minutes",
        "invocations",
        "final_state",
        "trajectory",
        "events",
    }
    original = getattr(fresh._snapshot, field_name)
    object.__setattr__(fresh._snapshot, field_name, _changed_value(field_name, original))

    with pytest.raises(ValueError, match="fresh evaluation authority"):
        runner._admit_fresh_evaluation(fresh)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "field_name",
    (
        "scenario_id",
        "semantic_scenario_id",
        "label",
        "starts_at",
        "expected_terminal",
        "human_checkpoint_budget",
        "required_checkpoint_types",
        "dispatch_failures",
        "expected_event_rejections",
        "events",
    ),
)
def test_every_evaluator_scenario_field_is_sealed_after_mint(field_name: str) -> None:
    fresh = _mint(discriminator=0xA1)
    scenario = fresh.scenario
    assert {item.name for item in fields(scenario)} == {
        "scenario_id",
        "semantic_scenario_id",
        "label",
        "starts_at",
        "expected_terminal",
        "human_checkpoint_budget",
        "required_checkpoint_types",
        "dispatch_failures",
        "expected_event_rejections",
        "events",
    }
    original = getattr(scenario, field_name)
    object.__setattr__(scenario, field_name, _changed_value(field_name, original))

    with pytest.raises(ValueError, match="fresh evaluation authority"):
        runner._admit_fresh_evaluation(fresh)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "field_name",
    (
        "event_id",
        "event_type",
        "actor_id",
        "sequence",
        "payload",
        "triggers_agent",
        "at",
        "trigger",
        "delay_minutes",
    ),
)
def test_every_authored_event_field_is_sealed_after_mint(field_name: str) -> None:
    fresh = _mint(discriminator=0xA2)
    event = next(item for item in fresh.scenario.events if item.trigger is not None)
    assert {item.name for item in fields(event)} == {
        "event_id",
        "event_type",
        "actor_id",
        "sequence",
        "payload",
        "triggers_agent",
        "at",
        "trigger",
        "delay_minutes",
    }
    original = getattr(event, field_name)
    replacement = (
        None if field_name == "trigger" else _changed_value(field_name, original)
    )
    object.__setattr__(event, field_name, replacement)

    with pytest.raises(ValueError, match="fresh evaluation authority"):
        runner._admit_fresh_evaluation(fresh)  # type: ignore[attr-defined]


@pytest.mark.parametrize("field_name", ("kind", "type_name", "cycle_id"))
def test_every_conditional_trigger_field_is_sealed_after_mint(field_name: str) -> None:
    fresh = _mint(discriminator=0xA3)
    trigger = next(
        item.trigger for item in fresh.scenario.events if item.trigger is not None
    )
    assert {item.name for item in fields(trigger)} == {"kind", "type_name", "cycle_id"}
    original = getattr(trigger, field_name)
    object.__setattr__(trigger, field_name, _changed_value(field_name, original))

    with pytest.raises(ValueError, match="fresh evaluation authority"):
        runner._admit_fresh_evaluation(fresh)  # type: ignore[attr-defined]


@pytest.mark.parametrize("field_name", ("event_id", "code", "reason"))
def test_every_expected_event_rejection_field_is_sealed_after_mint(
    field_name: str,
) -> None:
    fresh = _mint(discriminator=0xA4)
    expected = fresh.scenario.expected_event_rejections[0]
    assert {item.name for item in fields(expected)} == {"event_id", "code", "reason"}
    original = getattr(expected, field_name)
    object.__setattr__(expected, field_name, _changed_value(field_name, original))

    with pytest.raises(ValueError, match="fresh evaluation authority"):
        runner._admit_fresh_evaluation(fresh)  # type: ignore[attr-defined]


def test_another_genuine_engine_outcome_cannot_be_substituted_after_mint() -> None:
    first = _mint(scenario_id="V1", discriminator=0xA5)
    second = _mint(scenario_id="V2", discriminator=0xA6)
    assert has_canonical_episode_outcome_provenance(second.outcome)
    object.__setattr__(first, "outcome", second.outcome)

    with pytest.raises(ValueError, match="fresh evaluation authority"):
        runner._admit_fresh_evaluation(first)  # type: ignore[attr-defined]


def test_direct_object_new_forgery_is_refused_even_with_creator_pid_spoof() -> None:
    fresh = _mint(discriminator=0xA7)
    forged = object.__new__(type(fresh))
    for item in fields(fresh):
        object.__setattr__(forged, item.name, getattr(fresh, item.name))
    with pytest.raises(AttributeError):
        object.__setattr__(forged, "_creator_pid", runner.os.getpid())

    with pytest.raises(ValueError, match="fresh evaluation authority"):
        runner._admit_fresh_evaluation(forged)  # type: ignore[attr-defined]


@pytest.mark.parametrize("hostile", (object(), {"oversized": "x" * (8 * 1024 * 1024)}))
def test_unsupported_and_oversized_post_mint_values_are_refused(hostile: object) -> None:
    fresh = _mint(discriminator=0xA8)
    object.__setattr__(fresh._snapshot, "final_state", MappingProxyType({"x": hostile}))

    with pytest.raises(ValueError, match="fresh evaluation authority") as caught:
        runner._admit_fresh_evaluation(fresh)  # type: ignore[attr-defined]
    assert caught.value.__cause__ is None


def test_cyclic_post_mint_value_is_refused_without_retaining_a_cause() -> None:
    fresh = _mint(discriminator=0xA9)
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    object.__setattr__(fresh._snapshot, "final_state", MappingProxyType(cyclic))

    with pytest.raises(ValueError, match="fresh evaluation authority") as caught:
        runner._admit_fresh_evaluation(fresh)  # type: ignore[attr-defined]
    assert caught.value.__cause__ is None


def test_deep_post_mint_value_is_refused_deterministically() -> None:
    fresh = _mint(discriminator=0xAC)
    deep: Any = "leaf"
    for _ in range(70):
        deep = (deep,)
    object.__setattr__(fresh._snapshot, "final_state", MappingProxyType({"x": deep}))

    with pytest.raises(ValueError, match="fresh evaluation authority") as caught:
        runner._admit_fresh_evaluation(fresh)  # type: ignore[attr-defined]
    assert caught.value.__cause__ is None


def test_type_sensitive_fingerprint_distinguishes_bool_from_int() -> None:
    fresh = _mint(discriminator=0xAA)
    object.__setattr__(fresh._snapshot, "invocations", True)

    with pytest.raises(ValueError, match="fresh evaluation authority"):
        runner._admit_fresh_evaluation(fresh)  # type: ignore[attr-defined]


def _traversed_fingerprint_values(value: Any) -> int:
    if is_dataclass(value) and not isinstance(value, type):
        return 1 + sum(
            _traversed_fingerprint_values(getattr(value, item.name))
            for item in fields(value)
        )
    if type(value) in (dict, MappingProxyType):
        return 1 + sum(
            _traversed_fingerprint_values(key) + _traversed_fingerprint_values(item)
            for key, item in value.items()
        )
    if type(value) in (list, tuple, set, frozenset):
        return 1 + sum(_traversed_fingerprint_values(item) for item in value)
    return 1


def test_fingerprint_budget_charges_each_output_byte_and_traversed_value_once() -> None:
    fresh = _mint(discriminator=0xAD)
    values = (
        None,
        False,
        7,
        1.5,
        "text",
        b"bytes",
        {b"key": [b"value"]},
        MappingProxyType({b"key": (b"value",)}),
        {b"set-value"},
        frozenset({b"frozen-value"}),
        fresh._snapshot,
        fresh.scenario,
    )

    for value in values:
        state = runner._FreshFingerprintState()  # type: ignore[attr-defined]
        initial_bytes = state.remaining_bytes
        initial_nodes = state.remaining_nodes

        encoded = runner._fresh_value_fingerprint(value, state)  # type: ignore[attr-defined]

        assert initial_bytes - state.remaining_bytes == len(encoded)
        assert initial_nodes - state.remaining_nodes == _traversed_fingerprint_values(
            value
        )


def test_primitive_fingerprint_exact_byte_limit_and_one_more() -> None:
    byte_limit = 8 * 1024 * 1024
    exact_value = b"x" * (byte_limit - 1)
    exact_state = runner._FreshFingerprintState()  # type: ignore[attr-defined]

    encoded = runner._fresh_value_fingerprint(exact_value, exact_state)  # type: ignore[attr-defined]

    assert len(encoded) == byte_limit
    assert exact_state.remaining_bytes == 0
    assert exact_state.remaining_nodes == 100_000 - 1

    with pytest.raises(
        runner._FreshFingerprintRefusal,  # type: ignore[attr-defined]
        match="oversized",
    ):
        runner._fresh_value_fingerprint(  # type: ignore[attr-defined]
            b"x" * byte_limit,
            runner._FreshFingerprintState(),  # type: ignore[attr-defined]
        )


def test_nested_fingerprint_accepts_the_exact_byte_limit_and_refuses_one_more() -> None:
    byte_limit = 8 * 1024 * 1024
    exact_value = [b"x" * (byte_limit - 10)]
    exact_state = runner._FreshFingerprintState()  # type: ignore[attr-defined]

    encoded = runner._fresh_value_fingerprint(exact_value, exact_state)  # type: ignore[attr-defined]

    assert len(encoded) == byte_limit
    assert exact_state.remaining_bytes == 0
    assert exact_state.remaining_nodes == 100_000 - 2

    oversized_value = [b"x" * (byte_limit - 9)]
    with pytest.raises(
        runner._FreshFingerprintRefusal,  # type: ignore[attr-defined]
        match="oversized",
    ):
        runner._fresh_value_fingerprint(  # type: ignore[attr-defined]
            oversized_value,
            runner._FreshFingerprintState(),  # type: ignore[attr-defined]
        )


def test_external_mint_registry_releases_authority_with_the_wrapper() -> None:
    # Exclude unreachable authorities left by earlier tests from the live baseline.
    gc.collect()
    baseline = len(runner._FRESH_MINTS)  # type: ignore[attr-defined]
    fresh = _mint(discriminator=0xAB)
    wrapper_id = id(fresh)
    fresh_ref = weakref.ref(fresh)
    assert len(runner._FRESH_MINTS) == baseline + 1  # type: ignore[attr-defined]

    del fresh
    gc.collect()

    assert fresh_ref() is None
    assert wrapper_id not in runner._FRESH_MINTS  # type: ignore[attr-defined]
    assert len(runner._FRESH_MINTS) <= baseline  # type: ignore[attr-defined]


def test_refused_cyclic_authority_graphs_are_released_after_caller_cleanup() -> None:
    # Exclude unreachable authorities left by earlier tests from the live baseline.
    gc.collect()
    baseline = len(runner._FRESH_MINTS)  # type: ignore[attr-defined]

    def make_refused_cycle(discriminator: int) -> tuple[weakref.ReferenceType[Any], int]:
        fresh = _mint(discriminator=discriminator)
        fresh_ref = weakref.ref(fresh)
        wrapper_id = id(fresh)
        object.__setattr__(
            fresh._snapshot,
            "final_state",
            MappingProxyType({"wrapper": fresh}),
        )
        with pytest.raises(ValueError, match="fresh evaluation authority"):
            runner._admit_fresh_evaluation(fresh)  # type: ignore[attr-defined]
        return fresh_ref, wrapper_id

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        authorities = [make_refused_cycle(0xB0 + index) for index in range(12)]
        assert len(runner._FRESH_MINTS) == baseline + 12  # type: ignore[attr-defined]
    finally:
        if gc_was_enabled:
            gc.enable()

    gc.collect()

    assert all(reference() is None for reference, _ in authorities)
    assert all(
        wrapper_id not in runner._FRESH_MINTS  # type: ignore[attr-defined]
        for _, wrapper_id in authorities
    )
    assert len(runner._FRESH_MINTS) == baseline  # type: ignore[attr-defined]


def test_stale_weakref_callback_cannot_retire_a_live_reused_registry_slot() -> None:
    fresh = _mint(discriminator=0xBC)
    wrapper_id = id(fresh)
    record = runner._FRESH_MINTS[wrapper_id]  # type: ignore[attr-defined]
    stale_ref = weakref.ref(fresh)
    assert stale_ref is not record.wrapper_ref

    runner._retire_fresh_mint(wrapper_id, stale_ref)  # type: ignore[attr-defined]

    assert runner._FRESH_MINTS[wrapper_id] is record  # type: ignore[attr-defined]
    assert runner._admit_fresh_evaluation(fresh)[0] is fresh._snapshot  # type: ignore[attr-defined]


def test_replaced_mint_record_cannot_admit_after_fingerprint_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh = _mint(discriminator=0xBE)
    wrapper_id = id(fresh)
    record = runner._FRESH_MINTS[wrapper_id]  # type: ignore[attr-defined]
    fingerprint_started = threading.Event()
    resume_fingerprint = threading.Event()
    original_fingerprint = runner._fresh_fingerprint  # type: ignore[attr-defined]

    def blocking_fingerprint(value: Any) -> bytes:
        if value is fresh._snapshot:
            fingerprint_started.set()
            assert resume_fingerprint.wait(timeout=10)
        return original_fingerprint(value)

    monkeypatch.setattr(runner, "_fresh_fingerprint", blocking_fingerprint)
    with ThreadPoolExecutor(max_workers=1) as pool:
        admission = pool.submit(runner._admit_fresh_evaluation, fresh)  # type: ignore[attr-defined]
        assert fingerprint_started.wait(timeout=10)
        runner._FRESH_MINTS[wrapper_id] = replace(record)  # type: ignore[attr-defined]
        resume_fingerprint.set()
        with pytest.raises(ValueError, match="fresh evaluation authority"):
            admission.result(timeout=10)
    runner._retire_fresh_mint(  # type: ignore[attr-defined]
        wrapper_id,
        runner._FRESH_MINTS[wrapper_id].wrapper_ref,  # type: ignore[attr-defined]
    )


def test_blocked_fingerprint_does_not_block_an_independent_registry_retirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = _mint(discriminator=0xBF)
    independent = _mint(discriminator=0xC0)
    independent_id = id(independent)
    independent_record = runner._FRESH_MINTS[independent_id]  # type: ignore[attr-defined]
    fingerprint_started = threading.Event()
    resume_fingerprint = threading.Event()
    original_fingerprint = runner._fresh_fingerprint  # type: ignore[attr-defined]

    def blocking_fingerprint(value: Any) -> bytes:
        if value is blocked._snapshot:
            fingerprint_started.set()
            assert resume_fingerprint.wait(timeout=10)
        return original_fingerprint(value)

    monkeypatch.setattr(runner, "_fresh_fingerprint", blocking_fingerprint)
    with ThreadPoolExecutor(max_workers=2) as pool:
        admission = pool.submit(runner._admit_fresh_evaluation, blocked)  # type: ignore[attr-defined]
        assert fingerprint_started.wait(timeout=10)
        retirement = pool.submit(
            runner._retire_fresh_mint,  # type: ignore[attr-defined]
            independent_id,
            independent_record.wrapper_ref,
        )
        try:
            retirement.result(timeout=2)
        finally:
            resume_fingerprint.set()
        assert admission.result(timeout=10)[0] is blocked._snapshot

    assert independent_id not in runner._FRESH_MINTS  # type: ignore[attr-defined]
    blocked_id = id(blocked)
    runner._retire_fresh_mint(  # type: ignore[attr-defined]
        blocked_id,
        runner._FRESH_MINTS[blocked_id].wrapper_ref,  # type: ignore[attr-defined]
    )


def test_mint_registry_record_contains_only_detached_primitive_authority_state() -> None:
    fresh = _mint(discriminator=0xBD)
    record = runner._FRESH_MINTS[id(fresh)]  # type: ignore[attr-defined]

    assert {item.name for item in fields(record)} == {
        "wrapper_ref",
        "creator_pid",
        "outcome_id",
        "executed_id",
        "snapshot_id",
        "scenario_id",
        "replay_ok",
        "snapshot_fingerprint",
        "scenario_fingerprint",
    }
    assert record.wrapper_ref() is fresh
    assert all(
        type(getattr(record, name)) is int
        for name in (
            "creator_pid",
            "outcome_id",
            "executed_id",
            "snapshot_id",
            "scenario_id",
        )
    )
    assert type(record.replay_ok) in (bool, type(None))
    assert type(record.snapshot_fingerprint) is bytes
    assert type(record.scenario_fingerprint) is bytes


@pytest.mark.parametrize("failure_site", ["mint", "admit"])
def test_concurrent_gc_worker_errors_propagate_without_hanging(failure_site: str) -> None:
    # A subprocess deadline catches executor-shutdown hangs as well as masked errors.
    script = f"""
import runpy
from types import SimpleNamespace
namespace = runpy.run_path('tests/test_fresh_evaluation_authority.py')
test = namespace['test_concurrent_mint_admit_and_cyclic_gc_leave_no_registry_records']
class WorkerFailure(Exception):
    pass
class Fresh:
    def __init__(self, index):
        self.index = index
        self._snapshot = SimpleNamespace()
def mint(*, discriminator):
    if {failure_site!r} == 'mint' and discriminator == 0xCB:
        raise WorkerFailure('original worker failure')
    return Fresh(discriminator)
def admit(fresh):
    if {failure_site!r} == 'admit' and fresh.index == 0xCB:
        raise WorkerFailure('original worker failure')
    return (fresh._snapshot,)
test.__globals__['_mint'] = mint
namespace['runner']._admit_fresh_evaluation = admit
try:
    test()
except WorkerFailure as error:
    assert str(error) == 'original worker failure'
else:
    raise AssertionError('worker failure was swallowed')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=20
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_concurrent_mint_admit_and_cyclic_gc_leave_no_registry_records() -> None:
    # Exclude unreachable authorities left by earlier tests from the live baseline.
    gc.collect()
    baseline = len(runner._FRESH_MINTS)  # type: ignore[attr-defined]
    worker_count = 12
    # Twelve full executions/replays plus admission contend under branch coverage:
    # Python 3.11 can take over three minutes here. This is not a speed assertion.
    timeout = 600
    minted = threading.Barrier(worker_count + 1, timeout=timeout)
    release = threading.Event()

    def mint_admit_and_release(index: int) -> tuple[weakref.ReferenceType[Any], int]:
        try:
            fresh = _mint(discriminator=0xC0 + index)
            wrapper_id = id(fresh)
            assert runner._admit_fresh_evaluation(fresh)[0] is fresh._snapshot  # type: ignore[attr-defined]
            object.__setattr__(
                fresh._snapshot,
                "final_state",
                MappingProxyType({"wrapper": fresh}),
            )
        except BaseException:
            # Wake peers and the observer; do not hide this error behind a timeout.
            minted.abort()
            raise
        minted.wait()
        assert release.wait(timeout=timeout)
        return weakref.ref(fresh), wrapper_id

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        indices = range(worker_count)
        futures = [pool.submit(mint_admit_and_release, index) for index in indices]
        try:
            minted.wait()
            assert len(runner._FRESH_MINTS) == baseline + worker_count  # type: ignore[attr-defined]
        except threading.BrokenBarrierError:
            release.set()
            # Peers may only report the aborted barrier. Prefer the causal error.
            for future in futures:
                try:
                    future.result(timeout=timeout)
                except threading.BrokenBarrierError:
                    continue
            raise
        finally:
            # Also release workers if the live-registry assertion fails.
            release.set()
        gc.collect()
        authorities = [future.result(timeout=timeout) for future in futures]

    gc.collect()

    assert all(reference() is None for reference, _ in authorities)
    assert all(
        wrapper_id not in runner._FRESH_MINTS  # type: ignore[attr-defined]
        for _, wrapper_id in authorities
    )
    assert len(runner._FRESH_MINTS) == baseline  # type: ignore[attr-defined]


def test_fresh_input_is_private_slotted_immutable_and_replace_loses_authority() -> None:
    import operatebench

    fresh = runner._execute_and_self_check(  # type: ignore[attr-defined]
        load_spec(FIXTURE),
        "V1",
        "reference",
        operation_instance_id="opinst_c123456789abcdef0123456789abcdef",
    )

    assert is_dataclass(fresh)
    assert not hasattr(fresh, "__dict__")
    assert not hasattr(operatebench, "_FreshEvaluationInput")
    assert "_evaluate_fresh" not in evaluator.__all__
    with pytest.raises(FrozenInstanceError):
        fresh.replay_ok = False
    with pytest.raises(ValueError, match="fresh evaluation authority"):
        evaluator._evaluate_fresh(replace(fresh))  # type: ignore[attr-defined]


def test_authority_surface_contains_no_later_b3_or_b4_api() -> None:
    forbidden = {
        "ExecutedRuntimeEvidenceV1",
        "EpisodeOutcomeV2",
        "ReplayCapabilityV1",
        "EvaluatorGradeV1",
        "Decision",
        "can_runtime_replay",
        "can_evaluator_regrade",
        "regrade",
        "consume",
    }
    names: set[str] = set()
    for module in (runner, evaluator):
        tree = ast.parse(Path(inspect.getsourcefile(module) or "").read_text())
        names.update(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        )

    assert names.isdisjoint(forbidden)
    assert set(runner.__all__).isdisjoint(forbidden | {"_FreshEvaluationInput"})
    assert set(evaluator.__all__).isdisjoint(forbidden | {"_evaluate_fresh"})
    assert tuple(inspect.signature(evaluator._evaluate_fresh).parameters) == (  # type: ignore[attr-defined]
        "fresh",
    )
