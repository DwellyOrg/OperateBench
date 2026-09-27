"""Finished captures release caller resources without relying on cyclic GC."""

import copy
import gc
import traceback
import weakref
from dataclasses import replace

import pytest

from operatebench.agents.transport import ExecutionRecord, ProviderFailure
from operatebench.sdk.profile_packs.pack import COMPLIANCE_COMMANDS
from tests.test_three_flow_runtime import no_network, scripted_handler, setup_transport
from tools import three_flow_runtime as runtime

# Reuse the HTTP suite's network tripwire as an autouse fixture.
assert no_network


@pytest.fixture
def delayed_gc():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


@pytest.fixture
def captures(monkeypatch):
    retained = []
    original = runtime._Capture

    class Capture(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            retained.append(self)

    monkeypatch.setattr(runtime, "_Capture", Capture)
    return retained


def execute(tmp_path, flow, scenario, captures):
    calls = []
    trial = runtime.Trial("lifetime", "trial", flow, scenario, "gpt-6-astra", 128000)
    budget, transport = setup_transport(tmp_path, trial, scripted_handler(flow, calls))
    ref = weakref.ref(transport)
    observer = [].append
    transport.wire.observer = observer
    trial = replace(trial, provider_binding=runtime.provider_identity(transport))
    try:
        record = runtime.run_trial(trial, transport, output=tmp_path / "trial")
        assert transport.wire.observer is observer
        assert budget.totals()["unknown"] == 0
        assert budget.totals()["pending"] == 0
        count = len(calls)
        assert count > 0
        assert (
            runtime.replay_trial(
                trial, record, expected_record_digest=record["record_digest"]
            )["provider_calls"]
            == 0
        )
        assert len(calls) == count
        capture = captures[-1]
        before = copy.deepcopy((record, capture.partial.rows))
        evidence = (tmp_path / "trial" / "partial.ndjson").read_bytes()
        assert capture.closed and capture.inner is None
        capture.release()  # Idempotent and must not mutate finalized evidence.
        with pytest.raises(RuntimeError, match="capture is closed"):
            capture.send(None)
        assert (record, capture.partial.rows) == before
        assert (tmp_path / "trial" / "partial.ndjson").read_bytes() == evidence
        # Caller still owns a usable transport and its closure.
        assert transport.model == trial.model
    finally:
        try:
            transport.close()
        finally:
            budget.close()
    return ref


@pytest.mark.parametrize(
    "flow,scenario",
    [
        ("commerce", "UK_NORMAL"),
        ("compliance", COMPLIANCE_COMMANDS.reference_scenarios[0]),
        ("maintenance", "V1"),
    ],
)
def test_finished_capture_releases_transport(
    tmp_path, captures, delayed_gc, flow, scenario
):
    ref = execute(tmp_path, flow, scenario, captures)
    assert ref() is None


@pytest.mark.parametrize("fault", ["provider", "runtime", "none"])
@pytest.mark.parametrize("cleanup", ["close", "finish", "none"])
def test_release_with_retained_exception_and_cleanup_failure(
    tmp_path, monkeypatch, captures, delayed_gc, fault, cleanup
):
    primary = (
        ProviderFailure(
            ExecutionRecord(
                "model",
                "gpt-6-astra",
                None,
                128000,
                0,
                0,
                (),
                True,
                "provider_transport",
                "original",
            ),
            "original",
        )
        if fault == "provider"
        else ValueError("original")
    )
    secondary = OSError("evidence cleanup")
    trial = runtime.Trial(
        "lifetime", "trial", "commerce", "UK_NORMAL", "gpt-6-astra", 128000
    )
    budget, transport = setup_transport(tmp_path, trial, scripted_handler("commerce", []))
    ref = weakref.ref(transport)
    observer = [].append
    transport.wire.observer = observer
    trial = replace(trial, provider_binding=runtime.provider_identity(transport))
    if fault != "none":

        def fail(*args, **kwargs):
            raise primary

        monkeypatch.setattr(runtime, "_execute", fail)
    if cleanup != "none":
        original = getattr(runtime.PartialExecutionEvidence, cleanup)

        def fail_cleanup(self, *args, **kwargs):
            original(self, *args, **kwargs)
            raise secondary

        monkeypatch.setattr(runtime.PartialExecutionEvidence, cleanup, fail_cleanup)
    retained = None
    try:
        try:
            runtime.run_trial(trial, transport, output=tmp_path / "trial")
        except BaseException as exc:
            retained = exc
        assert retained is (
            primary if fault != "none" else secondary if cleanup != "none" else None
        )
        assert transport.wire.observer is observer
        assert captures[-1].closed and captures[-1].inner is None
        if fault != "none" and cleanup != "none":
            assert "Trial evidence finalization also failed" in retained.__notes__
            if cleanup == "close":
                assert "Trial evidence cleanup also failed" in retained.__notes__
        if retained is not None:
            # Keeping a traceback inherently keeps run_trial's caller-owned
            # argument alive; the capture must already be detached regardless.
            traceback.clear_frames(retained.__traceback__)
        traceback.clear_frames(primary.__traceback__)
        traceback.clear_frames(secondary.__traceback__)
    finally:
        try:
            transport.close()
        finally:
            budget.close()
    del transport
    assert ref() is None


@pytest.mark.parametrize("inside_handler", [False, True])
def test_final_cleanup_failure_is_not_swallowed(tmp_path, monkeypatch, inside_handler):
    trial = runtime.Trial(
        "lifetime", "trial", "commerce", "UK_NORMAL", "gpt-6-astra", 128000
    )
    budget, transport = setup_transport(tmp_path, trial, scripted_handler("commerce", []))
    trial = replace(trial, provider_binding=runtime.provider_identity(transport))
    original = runtime.PartialExecutionEvidence.close
    secondary = OSError("final-only close error")

    def close(self):
        was_closed = self._closed
        original(self)
        if was_closed:
            raise secondary

    monkeypatch.setattr(runtime.PartialExecutionEvidence, "close", close)

    def invoke():
        with pytest.raises(OSError, match="final-only close error"):
            runtime.run_trial(trial, transport, output=tmp_path / "trial")

    try:
        if inside_handler:
            try:
                raise ValueError("unrelated handled caller exception")
            except ValueError:
                invoke()
        else:
            invoke()
    finally:
        try:
            transport.close()
        finally:
            budget.close()
