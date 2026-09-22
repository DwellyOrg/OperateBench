"""Offline semantic-prefix retention; no provider sockets or private fixtures."""

import json
import os

import pytest

from operatebench.agents.model import ModelAgent
from operatebench.agents.transport import ProviderFailure
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.runner import run_episode
from tests.model_transport import TEST_MODEL, ReferenceDrivenTransport
from tests.test_operatebench_provenance_binding import FIXTURE


def test_budget_exclusion_exposes_non_scored_semantic_prefix():
    with pytest.raises(ProviderFailure) as caught:
        run_episode(
            load_spec(FIXTURE),
            "V1",
            "partial-test",
            agent_factory=lambda: ModelAgent(
                ReferenceDrivenTransport(),
                model=TEST_MODEL,
                agent_id="partial-test",
                max_transport_calls=3,
            ),
            agent_kind="model",
        )
    partial = caught.value.partial_evidence
    assert partial is not None
    rows = partial.rows
    assert rows[0]["schema"] == "operatebench.partial-execution.v1"
    assert rows[0]["scoring"] == "NON-SCORED"
    assert rows[-1]["classification"] == "excluded"
    assert rows[-1]["exclusion_code"] == "provider_budget"
    receipts = [row["receipt"] for row in rows if row["kind"] == "receipt"]
    assert [r["index"] for r in receipts] == list(range(len(receipts)))
    assert any(r["record_type"] == "effect_accepted" for r in receipts)
    assert any(row["kind"] == "state" for row in rows)
    assert "quality" not in rows[-1]
    assert caught.value.record.excluded
    assert len(caught.value.record.attempts) == 3


@pytest.mark.parametrize("server_error", [False, True])
def test_provider_recorder_durably_keeps_excluded_prefix(tmp_path, server_error):
    import httpx

    from operatebench.execution_ledger import read_execution_ledger
    from tests.provider_evidence_runs import model_run_with_ledger

    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    result = model_run_with_ledger(
        directory,
        load_spec(FIXTURE),
        max_calls=3 if not server_error else 60,
        faults={4: httpx.Response(503, json={"error": {"message": "private-error"}})}
        if server_error
        else None,
    )
    assert result.failure is not None
    partial = result.failure.partial_evidence
    assert partial is result.recorder.partial_evidence
    assert partial.path == result.ledger_path.with_suffix(".partial.ndjson")
    rows = [json.loads(line) for line in partial.path.read_text().splitlines()]
    assert rows == partial.rows
    assert os.stat(partial.path).st_mode & 0o777 == 0o600
    assert "private-error" not in partial.path.read_text()
    assert rows[-1]["classification"] == "excluded"
    ledger = read_execution_ledger(result.ledger_path)
    assert rows[0]["identity"]["execution_run_id"] == ledger.header.execution_run_id
    assert rows[-1]["exclusion_code"] == (
        "provider_transport" if server_error else "provider_budget"
    )
    assert any(r.get("receipt", {}).get("record_type") == "effect_accepted" for r in rows)
    assert rows[-1]["ledger_terminal"] == ledger.terminal.as_dict()
    assert rows[-1]["ledger_digest_sha256"] == ledger.ledger_digest_sha256
    assert rows[-2]["kind"] == "state"
    assert len([r for r in rows if r["kind"] == "decision"]) == 3


def test_decisions_and_refusals_survive_malformed_unknown_then_outage(tmp_path):
    import httpx

    from tests.openai_transport import function_call_item, responses_body
    from tests.provider_evidence_runs import model_run_with_ledger

    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    faults = {
        4: httpx.Response(
            200,
            json=responses_body(
                [
                    function_call_item(
                        "retrieve",
                        json.dumps(
                            {"requests": [{"tool": "unknown-inner", "arguments": {}}]}
                        ),
                    )
                ],
                model="gpt-5.6-luna",
            ),
        ),
        5: httpx.Response(
            200,
            json=responses_body(
                [function_call_item("private-unknown-outer", "{}")], model="gpt-5.6-luna"
            ),
        ),
        6: httpx.Response(503, json={"error": {"message": "private-5xx"}}),
    }
    result = model_run_with_ledger(directory, load_spec(FIXTURE), faults=faults)
    assert result.failure is not None
    rows = result.failure.partial_evidence.rows
    decisions = [r["decision"] for r in rows if r["kind"] == "decision"]
    assert len(decisions) == 5
    assert decisions[-2]["outcome"]["requests"] == [
        {"tool": "unknown-inner", "arguments": {}}
    ]
    assert decisions[-1]["outcome"] == {"kind": "MALFORMED", "code": "MODEL_UNKNOWN_TOOL"}
    receipts = [r["receipt"] for r in rows if r["kind"] == "receipt"]
    refused = [
        r
        for r in receipts
        if r["record_type"] in ("retrieval_refused", "outcome_rejected")
    ]
    assert [r["code"] for r in refused] == [
        "UNKNOWN_RETRIEVAL_TOOL",
        "MALFORMED_AGENT_OUTCOME",
    ]
    assert "private-unknown-outer" not in json.dumps(rows)
    assert "private-5xx" not in json.dumps(rows)


def test_control_budget_closure_uses_ledger_not_coarse_adapter_fault(tmp_path):
    from operatebench.execution_ledger import read_execution_ledger
    from tests.provider_evidence_runs import model_run_with_ledger

    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    result = model_run_with_ledger(directory, load_spec(FIXTURE), cap="0.000001")
    assert result.failure is not None
    ledger = read_execution_ledger(result.ledger_path)
    closure = result.failure.partial_evidence.rows[-1]
    assert closure["classification"] == ledger.status == "excluded"
    assert (
        closure["exclusion_code"] == ledger.terminal.exclusion_code == "provider_budget"
    )
    assert closure["ledger_terminal"]["totals"] == ledger.terminal.as_dict()["totals"]


def test_closed_writer_refuses_append_without_changing_prefix(tmp_path):
    from operatebench.partial_evidence import PartialExecutionEvidence

    partial = PartialExecutionEvidence({}, path=tmp_path / "partial.ndjson")
    partial.finish("excluded", "provider_budget")
    before = partial.path.read_bytes()
    with pytest.raises(ValueError, match="closed"):
        partial.receipt({"index": 0})
    assert partial.path.read_bytes() == before


def test_unknown_fault_is_not_retained_as_free_text():
    from operatebench.partial_evidence import PartialExecutionEvidence

    partial = PartialExecutionEvidence({})
    partial.finish("excluded", "private-failure-detail")
    assert partial.rows[-1]["exclusion_code"] == "unknown"
    assert "private-failure-detail" not in json.dumps(partial.rows)


def test_unsupported_serialization_never_stringifies_objects(tmp_path):
    from operatebench.jsonsafe import JsonSafetyError
    from operatebench.partial_evidence import PartialExecutionEvidence

    class Hostile:
        def __str__(self):
            pytest.fail("must not stringify")

        def __repr__(self):
            pytest.fail("must not repr")

    partial = PartialExecutionEvidence({}, path=tmp_path / "partial.ndjson")
    before = partial.path.read_bytes()
    with pytest.raises(JsonSafetyError):
        partial.receipt({"unsupported": Hostile()})
    assert partial.path.read_bytes() == before
    partial.finish("aborted")


def test_short_write_failure_keeps_prefix_and_poisoned_writer(tmp_path, monkeypatch):
    from operatebench.partial_evidence import PartialExecutionEvidence

    partial = PartialExecutionEvidence({}, path=tmp_path / "partial.ndjson")
    before = partial.path.read_bytes()
    original = os.write
    calls = []

    def failing(fd, data):
        calls.append(1)
        if len(calls) == 1:
            return original(fd, data[:7])
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(os, "write", failing)
    with pytest.raises(OSError):
        partial.receipt({"index": 0})
    assert partial.path.read_bytes().startswith(before)
    assert len(partial.rows) == 1
    with pytest.raises(ValueError, match="closed"):
        partial.receipt({"index": 1})
    partial.finish("aborted")
    assert len(calls) == 2


def test_header_binds_exact_runtime_source(tmp_path):
    import hashlib
    from pathlib import Path

    from operatebench.partial_evidence import PartialExecutionEvidence

    partial = PartialExecutionEvidence({})
    sources = partial.rows[0]["source_files_sha256"]
    assert "runner.py" in sources
    assert (
        sources["runner.py"]
        == hashlib.sha256(Path("src/operatebench/runner.py").read_bytes()).hexdigest()
    )


def test_crash_during_next_call_keeps_all_prior_decisions_and_receipts(tmp_path):
    import subprocess
    import sys

    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    script = """
import os, sys
from pathlib import Path
import tests.provider_evidence_runs as helpers
from tests.test_operatebench_provenance_binding import FIXTURE
from operatebench.domains.lettings.maintenance.spec import load_spec
original = helpers.reference_handler
def crashing(wire, **kwargs):
    inner = original(wire, **kwargs)
    def handler(request):
        if len(wire.bodies) == 3:
            os._exit(73)
        return inner(request)
    return handler
helpers.reference_handler = crashing
helpers.model_run_with_ledger(Path(sys.argv[1]), load_spec(FIXTURE))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script, str(directory)],
        env={"PYTHONPATH": "src:.", "PATH": os.environ["PATH"]},
        timeout=30,
        capture_output=True,
    )
    assert proc.returncode == 73, proc.stderr.decode()
    rows = [
        json.loads(line)
        for line in (directory / "execution_ledger.partial.ndjson")
        .read_text()
        .splitlines()
    ]
    assert len([r for r in rows if r["kind"] == "decision"]) == 3
    assert rows[-1]["kind"] == "state"
    assert not any(r["kind"] == "closure" for r in rows)
    assert any(r.get("receipt", {}).get("record_type") == "effect_accepted" for r in rows)
