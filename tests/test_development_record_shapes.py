"""Development-owned envelopes are strict; domain JSON remains extensible."""

import copy
import json
from dataclasses import replace

import pytest

from operatebench.agents.transport import ForbiddenTransport
from operatebench.domains.commerce.return_refund.agents import NEGATIVE_AGENTS
from operatebench.sdk import development_record as codec
from operatebench.sdk import development_runtime as runtime
from tests.test_development_runtime import SPEC, factories, grade, reseal


@pytest.fixture(scope="module")
def genuine_record():
    return runtime.run(factories(), SPEC, "V2")


@pytest.mark.parametrize("boundary", ["validate", "write", "read"])
@pytest.mark.parametrize(
    "field,value",
    [
        pytest.param("episode", None, id="null-episode"),
        pytest.param("episode", {"invented": True}, id="unknown-episode"),
        pytest.param("evaluation", True, id="boolean-evaluation"),
    ],
)
def test_review_holes_refused(genuine_record, tmp_path, boundary, field, value):
    record = copy.deepcopy(genuine_record)
    record[field] = value
    assert_refused(record, tmp_path, boundary, field)


def assert_refused(record, tmp_path, boundary, envelope):
    reseal(record)
    path = tmp_path / "record.json"
    if boundary == "read":
        # Deliberately bypass the writer: fixed read refusal is expected even
        # when independently supplied raw JSON has the correct content digest.
        path.write_bytes(json.dumps(record, allow_nan=False).encode("utf-8"))
    with pytest.raises(runtime.DevelopmentRuntimeError, match=envelope):
        if boundary == "read":
            codec.read_record(path)
        elif boundary == "write":
            codec.write_record(path, record)
        else:
            codec.validate_record(record)
    if boundary == "write":
        assert not path.exists()


@pytest.mark.parametrize("boundary", ["write", "read"])
@pytest.mark.parametrize("envelope", ["episode", "evaluation"])
@pytest.mark.parametrize("change", ["missing", "unknown"])
def test_exact_envelope_fields(genuine_record, tmp_path, boundary, envelope, change):
    record = copy.deepcopy(genuine_record)
    if change == "missing":
        # Exercise every required field, not just a representative member.
        for field in genuine_record[envelope]:
            candidate = copy.deepcopy(record)
            del candidate[envelope][field]
            assert_refused(candidate, tmp_path, boundary, envelope)
    else:
        record[envelope]["invented"] = True
        assert_refused(record, tmp_path, boundary, envelope)


@pytest.mark.parametrize("boundary", ["write", "read"])
@pytest.mark.parametrize(
    "envelope,field,value",
    [
        ("episode", "status", None),
        ("episode", "terminal_outcome", False),
        ("episode", "replay_final", 1),
        ("episode", "started_at", 1),
        ("episode", "ended_at", None),
        ("episode", "simulated_minutes", True),
        ("episode", "simulated_minutes", 1.5),
        ("episode", "invocations", False),
        ("episode", "invocations", "1"),
        ("episode", "final_state", []),
        ("episode", "final_state_digest_sha256", "not-a-digest"),
        ("episode", "trajectory_digest_sha256", True),
        ("episode", "events", {}),
        ("episode", "events", [None]),
        ("episode", "trajectory", {}),
        ("episode", "trajectory", [True]),
        ("evaluation", "reliable", 1),
        ("evaluation", "dimensions", []),
        ("evaluation", "dimensions", {}),
        ("evaluation", "dimensions", {"custom": 1}),
        ("evaluation", "findings", {}),
        ("evaluation", "findings", [None]),
        ("evaluation", "terminal_outcome", None),
    ],
)
def test_envelope_types(genuine_record, tmp_path, boundary, envelope, field, value):
    record = copy.deepcopy(genuine_record)
    record[envelope][field] = value
    assert_refused(record, tmp_path, boundary, envelope)


@pytest.mark.parametrize("method", ["run", "run_mock_model"])
@pytest.mark.parametrize("scenario", ["V1", "V2", "V3"])
def test_genuine_reference_roundtrip(tmp_path, monkeypatch, method, scenario):
    check_genuine_roundtrip(tmp_path, monkeypatch, method, scenario, "reference", True)


@pytest.mark.parametrize("method", ["run", "run_mock_model"])
@pytest.mark.parametrize(
    "agent,scenario",
    [
        (entry.agent_id, scenario)
        for entry in NEGATIVE_AGENTS
        for scenario in entry.scenarios
    ],
)
def test_genuine_negative_roundtrip(tmp_path, monkeypatch, method, agent, scenario):
    check_genuine_roundtrip(tmp_path, monkeypatch, method, scenario, agent, False)


def check_genuine_roundtrip(tmp_path, monkeypatch, method, scenario, agent, reliable):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        pytest.fail("provider reached")

    monkeypatch.setattr(ForbiddenTransport, "send", forbidden)
    path = tmp_path / "genuine.json"
    trusted = factories()
    record = getattr(runtime, method)(trusted, SPEC, scenario, agent, output=path)
    observed = codec.read_record(path)
    assert observed == record
    assert observed["evaluation"]["reliable"] is reliable
    replay = runtime.replay(trusted, SPEC, observed)
    assert replay["consistent"] is True
    assert replay["evaluation"]["reliable"] is reliable
    assert replay["provider_calls"] == 0
    assert replay["decisions_consumed"] == len(record["decisions"])
    assert calls == []


def nested_grade(episode, spec, scenario):
    result = grade(episode, spec, scenario)
    nested = {}
    for _ in range(60):
        nested = {"wrapper": nested}
    result["findings"] = [
        nested,
        {"arbitrary": [None, True, 7, 1.5, "text", {"custom": []}]},
    ]
    result["dimensions"]["custom_dimension"] = True
    return result


def test_genuine_nested_evaluator_roundtrip(tmp_path):
    trusted = replace(factories(), evaluate_episode=nested_grade)
    path = tmp_path / "nested.json"
    record = runtime.run(trusted, SPEC, "V2", output=path)
    assert codec.read_record(path) == record
    assert runtime.replay(trusted, SPEC, record)["consistent"] is True


def test_domain_payloads_remain_extensible(genuine_record, tmp_path):
    record = copy.deepcopy(genuine_record)
    payload = {"custom": [None, True, 1, 1.5, "text", {"nested": []}]}
    record["episode"]["final_state"] = payload
    record["episode"]["events"] = [payload]
    record["episode"]["trajectory"] = [payload]
    record["episode"]["terminal_outcome"] = None
    record["evaluation"]["findings"] = [payload]
    path = tmp_path / "extensible.json"
    codec.write_record(path, reseal(record))
    assert codec.read_record(path) == record
    # Codec shape admission is not semantic endorsement of altered domain data.
    with pytest.raises(runtime.DevelopmentRuntimeError, match="full episode differs"):
        runtime.replay(factories(), SPEC, record)
