"""Shared development lane: actual legacy Commerce Core, not successor stubs."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from operatebench.domains.commerce.return_refund import agents, evaluator, operation
from operatebench.domains.commerce.return_refund import spec as specs

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "examples/operatebench/commerce_return_refund_v0_1.yaml"


def load_spec(path):
    raw = specs.load_spec(path)
    return SimpleNamespace(
        operation_id=raw.operation_id,
        content_digest=raw.spec_digest_sha256,
        scenarios={s: raw.scenario(s) for s in raw.scenario_ids()},
        raw=raw,
    )


def build_domain(spec, scenario):
    return operation.ReturnRefundOperation(spec.raw, scenario)


def grade(episode, spec, scenario):
    result = evaluator.evaluate(episode, True, spec.raw.scenario(scenario))
    return {
        "reliable": result.reliable,
        "dimensions": {d.name: d.ok for d in result.dimensions},
        "findings": [],
        "terminal_outcome": result.terminal_outcome,
    }


def profile(spec, scenario):
    return {
        "profile_id": "legacy-synthetic",
        "profile_version": "0.1.0",
        "profile_digest": spec.content_digest,
        "jurisdiction": "SYNTHETIC",
    }


def factories():
    from operatebench.sdk.development_runtime import DevelopmentFactories

    return DevelopmentFactories(
        pack_id="test.commerce",
        pack_version="0.1.0",
        operation_type="commerce.return_refund",
        load_spec=load_spec,
        build_domain=build_domain,
        build_agent=agents.build_agent,
        evaluate_episode=grade,
        profile_identity=profile,
        implementation_paths=(Path(operation.__file__).parent,),
        dispatch_failures=lambda spec, scenario: (
            spec.scenarios[scenario].dispatch_failures
        ),
    )


@pytest.mark.parametrize("scenario", ["V1", "V2", "V3"])
def test_real_core_record_and_replay(scenario):
    from operatebench.sdk import development_runtime as runtime

    record = runtime.run(factories(), SPEC, scenario, "reference")
    assert record["evaluation"]["reliable"] is True
    assert record["episode"]["events"] and record["episode"]["trajectory"]
    replay = runtime.replay(factories(), SPEC, record)
    assert replay["consistent"] is True
    assert replay["provider_calls"] == 0
    assert replay["decisions_consumed"] == len(record["decisions"])


@pytest.mark.parametrize("scenario", ["V1", "V2", "V3"])
def test_mock_model_roundtrip_write_once_and_zero_provider(
    tmp_path, scenario, monkeypatch
):
    from operatebench.sdk import development_runtime as runtime
    from operatebench.sdk.development_record import read_record

    path = tmp_path / "run.json"
    trusted = factories()
    record = runtime.run_mock_model(trusted, SPEC, scenario, output=path)
    assert record["agent"]["kind"] == "reference_driven_mock_model"
    assert record["evaluation"]["reliable"] is True
    assert record["agent"]["mock_calls"] == len(record["decisions"])
    assert read_record(path) == record
    # Neither the real ModelAgent's transport nor the original factory is replayed.
    from operatebench.agents.model import ModelAgent

    monkeypatch.setattr(ModelAgent, "_send", lambda *a: pytest.fail("provider reached"))
    monkeypatch.setattr(agents, "build_agent", lambda *a: pytest.fail("agent rerun"))
    result = runtime.replay(trusted, SPEC, read_record(path))
    assert result["provider_calls"] == 0
    assert result["decisions_consumed"] == len(record["decisions"])


def test_record_rejects_extra_top_level_field():
    from operatebench.sdk import development_runtime as runtime

    record = runtime.run(factories(), SPEC, "V2")
    record["untrusted"] = True
    with pytest.raises(runtime.DevelopmentRuntimeError):
        runtime.replay(factories(), SPEC, record)


def reseal(record):
    from operatebench.agents.transport import content_digest

    record["record_digest"] = content_digest(
        {k: v for k, v in record.items() if k != "record_digest"}
    )
    return record


def test_agent_relabel_is_not_playback():
    from operatebench.sdk import development_runtime as runtime

    record = runtime.run(factories(), SPEC, "V2")
    record["agent"]["agent_id"] = "complete_early"
    record["identity"]["agent_id"] = "complete_early"
    with pytest.raises(runtime.DevelopmentRuntimeError):
        runtime.replay(factories(), SPEC, reseal(record))


def test_policy_disclosure_is_fresh_and_unsupported_escalate_never_reaches_core():
    from dataclasses import replace

    from operatebench.core.outcomes import Escalate, Wait
    from operatebench.sdk import development_runtime as runtime

    seen = []

    class Agent:
        agent_id = "probe"

        def begin_episode(self, identity):
            pass

        def decide(self, observation):
            support = observation.policy["development_outcome_support"]
            seen.append(support["supported"][:])
            support["supported"].clear()
            if len(seen) == 1:
                return Wait("probe", (), 1)
            return Escalate("forbidden", "review", (), 1, "unsupported")

    def build_probe(agent_id):
        return Agent()

    trusted = replace(factories(), build_agent=build_probe)
    with pytest.raises(runtime.DevelopmentRuntimeError, match="unsupported"):
        runtime.run(trusted, SPEC, "V2", "probe")
    assert seen == [["ACT", "WAIT", "COMPLETE", "RETRIEVE"]] * 2


@pytest.fixture(scope="module")
def fixed_record():
    from operatebench.sdk.development_runtime import run

    return run(factories(), SPEC, "V2")


@pytest.fixture(scope="module")
def model_record():
    from operatebench.sdk.development_runtime import run_mock_model

    return run_mock_model(factories(), SPEC, "V1")


@pytest.mark.parametrize(
    "path,value",
    [
        (("binding", "profile", "profile_digest"), "0" * 64),
        (("binding", "profile", "jurisdiction"), "UK"),
        (("binding", "scenario_id"), "V1"),
        (("binding", "scenario_id"), []),
        (("binding", "runtime_digest"), "0" * 64),
        (("binding", "implementation_digest"), "0" * 64),
        (("binding", "spec_digest"), "0" * 64),
        (("identity", "operation_instance_id"), "not-an-instance"),
        (("identity", "scenario_id"), "V3"),
        (("episode", "final_state", "injected"), True),
        (("episode", "events"), []),
        (("episode", "trajectory"), []),
        (("episode", "replay_final"), 1),
        (("evaluation", "reliable"), 1),
        (("evaluation", "dimensions", "invented"), True),
        (("decisions", 0, "observation_digest_sha256"), "0" * 64),
        (("decisions", 0, "invocation_index"), True),
        (("decisions", 0, "unknown"), 0),
        (("decisions", 0, "outcome", "unknown"), 0),
        (("decisions", 0, "outcome", "kind"), "invented"),
    ],
)
def test_resealed_mutations_fail_closed(fixed_record, path, value):
    from operatebench.sdk import development_runtime as runtime

    record = copy.deepcopy(fixed_record)
    parent = record
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(runtime.DevelopmentRuntimeError):
        runtime.replay(factories(), SPEC, reseal(record))


@pytest.mark.parametrize("mutation", ["drop", "append", "reorder"])
def test_exact_tape_coverage(fixed_record, mutation):
    from operatebench.sdk import development_runtime as runtime

    record = copy.deepcopy(fixed_record)
    if mutation == "drop":
        record["decisions"].pop()
    elif mutation == "append":
        record["decisions"].append(copy.deepcopy(record["decisions"][-1]))
    else:
        record["decisions"][:2] = reversed(record["decisions"][:2])
    with pytest.raises(runtime.DevelopmentRuntimeError):
        runtime.replay(factories(), SPEC, reseal(record))


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_request",
        "missing_request",
        "wrong_request",
        "relabel_reference",
        "protocol",
        "model",
        "ceiling",
    ],
)
def test_exact_mock_request_coverage(model_record, mutation):
    from operatebench.sdk import development_runtime as runtime

    record = copy.deepcopy(model_record)
    agent = record["agent"]
    if mutation == "extra_request":
        agent["request_digests"].append("0" * 64)
    elif mutation == "missing_request":
        agent["request_digests"].pop()
    elif mutation == "wrong_request":
        agent["request_digests"][0] = "0" * 64
    elif mutation == "relabel_reference":
        agent["reference_agent_id"] = "complete_early"
    elif mutation == "protocol":
        agent["protocol_version"] = "invented"
    elif mutation == "model":
        agent["model"] = "real-model-claim"
    else:
        agent["max_output_tokens"] = True
    with pytest.raises(runtime.DevelopmentRuntimeError):
        runtime.replay(factories(), SPEC, reseal(record))


def test_write_once_safe_paths_and_checksum(tmp_path, fixed_record):
    from operatebench.sdk.development_record import read_record, write_record
    from operatebench.sdk.development_runtime import DevelopmentRuntimeError

    path = tmp_path / "valid.json"
    write_record(path, fixed_record)
    assert read_record(path) == fixed_record
    before = path.read_bytes()
    with pytest.raises(DevelopmentRuntimeError, match="exists"):
        write_record(path, fixed_record)
    assert path.read_bytes() == before
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(DevelopmentRuntimeError):
        write_record(link, fixed_record)
    directory_link = tmp_path / "redirect"
    directory_link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(DevelopmentRuntimeError):
        write_record(directory_link / "new.json", fixed_record)
    assert not (tmp_path / "new.json").exists()
    changed = copy.deepcopy(fixed_record)
    changed["scope"] = "official"
    with pytest.raises(DevelopmentRuntimeError):
        write_record(tmp_path / "invalid.json", changed)
    assert not (tmp_path / "invalid.json").exists()


@pytest.mark.parametrize(
    "body",
    [
        b'{"x":1,"x":2}',
        b'{"x":{"a":1,"a":2}}',
        b'{"x": NaN}',
        b'{"x": 1e99999}',
        b"null",
        b"[]",
        b"{} trailing",
        b"\xff",
        b"[" * 1000 + b"]" * 1000,
    ],
)
def test_strict_json_reader(tmp_path, body):
    from operatebench.sdk.development_record import read_record
    from operatebench.sdk.development_runtime import DevelopmentRuntimeError

    path = tmp_path / "invalid.json"
    path.write_bytes(body)
    with pytest.raises(DevelopmentRuntimeError):
        read_record(path)


def test_bounded_reads_and_in_memory_json(tmp_path, fixed_record, monkeypatch):
    from operatebench.sdk import development_record as codec
    from operatebench.sdk.development_runtime import DevelopmentRuntimeError

    monkeypatch.setattr(codec, "MAX_RUN_BYTES", 128)
    path = tmp_path / "large.json"
    path.write_bytes(b" " * 129)
    with pytest.raises(DevelopmentRuntimeError):
        codec.read_record(path)
    with pytest.raises(DevelopmentRuntimeError):
        codec.write_record(tmp_path / "output.json", fixed_record)
    assert not (tmp_path / "output.json").exists()
    with pytest.raises(DevelopmentRuntimeError):
        codec.canonical({"a": (1, 2)})
    with pytest.raises(DevelopmentRuntimeError):
        codec.canonical({1: "not a string"})
    with pytest.raises(DevelopmentRuntimeError):
        codec.canonical({"a": float("nan")})


def test_negative_is_replayed_but_not_promoted_to_reliable():
    from operatebench.sdk import development_runtime as runtime

    record = runtime.run(factories(), SPEC, "V1", "complete_early")
    assert record["evaluation"]["reliable"] is False
    result = runtime.replay(factories(), SPEC, record)
    assert result["consistent"] is True
    assert result["evaluation"]["reliable"] is False


@pytest.mark.parametrize("method", ["run", "run_mock_model"])
@pytest.mark.parametrize("kind", ["ask", "escalate"])
def test_unsupported_outcomes_abort_without_output(tmp_path, method, kind):
    from dataclasses import replace

    from operatebench.core.outcomes import Ask, Escalate, Wait
    from operatebench.sdk import development_runtime as runtime

    class Unsupported:
        agent_id = "unsupported"

        def begin_episode(self, identity):
            pass

        def decide(self, observation):
            assert (
                "ESCALATE"
                in observation.policy["development_outcome_support"]["unsupported"]
            )
            if kind == "ask":
                return Ask(
                    "customer", "unsupported", Wait("reason", (), 1), "correlation"
                )
            return Escalate("checkpoint", "review", (), 1, "reason")

    def build_unsupported(agent_id):
        return Unsupported()

    trusted = replace(factories(), build_agent=build_unsupported)
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(runtime.DevelopmentRuntimeError, match="unsupported"):
        getattr(runtime, method)(trusted, SPEC, "V2", "unsupported", output=output)
    assert not output.exists()


def test_overdeep_json_is_refused_before_parsing(tmp_path, monkeypatch):
    from operatebench.sdk import development_record as codec
    from operatebench.sdk.development_runtime import DevelopmentRuntimeError

    path = tmp_path / "deep.json"
    path.write_bytes(b"[" * 65 + b"0" + b"]" * 65)
    monkeypatch.setattr(
        codec.json, "loads", lambda *a, **kw: pytest.fail("deep JSON reached parser")
    )
    with pytest.raises(DevelopmentRuntimeError):
        codec.read_record(path)


def test_model_requests_are_json_native_and_only_current_public_view(monkeypatch):
    from operatebench.agents.transport import content_digest
    from operatebench.sdk import development_runtime as runtime

    original = runtime._ReferenceTransport.send
    seen = []

    def inspect_request(self, request):
        assert json.loads(json.dumps(request.prompt)) == request.prompt
        public = request.prompt["observation"]
        assert (
            not {"scenario_id", "expected_terminal", "state", "future_events"}
            & public.keys()
        )
        assert public["policy"]["development_outcome_support"]["unsupported"] == [
            "ASK",
            "ESCALATE",
        ]
        assert content_digest(public) == request.observation_digest_sha256
        seen.append(request.request_digest_sha256)
        return original(self, request)

    monkeypatch.setattr(runtime._ReferenceTransport, "send", inspect_request)
    record = runtime.run_mock_model(factories(), SPEC, "V2")
    assert seen == record["agent"]["request_digests"]


def test_write_at_byte_limit_is_readable(tmp_path, fixed_record, monkeypatch):
    from operatebench.sdk import development_record as codec

    monkeypatch.setattr(codec, "MAX_RUN_BYTES", len(codec.canonical(fixed_record)))
    path = tmp_path / "boundary.json"
    codec.write_record(path, fixed_record)
    assert codec.read_record(path) == fixed_record


@pytest.mark.parametrize(
    "callback",
    [
        "load_spec",
        "build_domain",
        "build_agent",
        "evaluate_episode",
        "profile_identity",
        "dispatch_failures",
    ],
)
def test_every_callback_identity_is_bound(fixed_record, callback):
    from dataclasses import replace

    from operatebench.sdk import development_runtime as runtime

    trusted = factories()
    original = getattr(trusted, callback)

    def replacement(*args):
        return original(*args)

    changed = replace(trusted, **{callback: replacement})
    with pytest.raises(runtime.DevelopmentRuntimeError, match="binding mismatch"):
        runtime.replay(changed, SPEC, fixed_record)


def test_transitive_domain_resource_is_bound(tmp_path):
    from dataclasses import replace

    from operatebench.sdk import development_runtime as runtime

    resource = tmp_path / "policy.txt"
    resource.write_text("original policy")
    trusted = factories()
    trusted = replace(
        trusted, implementation_paths=(*trusted.implementation_paths, resource)
    )
    record = runtime.run(trusted, SPEC, "V2")
    resource.write_text("changed policy")
    with pytest.raises(runtime.DevelopmentRuntimeError, match="binding mismatch"):
        runtime.replay(trusted, SPEC, record)
