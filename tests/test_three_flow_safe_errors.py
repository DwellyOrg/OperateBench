"""Synthetic-only provider errors; no credentials or remote dispatch."""

import json

import httpx
import pytest

from operatebench.agents.transport import ProviderFailure
from tests.test_three_flow_providers import fixture
from tools.three_flow_runtime import run_trial


def test_mistral_error_projection_and_classification(tmp_path):
    secret = "reflected-credential-do-not-retain"

    def wrapper(mock):
        return lambda request: httpx.Response(
            429,
            json={"message": secret, "type": secret, "code": secret},
            headers={"Retry-After": "3", "Authorization": secret, "X-Request-ID": secret},
        )

    budget, _mock, transport, trial = fixture(tmp_path, "mistral", handler=wrapper)
    try:
        with pytest.raises(ProviderFailure) as caught:
            run_trial(trial, transport, output=tmp_path / "trial")
        assert caught.value.fault == "provider_rate_limited"
        projection = transport.wire.captures[-1]["response_projection"]
        assert projection["http_status"] == 429
        assert projection["operation"] == "chat_completions"
        assert projection["reason"] == "unknown"
        assert projection["retry_after_seconds"] == 3
        assert projection["automatic_resubmission"] is False
        assert secret not in json.dumps(projection)
        assert secret not in (tmp_path / "trial" / "partial.ndjson").read_text()
        assert len(transport.wire.captures) == 1
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("kind", ["response_invalid", "cost", "bug"])
def test_campaign_preserves_known_faults_and_internal_bugs(tmp_path, monkeypatch, kind):
    from operatebench.providers.cost import CostCapExceededError
    from operatebench.providers.wire import wire_invalid

    budget, _mock, transport, trial = fixture(tmp_path, "mistral")
    error = {
        "response_invalid": wire_invalid("safe test detail"),
        "cost": CostCapExceededError("safe test detail"),
        "bug": KeyError("private bug detail"),
    }[kind]

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(
        transport.guard if kind == "cost" else transport,
        "authorize" if kind == "cost" else "_dispatch",
        fail,
    )
    try:
        with pytest.raises(KeyError if kind == "bug" else ProviderFailure) as caught:
            run_trial(trial, transport, output=tmp_path / "trial")
        if kind != "bug":
            assert caught.value.fault == (
                "provider_budget" if kind == "cost" else "provider_response_invalid"
            )
        assert (
            "private bug detail"
            not in (tmp_path / "trial" / "partial.ndjson").read_text()
        )
    finally:
        transport.close()
        budget.close()


def test_two_lanes_share_pacing_and_retry_after_without_resubmission(
    tmp_path, monkeypatch
):
    import threading
    from dataclasses import replace

    from tools import three_flow_errors as errors
    from tools import three_flow_http as http
    from tools.three_flow_runtime import provider_identity

    assert hasattr(errors, "ProviderPacer")
    now = [0.0]
    waits = []
    calls = []
    first_entered = threading.Event()
    second_started = threading.Event()

    def sleep(delay):
        assert delay > 0
        waits.append(delay)
        now[0] += delay

    pacer = errors.ProviderPacer(clock=lambda: now[0], sleep=sleep)
    monkeypatch.setattr(http, "MISTRAL_PACER", pacer)
    lanes = []
    for index in range(2):
        parent = tmp_path / str(index)
        parent.mkdir(mode=0o700)

        def wrapper(mock, index=index):
            def respond(request):
                calls.append((index, now[0]))
                if index == 0:
                    first_entered.set()
                    assert second_started.wait(5)
                    return httpx.Response(
                        429, json={"message": "unknown"}, headers={"Retry-After": "2"}
                    )
                return mock.handle(request)

            return respond

        budget, _mock, transport, trial = fixture(
            parent, "mistral", handler=wrapper, transport_policy="paced-safe-errors-v1"
        )
        trial = replace(trial, provider_binding=provider_identity(transport))
        lanes.append((budget, transport, trial, parent))

    outcomes = []

    def run(index):
        if index == 1:
            assert first_entered.wait(5)
            second_started.set()
        budget, transport, trial, parent = lanes[index]
        try:
            run_trial(trial, transport, output=parent / "trial")
            outcomes.append("scored")
        except ProviderFailure as exc:
            outcomes.append(exc.fault)
        finally:
            transport.close()
            budget.close()

    threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(15)
        assert not thread.is_alive()
    assert sorted(outcomes) == ["provider_rate_limited", "scored"]
    assert calls[:2] == [(0, 0), (1, 2)]
    assert sum(i == 0 for i, _ in calls) == 1
    assert waits[0] == 2


def test_installed_river_struct_is_projected_without_values():
    import sys

    if sys.version_info < (3, 12):
        pytest.skip("River SDK requires Python >=3.12")
    from tools.three_flow_errors import river_failed
    from tools.three_flow_river import sdk

    pb, _ = sdk()
    secret = "reflected-credential"
    error = pb.RequestFailedResponse(
        error_category=secret,
        message=secret,
        details={"reason": secret, "authorization": secret},
    )
    projection = river_failed(error)
    assert projection["structured_details_present"] is True
    assert projection["error_category"] == "unknown"
    assert secret not in json.dumps(projection)


def test_unknown_grpc_status_is_not_reflected():
    grpc = pytest.importorskip("grpc")
    from tools.three_flow_errors import grpc_error

    class Unknown(grpc.RpcError):
        def code(self):
            return "reflected-credential"

    projection = grpc_error(Unknown("reflected-credential"), "https://credential.invalid")
    assert projection["grpc_status"] == "UNKNOWN"
    assert projection["operation"] == "unknown"
    assert "credential" not in json.dumps(projection)


def test_real_grpc_error_projection_does_not_read_text():
    grpc = pytest.importorskip("grpc")
    from tools import three_flow_errors as errors

    class Refusal(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.INVALID_ARGUMENT

        def details(self):
            raise AssertionError("must not inspect provider text")

        def __str__(self):
            raise AssertionError("must not stringify error")

    assert hasattr(errors, "grpc_error")
    projection = errors.grpc_error(Refusal(), "InferenceGenerate")
    assert projection["grpc_status"] == "INVALID_ARGUMENT"
    assert projection["reason"] == "unknown"
    assert projection["operation"] == "InferenceGenerate"
    assert projection["automatic_resubmission"] is False


@pytest.mark.parametrize(
    "value",
    [
        "NaN",
        "inf",
        "-1",
        "1.1",
        "86401",
        "9" * 1000,
        True,
        3,
        None,
        "credential",
        "\uff11\uff12",
    ],
)
def test_retry_after_rejects_untrusted_values(value):
    from tools.three_flow_errors import retry_after_seconds

    assert retry_after_seconds(value) is None
