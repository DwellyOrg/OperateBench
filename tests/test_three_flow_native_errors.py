"""Actual installed generated stub, raw RPC faults, no network capability."""

import json
import sys

import pytest

from operatebench.agents.transport import ProviderFailure
from tests.test_three_flow_river import fixture
from tools.three_flow_runtime import run_trial

pytest_plugins = ["tests.test_three_flow_river"]

pytestmark = pytest.mark.skipif(sys.version_info < (3, 12), reason="installed River SDK")


@pytest.mark.parametrize("phase", ["submit", "poll", "failed_struct"])
def test_raw_rpc_failure_closes_partial_and_retains_liability(
    tmp_path, assets, monkeypatch, phase
):
    import grpc

    budget, channel, transport, trial = fixture(tmp_path, assets)
    secret = "provider-secret-must-not-retain"
    calls = []
    original = channel.unary_unary

    class Refusal(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.INVALID_ARGUMENT

        def details(self):
            raise AssertionError("no raw text inspection")

        def __str__(self):
            raise AssertionError("no raw text conversion")

    def factory(path, **kwargs):
        rpc = original(path, **kwargs)

        def invoke(request, **options):
            calls.append(path)
            if phase == "submit" or (phase == "poll" and path.endswith("RetrieveFuture")):
                raise Refusal()
            if phase == "failed_struct" and path.endswith("RetrieveFuture"):
                failed = channel.pb.RequestFailedResponse(
                    error_category=secret,
                    message=secret,
                    details={"reason": secret, "code": 220, "authorization": secret},
                )
                response = channel.pb.RetrieveFutureResponse(failed=failed)
                return kwargs["response_deserializer"](response.SerializeToString())
            return rpc(request, **options)

        return invoke

    monkeypatch.setattr(channel, "unary_unary", factory)
    try:
        expected = (
            "provider_server_error"
            if phase == "failed_struct"
            else "provider_request_rejected"
        )
        with pytest.raises(ProviderFailure) as caught:
            run_trial(trial, transport, output=tmp_path / "trial")
        assert caught.value.fault == expected
        assert transport.last_turn["classification"] == expected
        text = (tmp_path / "trial" / "partial.ndjson").read_text()
        assert secret not in text
        rows = [json.loads(line) for line in text.splitlines()]
        assert rows[-1]["classification"] == "excluded"
        assert rows[-1]["exclusion_code"] == expected
        projection = transport.wire.captures[-1]["response_projection"]
        assert projection["reason"] == "unknown"
        assert projection["automatic_resubmission"] is False
        if phase == "failed_struct":
            assert projection["structured_details_present"] is True
            assert "220" not in json.dumps(projection)
        else:
            assert projection["grpc_status"] == "INVALID_ARGUMENT"
            assert projection["grpc_status_code"] == 3
        assert any(
            row["kind"] == "wire_response"
            and row.get("response_projection") == projection
            for row in rows
        )
        assert len(calls) == (1 if phase == "submit" else 2)
        assert sum(path.endswith("InferenceGenerate") for path in calls) == 1
        assert budget.totals()["unknown"] > 0
        assert budget.totals()["pending"] == 0
        assert len(budget.records) == 1
        assert not (tmp_path / "trial" / "record.json").exists()
    finally:
        transport.close()
        budget.close()
