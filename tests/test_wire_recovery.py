"""Offline recovery controls; never evidence of a live provider limit."""

import hashlib
import json

import httpx
import pytest

from operatebench.agents.transport import ProviderFailure
from tests.test_three_flow_providers import fixture
from tools.three_flow_runtime import run_trial


@pytest.mark.parametrize(
    "body",
    [
        b"{broken",
        b'{"message":"max_tokens must be <= 123. private"}',
        b'{"message":"max_tokens must be <= 123","message":"secret"}',
        b'{"error":{"code":"https://private.invalid","param":"Authorization","message":"secret"}}',
        b'{"message":"max_tokens must be <= 9999999999"}',
        b'{"message":"context_limit must be <= -1"}',
        b"[" * 2000,
        b" " * 16385,
    ],
)
def test_unrecognized_or_adversarial(body):
    from tools.three_flow_errors import http_error, typed_http_facts

    assert typed_http_facts(body) == {}
    projection = http_error(400, "chat_completions", None, body=body)
    assert projection == {
        "representation": "sanitized_http_error_v2",
        "diagnostic": "http_non_success",
        "body_retained": False,
        "operation": "chat_completions",
        "http_status": 400,
        "reason": "unknown",
        "retry_after_seconds": None,
        "automatic_resubmission": False,
    }


@pytest.mark.parametrize("parameter", ["max_tokens", "context_limit"])
def test_numeric_grammar(parameter):
    from tools.three_flow_errors import http_error, typed_http_facts

    body = json.dumps({"message": f"{parameter} must be <= 12345"}).encode()
    facts = {"limit_parameter": parameter, "reported_limit": 12345}
    assert typed_http_facts(body) == facts
    assert http_error(400, "chat_completions", None, body=body) == {
        "representation": "sanitized_http_error_v2",
        "diagnostic": "http_non_success",
        "body_retained": False,
        "operation": "chat_completions",
        "http_status": 400,
        "reason": "recognized_typed_error",
        "retry_after_seconds": None,
        "automatic_resubmission": False,
        **facts,
    }


@pytest.mark.parametrize("recognized", [True, False])
def test_real_mistral_sdk_400_one_dispatch_private_unknown_liability(
    tmp_path, recognized
):
    secret = "synthetic-private-key-and-instruction"
    body = json.dumps(
        {
            "error": {
                "message": "max_tokens must be less than or equal to 12345"
                if recognized
                else secret,
                "code": "invalid_parameter" if recognized else secret,
                "param": "max_tokens" if recognized else secret,
                "private": secret,
            }
        }
    ).encode()

    def wrapper(mock):
        return lambda request: httpx.Response(400, content=body)

    budget, _, transport, trial = fixture(tmp_path, "mistral", handler=wrapper)
    try:
        with pytest.raises(ProviderFailure) as caught:
            run_trial(trial, transport, output=tmp_path / "trial")
        assert caught.value.fault == "provider_request_rejected"
        assert len(transport.wire.captures) == 1
        projection = transport.wire.captures[0]["response_projection"]
        assert "response_body_sha256" not in projection
        assert hashlib.sha256(body).hexdigest() not in json.dumps(transport.wire.captures)
        assert projection["automatic_resubmission"] is False
        assert projection["body_retained"] is False
        assert projection["reason"] == (
            "recognized_typed_error" if recognized else "unknown"
        )
        if recognized:
            assert projection["reported_limit"] == 12345
            assert projection["limit_parameter"] == "max_tokens"
        assert secret not in json.dumps(transport.wire.captures)
        assert secret not in str(caught.value)
        assert secret not in (tmp_path / "trial" / "partial.ndjson").read_text()
        assert budget.totals()["unknown"] > 0
    finally:
        transport.close()
        budget.close()


def test_selected_river_shared_lane(tmp_path, monkeypatch):
    from tests.test_selected_six_http import fixture as selected_fixture
    from tools import three_flow_http as http
    from tools.three_flow_errors import ProviderPacer
    from tools.three_flow_selected import selected_registry

    now = [0.0]
    waits = []

    def sleep(delay):
        waits.append(delay)
        now[0] += delay

    pacer = ProviderPacer(clock=lambda: now[0], sleep=sleep)
    monkeypatch.setattr(http, "RIVER_HTTP_PACER", pacer)
    lanes = []
    for index, model in enumerate(("GLM-5.3", "moonshotai/Kimi-K3")):
        parent = tmp_path / str(index)
        parent.mkdir(mode=0o700)
        budget, transport, trial, _ = selected_fixture(
            parent, "river_http", model, river_pacer=pacer
        )
        lanes.append((budget, transport, trial, parent))
        assert transport.wire.pacer is pacer
        assert transport.settings["pacing_scope"] == "process-wide-river-http"
        registry = next(
            r for r in selected_registry("paced-safe-errors-v1") if r["model"] == model
        )
        assert registry["mapping"] == transport.request_mapping
        assert registry["request_settings"]["minimum_spacing_seconds"] == 1
    calls = []
    try:
        for index, (budget, transport, trial, parent) in enumerate(lanes):

            def respond(request, index=index):
                calls.append(now[0])
                assert json.loads(request.content)["max_tokens"] <= 30000
                return httpx.Response(
                    429 if index == 0 else 400,
                    json={"message": "synthetic private error"},
                    headers={"Retry-After": "4"} if index == 0 else {},
                )

            transport.wire.inner = httpx.MockTransport(respond)
            with pytest.raises(ProviderFailure):
                run_trial(trial, transport, output=parent / "trial")
            assert len(transport.wire.captures) == 1
            assert budget.totals()["unknown"] > 0
        pacer.call(lambda: calls.append(now[0]) or httpx.Response(200))
        assert calls == [0, 4, 5]
        assert waits == [4, 1]
    finally:
        for budget, transport, _, _ in lanes:
            transport.close()
            budget.close()
