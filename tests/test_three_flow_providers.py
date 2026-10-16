"""Network-blocked SDK-wire tests for the successor HTTP transports."""

import copy
import hashlib
import json
import socket
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest

from operatebench.agents.playback import PlaybackError
from operatebench.agents.pricing import LifecyclePricingPolicy
from operatebench.agents.transport import ProviderFailure, content_digest
from tools.aggregate_budget import SharedGuard
from tools.three_flow_budget import CampaignBudget
from tools.three_flow_http import MODELS, BudgetWire, HTTPCampaignTransport
from tools.three_flow_mock import SDKMockTransport
from tools.three_flow_runtime import Trial, provider_identity, replay_trial, run_trial


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def fixture(
    tmp_path,
    provider,
    *,
    handler=None,
    model=None,
    flow="commerce",
    scenario="UK_NORMAL",
    mode_profile="legacy-v1",
    transport_policy="legacy-v1",
    api_key="offline-not-a-credential",
):
    tmp_path.chmod(0o700)
    budget = CampaignBudget(
        tmp_path, campaign_id="test", cap=Decimal("1000"), create=True
    )
    maximum = 256000 if provider == "mistral" else 128000
    guard = SharedGuard(
        budget=budget,
        cell="trial",
        policy=LifecyclePricingPolicy(
            "offline-synthetic-not-approved", Decimal("1"), Decimal("2")
        ),
        max_output_tokens=maximum,
    )
    options = {}
    if provider == "meta":
        from tests.test_three_flow_meta import profile
        from tools.three_flow_live_factory import AdmittedGuard
        from tools.three_flow_profiles import META_MODELS

        model = META_MODELS[0] if model is None else model
        p = profile(model)
        maximum = p["context_max"]
        guard = AdmittedGuard(
            profile=p,
            budget=budget,
            cell="trial",
            policy=LifecyclePricingPolicy(
                "offline-synthetic-not-approved",
                Decimal(p["input_usd_per_million"]),
                Decimal(p["output_usd_per_million"]),
            ),
            max_output_tokens=maximum,
        )
        mode_profile = "off-or-minimum-v1"
        options["context_tokens"] = maximum
    else:
        model = MODELS[provider] if model is None else model
    mock = SDKMockTransport(
        "openai" if provider == "meta" else provider, flow, model=model
    )
    transport = HTTPCampaignTransport(
        provider=provider,
        model=model,
        api_key=api_key,
        inner=mock if handler is None else httpx.MockTransport(handler(mock)),
        guard=guard,
        max_output_tokens=maximum,
        transport_policy=transport_policy,
        **options,
        **({"mode_profile": mode_profile} if mode_profile != "legacy-v1" else {}),
    )
    trial = Trial(
        "test",
        "trial",
        flow,
        scenario,
        model,
        maximum,
        provider_identity(transport),
    )
    return budget, mock, transport, trial


@pytest.mark.parametrize("provider", MODELS)
def test_provider_sdk_roundtrip(tmp_path, provider):
    budget, mock, transport, trial = fixture(tmp_path, provider)
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        count = len(mock.calls)
        assert count > 0
        assert record["evaluation"]["reliable"]
        assert len(record["provider_turns"]) == count
        assert record["evidence_origin"] == "SDK_MOCK_NOT_LLM"
        assert (
            replay_trial(trial, record, expected_record_digest=record["record_digest"])[
                "provider_calls"
            ]
            == 0
        )
        assert len(mock.calls) == count
        if provider == "anthropic":
            assert all(
                c["thinking"] == {"type": "adaptive"}
                and c["tool_choice"] == {"type": "auto"}
                for c in mock.calls
            )
            assert all(t["output_tokens"] == 2 for t in record["provider_turns"])
        if provider == "mistral":
            assert all(0 < c["max_tokens"] < 256000 for c in mock.calls)
        assert budget.totals()["unknown"] == 0
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("flow", ["commerce", "compliance", "maintenance"])
def test_null_root_metadata_sdk_roundtrip_replays_without_provider(
    tmp_path, provider, flow
):
    def wrapper(mock):
        def handle(request):
            body = json.loads(mock.handle(request).content)
            key = "access_programs" if provider == "openai" else "diagnostics"
            assert key not in body  # The baseline fixture remains the old contract.
            body[key] = None
            return httpx.Response(200, json=body)

        return handle

    from tools.three_flow_campaign import assignments

    scenario = next(s["scenario"] for s in assignments("test") if s["flow"] == flow)
    budget, mock, transport, trial = fixture(
        tmp_path, provider, handler=wrapper, flow=flow, scenario=scenario
    )
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        calls = len(mock.calls)
        assert calls > 0
        assert record["evaluation"]["reliable"]
        assert all(t["input_tokens"] == 1 for t in record["provider_turns"])
        assert all(
            t["output_tokens"] == (1 if provider == "openai" else 2)
            for t in record["provider_turns"]
        )
        assert budget.totals()["unknown"] == 0
        assert (
            replay_trial(trial, record, expected_record_digest=record["record_digest"])[
                "provider_calls"
            ]
            == 0
        )
        assert len(mock.calls) == calls
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.parametrize(
    "fault", ["usage_bool", "usage_missing", "wrong_model", "wire", "network", "status"]
)
def test_provider_faults_hold_liability(tmp_path, provider, fault):
    calls = []

    def wrapper(mock):
        def handle(request):
            calls.append(1)
            if fault == "network":
                raise httpx.ReadError("offline", request=request)
            if fault == "status":
                return httpx.Response(
                    503, json={"error": {"message": "offline", "type": "api_error"}}
                )
            if fault == "wire":
                return httpx.Response(200, text="{bad")
            response = mock.handle(request)
            body = json.loads(response.content)
            if fault == "usage_bool":
                body["usage"][
                    "prompt_tokens" if provider == "mistral" else "input_tokens"
                ] = True
            elif fault == "usage_missing":
                body.pop("usage")
            elif fault == "wrong_model":
                body["model"] = "wrong"
            return httpx.Response(200, json=body)

        return handle

    budget, _, transport, trial = fixture(tmp_path, provider, handler=wrapper)
    try:
        with pytest.raises(ProviderFailure):
            run_trial(trial, transport, output=tmp_path / "trial")
        assert calls == [1]
        assert budget.totals()["unknown"] > 0
        assert budget.totals()["pending"] == 0
        assert not (tmp_path / "trial" / "record.json").exists()
        rows = [
            json.loads(line)
            for line in (tmp_path / "trial" / "partial.ndjson").read_text().splitlines()
        ]
        assert rows[-1]["classification"] == "excluded"
        assert any(r["kind"] == "wire_request" for r in rows)
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize(
    "field",
    [
        "settings",
        "model",
        "flow",
        "scenario",
        "source",
        "grade",
        "decisions",
        "origin",
        "request",
    ],
)
def test_coherently_resealed_changes_rejected(tmp_path, field):
    budget, _, transport, trial = fixture(tmp_path, "openai")
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
    finally:
        transport.close()
        budget.close()
    altered = copy.deepcopy(record)
    if field == "settings":
        altered["provider"]["settings"]["max_output_tokens"] = 8
    elif field in ("model", "flow", "scenario"):
        altered["trial"][field] = "other"
    elif field == "source":
        altered["binding"]["campaign_runtime_digest"] = "0" * 64
    elif field == "grade":
        altered["evaluation"]["reliable"] = False
    elif field == "decisions":
        altered["decisions"].pop()
    elif field == "origin":
        altered["evidence_origin"] = "PROVIDER_CANDIDATE"
    else:
        altered["request_digests"][0] = "0" * 64
    altered["record_digest"] = content_digest(
        {k: v for k, v in altered.items() if k != "record_digest"}
    )
    with pytest.raises(ValueError):
        replay_trial(trial, altered, expected_record_digest=record["record_digest"])
    # Even if an attacker supplies their resealed outer digest, independent
    # assignment/settings/source binding and actual replay still reject these.
    with pytest.raises((ValueError, PlaybackError)):
        replay_trial(trial, altered, expected_record_digest=altered["record_digest"])
    assert (
        replay_trial(trial, record, expected_record_digest=record["record_digest"])[
            "provider_calls"
        ]
        == 0
    )


def test_settings_refused_before_dispatch(tmp_path):
    budget, mock, transport, trial = fixture(tmp_path, "openai")
    try:
        wrong = replace(trial, provider_binding={})
        with pytest.raises(ValueError, match="settings"):
            run_trial(wrong, transport, output=tmp_path / "trial")
        assert not mock.calls
        assert budget.totals()["pending"] == 0
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("provider", MODELS)
def test_model_malformed_arguments_are_scored_not_provider_fault(tmp_path, provider):
    def wrapper(mock):
        def handle(request):
            response = mock.handle(request)
            body = json.loads(response.content)
            # Syntactically valid provider envelope, invalid agent tool name.
            if provider == "openai":
                body["output"][0]["name"] = "unknown_agent_tool"
            elif provider == "anthropic":
                body["content"][-1]["name"] = "unknown_agent_tool"
            else:
                body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = (
                    "unknown_agent_tool"
                )
            return httpx.Response(200, json=body)

        return handle

    budget, mock, transport, trial = fixture(tmp_path, provider, handler=wrapper)
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        assert len(mock.calls) > 0
        assert not record["evaluation"]["reliable"]
        assert budget.totals()["unknown"] == 0
        assert (
            replay_trial(trial, record, expected_record_digest=record["record_digest"])[
                "provider_calls"
            ]
            == 0
        )
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("provider", MODELS)
def test_sdk_cannot_read_provider_environment(tmp_path, provider, monkeypatch):
    import os

    original = type(os.environ).__getitem__

    def guarded(self, key):
        if key.startswith(("OPENAI_", "ANTHROPIC_", "MISTRAL_")):
            raise AssertionError("provider environment read forbidden")
        return original(self, key)

    monkeypatch.setattr(type(os.environ), "__getitem__", guarded)
    budget, mock, transport, trial = fixture(tmp_path, provider)
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        assert mock.calls
        assert record["evaluation"]["reliable"]
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("provider", [*MODELS, "meta"])
@pytest.mark.parametrize(
    "body_kind", ["literal", "escaped", "duplicate", "json_key", "malformed", "status"]
)
@pytest.mark.parametrize("key_kind", ["plain", "quote", "backslash"])
def test_credential_response_blocked_before_publication(
    tmp_path, provider, body_kind, key_kind, capsys
):
    # Assemble a synthetic capability at runtime; never a provider credential.
    key = "".join(
        (
            "synthetic",
            "-reflection-",
            {"plain": "", "quote": chr(34), "backslash": chr(92)}[key_kind],
            "capability",
        )
    )
    calls = []
    observed = []

    def wrapper(mock):
        def handle(request):
            calls.append(request)
            previous = transport.wire.observer

            def observe(row):
                observed.append(row)
                previous(row)

            transport.wire.observer = observe
            if body_kind == "escaped":
                escaped = "".join(f"\\u{ord(c):04x}" for c in key)
                data = ('{"nested":[{"value":"' + escaped + '"}]}').encode()
            elif body_kind == "duplicate":
                data = ('{"value":' + json.dumps(key) + ',"value":"safe"}').encode()
            elif body_kind == "json_key":
                data = json.dumps({"nested": [{key: "safe"}]}).encode()
            elif body_kind == "malformed":
                data = ('{"nested":' + json.dumps(key)).encode()
            else:
                data = json.dumps({"error": {"message": key}}).encode()
            if key_kind != "plain" and body_kind != "malformed":
                assert key.encode() not in data
            return httpx.Response(400 if body_kind == "status" else 200, content=data)

        return handle

    budget, _, transport, trial = fixture(
        tmp_path, provider, handler=wrapper, api_key=key
    )
    try:
        with pytest.raises(ProviderFailure) as caught:
            run_trial(trial, transport, output=tmp_path / "trial")
        # Inspect capture first so RED identifies the actual publication defect.
        assert all("response_utf8" not in row for row in transport.wire.captures)
        assert all("response_sha256" not in row for row in transport.wire.captures)
        responses = [row for row in observed if row["kind"] == "wire_response"]
        if body_kind == "status":
            assert len(responses) == 1
            assert responses[0]["response_projection"]["http_status"] == 400
        else:
            assert responses == []
            # OpenAI/Anthropic wrap transport exceptions as connection errors;
            # retain SDK classification rather than changing the error contract.
            assert caught.value.fault == (
                "provider_response_invalid"
                if provider == "mistral"
                else "provider_network_error"
            )
        assert len(calls) == len(transport.wire.captures) == 1
        assert sum(row.get("raw_rpcs", 0) for row in budget.records.values()) == 1
        assert budget.totals()["unknown"] > 0
        assert budget.totals()["pending"] == 0
        assert not (tmp_path / "trial" / "record.json").exists()
        assert key not in repr(transport)
        assert not _contains_decoded_secret(
            [transport.settings, observed, transport.last_turn, str(caught.value)], key
        )
        assert key not in str(calls[0].url)
        assert key.encode() not in calls[0].content
        for path in tmp_path.rglob("*"):
            if path.is_file():
                assert not _contains_decoded_secret(path.read_text(), key)
        captured = capsys.readouterr()
        assert not _contains_decoded_secret(captured.out + captured.err, key)
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("provider", [*MODELS, "meta"])
@pytest.mark.parametrize("key", ["", None, 123, b"synthetic"])
def test_constructor_rejects_invalid_key_before_wire(monkeypatch, provider, key):
    from tools import three_flow_http
    from tools.three_flow_profiles import META_MODELS

    def forbidden(*args, **kwargs):
        raise AssertionError("wire constructed before key validation")

    monkeypatch.setattr(three_flow_http, "BudgetWire", forbidden)
    from operatebench.providers.meta_responses import MetaConfigurationError

    error = MetaConfigurationError if provider == "meta" else ValueError
    with pytest.raises(error):
        HTTPCampaignTransport(
            provider=provider,
            model=META_MODELS[0] if provider == "meta" else None,
            api_key=key,
            inner=httpx.MockTransport(lambda request: pytest.fail("unexpected dispatch")),
            guard=None,
            max_output_tokens=128000,
        )


def test_meta_model_refusal_precedes_invalid_key():
    from operatebench.providers.meta_responses import MetaConfigurationError

    with pytest.raises(MetaConfigurationError, match="exact supported"):
        HTTPCampaignTransport(
            provider="meta",
            model="invalid-model",
            api_key="",
            inner=httpx.MockTransport(lambda request: pytest.fail("unexpected dispatch")),
            guard=None,
            max_output_tokens=128000,
        )


def _contains_decoded_secret(value, secret):
    if isinstance(value, str):
        if secret in value:
            return True
        try:
            decoded = json.loads(value, object_pairs_hook=list)
        except ValueError:
            return (
                any(_contains_decoded_secret(line, secret) for line in value.splitlines())
                if len(value.splitlines()) > 1
                else False
            )
        return decoded != value and _contains_decoded_secret(decoded, secret)
    if isinstance(value, dict):
        value = list(value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_decoded_secret(item, secret) for item in value)
    return False


@pytest.mark.parametrize("key_kind", ["quote", "backslash"])
@pytest.mark.parametrize(
    "shape",
    ["value", "key", "duplicate", "malformed", "depth", "nodes", "safe", "status"],
)
def test_budget_wire_escaped_strings(key_kind, shape):
    from tools.three_flow_http import ENDPOINTS

    key = "".join(
        ("synthetic-", chr(34) if key_kind == "quote" else chr(92), "-capability")
    )
    encoded = json.dumps(key)
    data = {
        "value": '{"nested":[{"value":' + encoded + "}]}",
        "key": '{"nested":[{' + encoded + ':"safe"}]}',
        "duplicate": '{"value":' + encoded + ',"value":"safe"}',
        "malformed": '{"nested":' + encoded,
        "depth": "[" * 1100 + "0" + "]" * 1100,
        "nodes": "[" + ",".join(["null"] * 100001) + "]",
        "safe": ' {"nested": ["ordinary", {"safe": true}]} \n',
        "status": '{"error":' + encoded + "}",
    }[shape].encode()
    assert key.encode() not in data
    calls, observed = [], []

    def handle(request):
        calls.append(1)
        return httpx.Response(400 if shape == "status" else 200, content=data)

    guard = SimpleNamespace(
        _allows_no_money_cap=False,
        cell="trial",
        budget=SimpleNamespace(dispatch=lambda *args: None),
    )
    wire = BudgetWire(httpx.MockTransport(handle), guard, ENDPOINTS["openai"])
    wire.expected_payload = {}
    wire.sensitive_response = key.encode()
    wire.observer = observed.append
    request = httpx.Request("POST", ENDPOINTS["openai"], json={})
    try:
        if shape in ("safe", "status"):
            response = wire.handle_request(request)
            assert response.content == data
            if shape == "safe":
                assert wire.captures[0]["response_utf8"].encode() == data
                assert (
                    wire.captures[0]["response_sha256"]
                    == hashlib.sha256(data).hexdigest()
                )
            else:
                assert wire.captures[0]["response_projection"]["http_status"] == 400
        else:
            from operatebench.providers.faults import AdapterProviderError

            with pytest.raises(AdapterProviderError):
                wire.handle_request(request)
            assert all(
                "response_utf8" not in row and "response_sha256" not in row
                for row in wire.captures
            )
            assert not any(row["kind"] == "wire_response" for row in observed)
        assert calls == [1]
        assert not _contains_decoded_secret([wire.captures, observed], key)
    finally:
        wire.close()
