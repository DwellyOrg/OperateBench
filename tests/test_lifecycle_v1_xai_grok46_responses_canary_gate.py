# SPDX-License-Identifier: Apache-2.0
"""Gate the illustrative offline xAI grok-4.6 Responses operator."""

from __future__ import annotations

import ast
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from operatebench.agents.pricing import RATE_SOURCE_OPERATOR, LifecyclePricingPolicy
from tools import run_lifecycle_v1_openai_luna_canary as openai_canary
from tools import run_lifecycle_v1_xai_grok_4_6_responses_canary as canary


def policy() -> LifecyclePricingPolicy:
    return LifecyclePricingPolicy(
        policy_id="illustrative_offline_grok46_v1",
        input_usd_per_mtok=Decimal("2"),
        output_usd_per_mtok=Decimal("6"),
        rate_source=RATE_SOURCE_OPERATOR,
    )


def prepare(directory: Path) -> None:
    directory.mkdir(mode=0o700)
    directory.chmod(0o700)


def source_identity() -> dict[str, str]:
    return {
        "merge_commit": "1" * 40,
        "git_tree": "2" * 40,
        "operator_sha256": hashlib.sha256(Path(canary.__file__).read_bytes()).hexdigest(),
    }


def write_preregistration(tmp_path: Path, output: Path, *, nonce: str) -> Path:
    record = canary._expected_preregistration(
        output, policy(), cap=Decimal("5.2"), source_identity=source_identity()
    )
    record["nonce"] = nonce
    parent = tmp_path / f"prereg-{nonce}"
    parent.mkdir(mode=0o700)
    path = parent / "xai-grok46-offline-example.json"
    path.write_text(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def definition(module: Any, name: str) -> ast.AST:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name
    )


def test_successor_authorization_binds_mapping_and_control_digests(
    tmp_path: Path,
) -> None:
    from operatebench.agents.xai_responses import lifecycle_xai_settings
    from operatebench.execution_ledger import ledger_digest

    output = tmp_path / "output"
    prepare(output)
    record = canary._expected_preregistration(
        output, policy(), cap=Decimal("5.2"), source_identity=source_identity()
    )
    assert record["schema"] == "operatebench.illustrative-offline-grok46.v1"
    assert record["request_mapping"] == "lifecycle_xai_responses_model_request_v7"
    assert record["controls_digest_sha256"] == ledger_digest(
        canary.controls_for(policy(), cap=Decimal("5.2")).as_dict()
    )
    assert record["provider_settings_digest_sha256"] == ledger_digest(
        lifecycle_xai_settings(model="grok-4.6", deadline_seconds=30.0)
    )
    assert "request-mapping-v2" in canary.LIVE_AUTHORIZATION_VALUE


def test_fixed_offline_xai_cell_and_trial_envelope() -> None:
    assert (
        canary.CANARY_PROVIDER,
        canary.CANARY_API,
        canary.CANARY_MODEL,
        canary.CANARY_SCENARIO_ID,
        canary.CANARY_EPISODES,
    ) == ("xai", "responses", "grok-4.6", "V1", 1)
    assert canary.EXPECTED_PROVIDER_CALLS == 46
    assert canary.MAX_PROVIDER_CALLS == 50
    assert canary.LARGEST_REQUEST_TOKEN_BOUND == 27_279
    assert canary.EXPECTED_TOKEN_UPPER_BOUND == 1_631_666
    assert canary.TOKEN_HARD_CAP == 1_773_550
    assert canary.MAX_OUTPUT_TOKENS == 8192
    per_call = (Decimal(27279) * 2 + Decimal(8192) * 6) / Decimal(1_000_000)
    assert per_call == Decimal("0.10371")
    assert per_call > canary.REJECTED_REQUEST_FEE_USD == Decimal("0.05")
    assert 50 * max(per_call, Decimal("0.05")) == Decimal("5.1855")
    assert canary.MAX_ATTEMPTS_PER_CALL == 1
    assert canary.SDK_MAX_RETRIES == 0
    assert canary.TURN_DEADLINE_SECONDS == 30.0
    assert canary.WALL_CLOCK_DEADLINE_SECONDS == 2700.0
    assert Decimal("2") == canary.FIXED_CANARY_INPUT_USD_PER_MTOK
    assert Decimal("6") == canary.FIXED_CANARY_OUTPUT_USD_PER_MTOK
    assert Decimal("5.2") == canary.FIXED_CANARY_COST_CAP_USD
    assert canary.FIXED_TRIAL_COUNT == 1
    assert Decimal("7.80") == canary.FIXED_AGGREGATE_CAP_USD
    assert canary.MODEL_IDENTIFIER_IMMUTABLE is False
    assert canary.PUBLIC_REPRODUCIBILITY_CLAIM is False
    assert canary.TOOL_OUTPUT_TOKEN_ACCOUNTING_ASSUMPTION is True
    assert (
        canary.LIFECYCLE_XAI_REQUEST_MAPPING_VERSION
        == "lifecycle_xai_responses_model_request_v7"
    )
    assert canary.XAI_RESPONSE_SERVER_EXTENSIONS == "xai_responses_server_extensions_v2"
    assert canary.PINNED_RESPONSE_EXTENSION_DIGEST == (
        "9b5c06b433ddeb9015ed3bb7b215100aecc196bdbbcfff71e99e642c094a85f4"
    )
    assert canary.RESPONSE_EXTENSION_DIGEST == (
        "9b5c06b433ddeb9015ed3bb7b215100aecc196bdbbcfff71e99e642c094a85f4"
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("XAI_RESPONSE_SERVER_EXTENSIONS", "other_extensions_v1"),
        ("RESPONSE_EXTENSION_DIGEST", "0" * 64),
    ],
)
def test_current_operator_rejects_unpinned_response_extension_identity(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setattr(canary, name, value)
    with pytest.raises(canary.CanaryRefusal, match="pinned_exact_cell_envelope_changed"):
        canary.check_pinned_controls()


def test_artifact_schema_closes_xai_to_both_implemented_api_surfaces() -> None:
    import json
    from pathlib import Path

    from operatebench.artifact import BINDABLE_PROVIDER_API_PAIRS

    expected = {
        ("openai", "responses"),
        ("anthropic", "messages"),
        ("xai", "chat_completions"),
        ("xai", "responses"),
        ("mistral", "chat_completions"),
    }
    assert set(BINDABLE_PROVIDER_API_PAIRS) == expected
    for path in (
        Path("docs/schemas/provider-execution-binding-v1.schema.json"),
        Path(
            "src/operatebench/resources/identity/provider-execution-binding-v1.schema.json"
        ),
    ):
        schema = json.loads(path.read_text(encoding="utf-8"))
        xai_rule = next(
            rule
            for rule in schema["allOf"]
            if rule["if"]["properties"]["provider"].get("const") == "xai"
        )
        assert xai_rule["then"]["properties"]["api"] == {
            "enum": ["chat_completions", "responses"]
        }


def test_record_binds_exact_illustrative_envelope_and_output_inode(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    prepare(output)
    record = canary._expected_preregistration(
        output, policy(), cap=Decimal("5.2"), source_identity=source_identity()
    )
    info = output.stat()
    assert (
        record["provider"],
        record["api"],
        record["model"],
        record["input_token_bound"],
        record["expected_token_upper_bound"],
        record["token_hard_cap"],
        record["worst_case_usd"],
        record["cost_cap_usd"],
    ) == (
        "xai",
        "responses",
        "grok-4.6",
        27_279,
        1_631_666,
        1_773_550,
        "5.1855",
        "5.2",
    )
    assert record["envelope_profile"] == "grok46_offline_example_v1"
    assert record["fleet_total_ceiling_usd"] == "13.00"
    assert record["fleet_xai_ceiling_usd"] == "7.80"
    assert record["fleet_reservation_usd"] == "5.2"
    assert (record["output_directory_dev"], record["output_directory_ino"]) == (
        info.st_dev,
        info.st_ino,
    )


def test_one_preregistration_is_permanently_consumed_and_not_reusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    prepare(output)
    nonce = "a" * 64
    path = write_preregistration(tmp_path, output, nonce=nonce)
    monkeypatch.setenv(canary.PREREGISTRATION_PATH_VARIABLE, str(path))
    monkeypatch.setattr(canary, "_runtime_source_identity", source_identity)
    # External consumption precedes any credential-loader stage.
    canary.consume_xai_grok46_responses_preregistration(
        output,
        policy(),
        mock_transport=httpx.MockTransport(lambda request: httpx.Response(200)),
        cap=Decimal("5.2"),
        before_credential=True,
    )
    canary.consume_xai_grok46_responses_preregistration(
        output,
        policy(),
        mock_transport=httpx.MockTransport(lambda request: httpx.Response(200)),
        cap=Decimal("5.2"),
    )
    marker = path.parent / (
        f".xaigrok46responses-{nonce}{canary.PREREGISTRATION_CONSUMED_SUFFIX}"
    )
    assert marker.read_bytes() == canary.PREREGISTRATION_CONSUMED_BYTES
    assert marker.stat().st_mode & 0o777 == 0o600
    with pytest.raises(canary.CanaryRefusal, match=canary.CODE_PREREGISTRATION_REFUSED):
        canary.consume_xai_grok46_responses_preregistration(
            output,
            policy(),
            mock_transport=httpx.MockTransport(lambda request: httpx.Response(200)),
            cap=Decimal("5.2"),
        )


def test_dispatch_cap_is_fifty_not_fifty_one() -> None:
    telemetry = canary.ProviderDispatchTelemetry(50)
    for _ in range(50):
        telemetry.observe_dispatch()
    with pytest.raises(canary.CanaryFailure, match="call_bound_exceeded"):
        telemetry.observe_dispatch()
    assert telemetry.calls_dispatched == 50


def test_environment_authority_alone_refuses_before_credential_or_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    prepare(output)
    reached: list[str] = []
    monkeypatch.setenv(
        canary.LIVE_AUTHORIZATION_VARIABLE, canary.LIVE_AUTHORIZATION_VALUE
    )
    monkeypatch.setenv(canary.CREDENTIAL_VARIABLE, "private-test-key")
    monkeypatch.setattr(
        canary, "open_live_transport", lambda: reached.append("transport")
    )
    monkeypatch.setattr(
        canary, "open_live_client", lambda *_a, **_k: reached.append("client")
    )
    with pytest.raises(canary.CanaryRefusal, match=canary.CODE_PREREGISTRATION_REFUSED):
        canary.run_live_cell(directory=output, policy=policy(), cap=Decimal("5.2"))
    assert reached == []
    assert canary.CREDENTIAL_VARIABLE not in canary.os.environ


def test_openai_trust_controls_remain_compact_ast_equal() -> None:
    shared = (
        "_runtime_source_identity",
        "_open_preregistration",
        "_decode_preregistration",
        "_open_output_directory",
        "_check_target_free",
        "_take_reservation",
        "reserve_output_names",
        "_audit_what_was_written",
        "_scan_what_was_written",
    )
    for name in shared:
        assert ast.dump(definition(canary, name), include_attributes=False) == ast.dump(
            definition(openai_canary, name), include_attributes=False
        ), name


def test_failure_report_is_private_and_never_claims_an_artifact() -> None:
    secret = "sk-SYNTHETIC-NEVER-VALID-TEST-CREDENTIAL"
    report = canary._failure(
        canary.MODE_LIVE,
        canary.CanaryFailure("provider_boundary_fault", secret),
        provider_calls_dispatched=1,
    )
    emitted = json.dumps(report)
    assert report["ok"] is False
    assert secret not in emitted
    assert "artifact" not in report


@pytest.mark.parametrize(
    "outcome", ["success", "missing_usage", "rejected_403", "rate_429"]
)
def test_full_sdk_wire_episode_or_usage_forfeit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    import httpx

    from operatebench.providers.cost import request_input_token_bound

    output = tmp_path / "wire"
    prepare(output)
    original = canary.ScriptedProvider
    bodies: list[dict[str, Any]] = []

    class CheckedProvider(original):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            bodies.append(body)
            assert str(request.url) == "https://api.x.ai/v1/responses"
            assert body["model"] == "grok-4.6"
            assert body["reasoning"] == {"effort": "high"}
            assert body["max_output_tokens"] == 8192
            assert (
                "request_mapping: lifecycle_xai_responses_model_request_v7\n"
                in body["instructions"]
            )
            assert body["store"] is False
            assert body["parallel_tool_calls"] is False
            assert body["tool_choice"] == "required"
            assert "temperature" not in body and "top_p" not in body
            assert "previous_response_id" not in body
            assert len(body["tools"]) == 6
            assert {item["type"] for item in body["tools"]} == {"function"}
            assert all(item["strict"] is False for item in body["tools"])
            assert request_input_token_bound(body, label="offline") <= 27279
            if outcome in ("rejected_403", "rate_429"):
                status = 403 if outcome == "rejected_403" else 429
                return httpx.Response(
                    status,
                    request=request,
                    json={"error": {"message": "offline rejection"}},
                )
            response = super().__call__(request)
            if outcome == "success":
                return response
            data = json.loads(response.content)
            data["usage"] = None
            return httpx.Response(200, request=request, json=data)

    monkeypatch.setattr(canary, "ScriptedProvider", CheckedProvider)
    if outcome != "success":
        with pytest.raises(canary.CanaryFailure):
            canary.run_offline_preflight(
                directory=output,
                policy=policy(),
                cap=Decimal("5.2"),
                environ={},
                mock_transport=httpx.MockTransport(CheckedProvider(max_calls=50)),
            )
        ledger = canary.read_execution_ledger(output / canary.LEDGER_NAME)
        assert len(bodies) == 1
        assert not (output / canary.ARTIFACT_NAME).exists()
        assert Decimal(ledger.totals.measured_cost_usd) == Decimal("0")
        assert Decimal(ledger.totals.forfeited_reservation_usd) > Decimal("0")
        assert ledger.totals.forfeited_reservation_usd == ledger.totals.reserved_usd
        first_bound = request_input_token_bound(bodies[0], label="offline")
        expected_reservation = (Decimal(first_bound) * 2 + Decimal(8192) * 6) / Decimal(
            1_000_000
        )
        assert Decimal(ledger.totals.forfeited_reservation_usd) == expected_reservation
        assert expected_reservation > Decimal("0.05")
    else:
        result = canary.run_offline_preflight(
            directory=output,
            policy=policy(),
            cap=Decimal("5.2"),
            environ={},
            mock_transport=httpx.MockTransport(CheckedProvider(max_calls=50)),
        )
        assert len(bodies) == 46
        ledger = canary.read_execution_ledger(output / canary.LEDGER_NAME)
        assert ledger.header.provider.settings["max_output_tokens"] == 8192
        assert (
            ledger.header.provider.request_mapping
            == "lifecycle_xai_responses_model_request_v7"
        )
        assert (
            ledger.header.provider.settings["request_mapping"]
            == ledger.header.provider.request_mapping
        )
        assert ledger.header.controls.max_output_tokens == 8192
        bounds = [request_input_token_bound(body, label="offline") for body in bodies]
        assert max(bounds) == 27249
        assert max(bounds) <= canary.LARGEST_REQUEST_TOKEN_BOUND == 27279
        assert all(
            (Decimal(bound) * 2 + Decimal(8192) * 6) / Decimal(1_000_000)
            > Decimal("0.05")
            for bound in bounds
        )
        assert result["bundle_audit"]["ok"] is True
        assert result["replay"]["ok"] is True
        assert result["replay"]["provider_calls"] == 0
        assert result["credential_read"] is False
        assert result["network_access"] is False
        assert result["budget"]["worst_case_usd"] == "5.1855"


def test_external_cli_refuses_without_consuming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from tools import consume_grok46_external_authorization as external

    output = tmp_path / "out"
    prepare(output)
    path = write_preregistration(tmp_path, output, nonce="d" * 64)
    monkeypatch.setenv(canary.PREREGISTRATION_PATH_VARIABLE, str(path))
    monkeypatch.setenv(
        canary.LIVE_AUTHORIZATION_VARIABLE, canary.LIVE_AUTHORIZATION_VALUE
    )
    monkeypatch.setattr(canary, "_runtime_source_identity", source_identity)
    assert external.main(["--output-dir", str(output)]) == 1
    assert list(path.parent.iterdir()) == [path]
    assert json.loads(capsys.readouterr().out) == {
        "ok": False,
        "credential_read": False,
        "network_access": False,
    }
    assert external.main(["--output-dir", str(output)]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_live_consumption_needs_source_bound_external_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    prepare(output)
    path = write_preregistration(tmp_path, output, nonce="b" * 64)
    monkeypatch.setenv(canary.PREREGISTRATION_PATH_VARIABLE, str(path))
    monkeypatch.setattr(canary, "_runtime_source_identity", source_identity)
    with pytest.raises(canary.CanaryRefusal):
        canary.consume_xai_grok46_responses_preregistration(
            output,
            policy(),
            mock_transport=httpx.MockTransport(lambda request: httpx.Response(200)),
            cap=Decimal("5.2"),
        )
    assert len(list(path.parent.iterdir())) == 1


def test_external_consumption_refuses_credential_environment_and_is_one_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    prepare(output)
    path = write_preregistration(tmp_path, output, nonce="c" * 64)
    monkeypatch.setenv(canary.PREREGISTRATION_PATH_VARIABLE, str(path))
    monkeypatch.setattr(canary, "_runtime_source_identity", source_identity)
    monkeypatch.setenv(canary.CREDENTIAL_VARIABLE, "not-read")
    with pytest.raises(canary.CanaryRefusal):
        canary.consume_xai_grok46_responses_preregistration(
            output,
            policy(),
            mock_transport=httpx.MockTransport(lambda request: httpx.Response(200)),
            cap=Decimal("5.2"),
            before_credential=True,
        )
    assert len(list(path.parent.iterdir())) == 1
    monkeypatch.delenv(canary.CREDENTIAL_VARIABLE)
    canary.consume_xai_grok46_responses_preregistration(
        output,
        policy(),
        mock_transport=httpx.MockTransport(lambda request: httpx.Response(200)),
        cap=Decimal("5.2"),
        before_credential=True,
    )
    with pytest.raises(canary.CanaryRefusal):
        canary.consume_xai_grok46_responses_preregistration(
            output,
            policy(),
            mock_transport=httpx.MockTransport(lambda request: httpx.Response(200)),
            cap=Decimal("5.2"),
            before_credential=True,
        )


@pytest.mark.parametrize("attack", ["bytes", "mode", "symlink", "record"])
def test_external_receipt_rejects_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack: str
) -> None:
    output = tmp_path / "out"
    prepare(output)
    path = write_preregistration(tmp_path, output, nonce="e" * 64)
    monkeypatch.setenv(canary.PREREGISTRATION_PATH_VARIABLE, str(path))
    monkeypatch.setattr(canary, "_runtime_source_identity", source_identity)
    canary.consume_xai_grok46_responses_preregistration(
        output,
        policy(),
        mock_transport=httpx.MockTransport(lambda request: httpx.Response(200)),
        cap=Decimal("5.2"),
        before_credential=True,
    )
    marker = next(p for p in path.parent.iterdir() if p != path)
    if attack == "bytes":
        marker.write_bytes(b"not a consumption record")
    elif attack == "mode":
        marker.chmod(0o644)
    elif attack == "symlink":
        target = tmp_path / "fake"
        target.write_bytes(canary.PREREGISTRATION_CONSUMED_BYTES)
        target.chmod(0o600)
        marker.unlink()
        marker.symlink_to(target)
    else:
        record = json.loads(path.read_text())
        record["nonce"] = "f" * 64
        path.write_text(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
    with pytest.raises(canary.CanaryRefusal):
        canary.consume_xai_grok46_responses_preregistration(
            output,
            policy(),
            mock_transport=httpx.MockTransport(lambda request: httpx.Response(200)),
            cap=Decimal("5.2"),
        )
