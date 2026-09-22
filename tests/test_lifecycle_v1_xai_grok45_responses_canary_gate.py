# SPDX-License-Identifier: Apache-2.0
"""Gate the fixed private xAI grok-4.5 Responses operator."""

from __future__ import annotations

import ast
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from operatebench.agents.pricing import RATE_SOURCE_OPERATOR, LifecyclePricingPolicy
from tools import run_lifecycle_v1_openai_luna_canary as openai_canary
from tools import run_lifecycle_v1_xai_grok_4_5_responses_canary as canary


def policy() -> LifecyclePricingPolicy:
    return LifecyclePricingPolicy(
        policy_id="operator_pinned_v1",
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
        output, policy(), cap=Decimal("3.96"), source_identity=source_identity()
    )
    record["nonce"] = nonce
    parent = tmp_path / f"prereg-{nonce}"
    parent.mkdir(mode=0o700)
    path = parent / "xai-grok45-responses-live.json"
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


def test_fixed_private_xai_cell_and_trial_envelope() -> None:
    assert (
        canary.CANARY_PROVIDER,
        canary.CANARY_API,
        canary.CANARY_MODEL,
        canary.CANARY_SCENARIO_ID,
        canary.CANARY_EPISODES,
    ) == ("xai", "responses", "grok-4.5", "V1", 1)
    assert canary.EXPECTED_PROVIDER_CALLS == 46
    assert canary.MAX_PROVIDER_CALLS == 50
    assert canary.LARGEST_REQUEST_TOKEN_BOUND == 27_278
    assert canary.EXPECTED_TOKEN_UPPER_BOUND == 1_443_204
    assert canary.TOKEN_HARD_CAP == 1_568_700
    assert canary.MAX_OUTPUT_TOKENS == 4096
    assert canary.TURN_DEADLINE_SECONDS == 30.0
    assert canary.WALL_CLOCK_DEADLINE_SECONDS == 2700.0
    assert Decimal("2") == canary.FIXED_CANARY_INPUT_USD_PER_MTOK
    assert Decimal("6") == canary.FIXED_CANARY_OUTPUT_USD_PER_MTOK
    assert Decimal("3.96") == canary.FIXED_CANARY_COST_CAP_USD
    assert canary.FIXED_TRIAL_COUNT == 10
    assert Decimal("39.566") == canary.FIXED_AGGREGATE_CAP_USD
    assert canary.MODEL_IDENTIFIER_IMMUTABLE is False
    assert canary.PUBLIC_REPRODUCIBILITY_CLAIM is False
    assert canary.TOOL_OUTPUT_TOKEN_ACCOUNTING_ASSUMPTION is True
    assert (
        canary.LIFECYCLE_XAI_REQUEST_MAPPING_VERSION
        == "lifecycle_xai_responses_model_request_v6"
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


def test_preregistration_binds_exact_private_envelope_and_output_inode(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    prepare(output)
    record = canary._expected_preregistration(
        output, policy(), cap=Decimal("3.96"), source_identity=source_identity()
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
        "grok-4.5",
        27_278,
        1_443_204,
        1_568_700,
        "3.9566",
        "3.96",
    )
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
    canary.consume_xai_grok45_responses_preregistration(
        output, policy(), cap=Decimal("3.96")
    )
    marker = path.parent / (
        f".xaigrok45responses-{nonce}{canary.PREREGISTRATION_CONSUMED_SUFFIX}"
    )
    assert marker.read_bytes() == canary.PREREGISTRATION_CONSUMED_BYTES
    assert marker.stat().st_mode & 0o777 == 0o600
    with pytest.raises(canary.CanaryRefusal, match=canary.CODE_PREREGISTRATION_REFUSED):
        canary.consume_xai_grok45_responses_preregistration(
            output, policy(), cap=Decimal("3.96")
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
        canary.run_live_cell(directory=output, policy=policy(), cap=Decimal("3.96"))
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
