"""Focused contract for the repeat-ready fixed OpenAI Luna operator.

Every provider episode uses the real OpenAI SDK over ``httpx.MockTransport``.
No test contacts OpenAI or reads a real credential.
"""

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
from operatebench.execution_ledger import read_execution_ledger
from operatebench.providers.cost import request_input_token_bound
from tools import run_lifecycle_v1_mistral_small_2603_canary as proven
from tools import run_lifecycle_v1_openai_luna_canary as canary


def policy() -> LifecyclePricingPolicy:
    return LifecyclePricingPolicy(
        policy_id="operator_pinned_v1",
        input_usd_per_mtok=Decimal("0.20"),
        output_usd_per_mtok=Decimal("1.20"),
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


def write_preregistration(
    tmp_path: Path, output: Path, *, nonce: str, mutate: tuple[str, Any] | None = None
) -> Path:
    record = canary._expected_preregistration(
        output, policy(), cap=Decimal("0.52"), source_identity=source_identity()
    )
    record["nonce"] = nonce
    if mutate is not None:
        record[mutate[0]] = mutate[1]
    parent = tmp_path / f"prereg-{nonce}"
    parent.mkdir(mode=0o700)
    path = parent / "openai-luna-live.json"
    path.write_text(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def authorise(
    tmp_path: Path, output: Path, monkeypatch: pytest.MonkeyPatch, *, nonce: str
) -> Path:
    path = write_preregistration(tmp_path, output, nonce=nonce)
    monkeypatch.setenv(
        canary.LIVE_AUTHORIZATION_VARIABLE, canary.LIVE_AUTHORIZATION_VALUE
    )
    monkeypatch.setenv(canary.CREDENTIAL_VARIABLE, "test-key-not-a-credential")
    monkeypatch.setenv(canary.PREREGISTRATION_PATH_VARIABLE, str(path))
    monkeypatch.setattr(canary, "_runtime_source_identity", source_identity)
    return path


def definition(module: Any, name: str) -> ast.AST:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name
    )


def test_fixed_cell_and_conservative_envelope_are_exact() -> None:
    assert (
        canary.CANARY_PROVIDER,
        canary.CANARY_API,
        canary.CANARY_MODEL,
        canary.CANARY_PROFILE_ID,
        canary.CANARY_AGENT_ID,
    ) == (
        "openai",
        "responses",
        "gpt-5.6-luna",
        "gpt56luna_reasoning_none_sampling_omitted_v2",
        "openai-gpt-5-6-luna",
    )
    assert canary.CANARY_OPERATION_VERSION == "0.6.0"
    assert canary.EXPECTED_PROVIDER_CALLS == 46
    assert canary.MAX_PROVIDER_CALLS == 50
    assert canary.LARGEST_REQUEST_TOKEN_BOUND == 27_241
    assert canary.EXPECTED_TOKEN_UPPER_BOUND == 1_441_502
    assert canary.TOKEN_HARD_CAP == 1_566_850
    assert policy().worst_case_usd(
        calls=50, input_token_bound=27_241, max_output_tokens=4_096
    ) == Decimal("0.518170")
    assert Decimal("0.52") == canary.FIXED_CANARY_COST_CAP_USD


def test_preregistration_binds_source_cell_envelope_and_output_inode(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    prepare(output)
    record = canary._expected_preregistration(
        output, policy(), cap=Decimal("0.52"), source_identity=source_identity()
    )
    info = output.stat()
    assert record["merge_commit"] == "1" * 40
    assert record["git_tree"] == "2" * 40
    assert record["operator_sha256"] == source_identity()["operator_sha256"]
    assert record["provider"] == "openai"
    assert record["api"] == "responses"
    assert record["model"] == "gpt-5.6-luna"
    assert record["request_profile"] == "gpt56luna_reasoning_none_sampling_omitted_v2"
    assert record["expected_provider_calls"] == 46
    assert record["max_provider_calls"] == 50
    assert record["input_token_bound"] == 27_241
    assert record["expected_token_upper_bound"] == 1_441_502
    assert record["token_hard_cap"] == 1_566_850
    assert record["worst_case_usd"] == "0.51817"
    assert record["cost_cap_usd"] == "0.52"
    assert record["output_directory_path"] == str(output.absolute())
    assert (record["output_directory_dev"], record["output_directory_ino"]) == (
        info.st_dev,
        info.st_ino,
    )


def test_proven_descriptor_and_durability_controls_remain_ast_equal() -> None:
    shared = (
        "_runtime_source_identity",
        "_open_preregistration",
        "_decode_preregistration",
        "_expected_preregistration",
        "_open_output_directory",
        "_check_target_free",
        "_take_reservation",
        "reserve_output_names",
        "_audit_what_was_written",
        "_scan_what_was_written",
    )
    for name in shared:
        assert ast.dump(definition(canary, name), include_attributes=False) == ast.dump(
            definition(proven, name), include_attributes=False
        ), name


def test_one_preregistration_is_permanently_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    prepare(output)
    path = write_preregistration(tmp_path, output, nonce="a" * 64)
    monkeypatch.setenv(canary.PREREGISTRATION_PATH_VARIABLE, str(path))
    monkeypatch.setattr(canary, "_runtime_source_identity", source_identity)

    identity = canary.consume_openai_luna_preregistration(
        output, policy(), cap=Decimal("0.52")
    )
    info = output.stat()
    assert identity == (info.st_dev, info.st_ino)
    marker = (
        path.parent / f".openailuna-{'a' * 64}{canary.PREREGISTRATION_CONSUMED_SUFFIX}"
    )
    assert marker.read_bytes() == canary.PREREGISTRATION_CONSUMED_BYTES
    assert marker.stat().st_mode & 0o777 == 0o600
    with pytest.raises(canary.CanaryRefusal, match=canary.CODE_PREREGISTRATION_REFUSED):
        canary.consume_openai_luna_preregistration(output, policy(), cap=Decimal("0.52"))


def test_offline_real_sdk_runs_exact_episode_and_freezes_all_native_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    prepare(output)
    observed: list[httpx.Request] = []

    class Observed(canary.ScriptedProvider):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            observed.append(request)
            return super().__call__(request)

    monkeypatch.setattr(canary, "ScriptedProvider", Observed)
    summary = canary.run_offline_preflight(
        directory=output, policy=policy(), cap=Decimal("0.52"), environ={}
    )

    assert summary["provider_calls"] == summary["attempts"] == 46
    assert summary["completion"] == {
        "artifact_8_complete": True,
        "terminal_status": "completed_successfully",
    }
    assert summary["replay"]["provider_calls"] == 0
    assert len(observed) == 46
    bodies = [json.loads(request.content) for request in observed]
    for request, body in zip(observed, bodies, strict=True):
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert body["model"] == "gpt-5.6-luna"
        assert body["reasoning"] == {"effort": "none"}
        assert body["max_output_tokens"] == 4096
        assert body["tool_choice"] == "required"
        assert body["parallel_tool_calls"] is False
        assert body["store"] is False
        assert "temperature" not in body
        assert "top_p" not in body
    bounds = [request_input_token_bound(body, "OpenAI request") for body in bodies]
    pinned_bounds = [
        16814,
        21914,
        16821,
        22156,
        16735,
        22070,
        16832,
        22423,
        16896,
        23121,
        16899,
        22283,
        16807,
        22796,
        16756,
        22652,
        16886,
        24266,
        16893,
        24509,
        16873,
        24914,
        16873,
        24914,
        16799,
        25041,
        16799,
        25067,
        16853,
        25589,
        16853,
        25770,
        16735,
        26000,
        16742,
        26243,
        16736,
        26237,
        16874,
        27035,
        16874,
        27191,
        16742,
        27059,
        16742,
        27241,
    ]
    assert all(wire <= bound for wire, bound in zip(bounds, pinned_bounds, strict=True))
    assert bounds == [
        bound - (30 if i % 2 else 0) for i, bound in enumerate(pinned_bounds)
    ]
    assert max(bounds) == 27_211
    assert max(bounds) <= canary.LARGEST_REQUEST_TOKEN_BOUND == 27_241


def test_dispatch_cap_is_fifty_not_fifty_one() -> None:
    telemetry = canary.ProviderDispatchTelemetry(50)
    for _ in range(50):
        telemetry.observe_dispatch()
    with pytest.raises(canary.CanaryFailure, match="call_bound_exceeded"):
        telemetry.observe_dispatch()
    assert telemetry.calls_dispatched == 50


def test_environment_only_authority_refuses_before_credential_or_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    prepare(output)
    reached: list[str] = []
    monkeypatch.setenv(
        canary.LIVE_AUTHORIZATION_VARIABLE, canary.LIVE_AUTHORIZATION_VALUE
    )
    monkeypatch.setenv(canary.CREDENTIAL_VARIABLE, "test-key-not-a-credential")
    monkeypatch.setattr(
        canary, "open_live_transport", lambda: reached.append("transport")
    )
    monkeypatch.setattr(
        canary, "open_live_client", lambda *_args, **_kwargs: reached.append("client")
    )

    with pytest.raises(canary.CanaryRefusal, match=canary.CODE_PREREGISTRATION_REFUSED):
        canary.run_live_cell(directory=output, policy=policy(), cap=Decimal("0.52"))
    assert reached == []
    assert canary.CREDENTIAL_VARIABLE not in canary.os.environ


def test_first_mocktransport_fault_reports_one_dispatch_and_no_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "out"
    prepare(output)
    authorise(tmp_path, output, monkeypatch, nonce="b" * 64)
    sdk_calls = 0

    def fail(request: httpx.Request) -> httpx.Response:
        nonlocal sdk_calls
        sdk_calls += 1
        raise httpx.ConnectError("synthetic transport failure", request=request)

    monkeypatch.setattr(canary, "open_live_transport", lambda: httpx.MockTransport(fail))
    status = canary.main(
        [
            "--live",
            "--output-dir",
            str(output),
            "--cost-cap-usd",
            "0.52",
            "--input-usd-per-mtok",
            "0.20",
            "--output-usd-per-mtok",
            "1.20",
        ]
    )
    summary = json.loads(capsys.readouterr().out)
    assert status != 0
    assert sdk_calls == summary["provider_calls_dispatched"] == 1
    assert summary["network_access"] is False
    assert "artifact" not in summary
    assert "evaluator" not in summary
    ledger = read_execution_ledger(output / canary.LEDGER_NAME)
    assert ledger.totals.provider_calls == ledger.totals.attempts == 1
    assert ledger.calls[0].attempts[0].response_received is False
    assert not (output / canary.ARTIFACT_NAME).exists()


@pytest.mark.parametrize("flag", ["--provider", "--model", "--scenario", "--episodes"])
def test_cli_has_no_cell_selector(flag: str) -> None:
    with pytest.raises(SystemExit):
        canary.build_parser().parse_args(
            [
                "--output-dir",
                "/tmp/no-run",
                "--cost-cap-usd",
                "0.52",
                "--input-usd-per-mtok",
                "0.20",
                "--output-usd-per-mtok",
                "1.20",
                flag,
                "changed",
            ]
        )
