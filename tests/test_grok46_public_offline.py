# SPDX-License-Identifier: Apache-2.0
"""Public Grok 4.6 entrypoints cannot spend or consume operator authority."""

import json
import os
import subprocess
import sys
from decimal import Decimal

import pytest

from tools import run_lifecycle_v1_xai_grok_4_6_responses_canary as canary


@pytest.mark.parametrize(
    "entry", ["run_live_cell", "_run_live_cell", "consume", "client", "transport"]
)
def test_public_entrypoints_refuse_before_any_side_effect(tmp_path, monkeypatch, entry):
    def forbidden(*args, **kwargs):
        pytest.fail("public entrypoint reached a side effect")

    monkeypatch.setenv(
        canary.LIVE_AUTHORIZATION_VARIABLE, canary.LIVE_AUTHORIZATION_VALUE
    )
    monkeypatch.setenv(
        canary.PREREGISTRATION_PATH_VARIABLE, str(tmp_path / "authority.json")
    )
    monkeypatch.setattr(canary, "check_pinned_controls", forbidden)
    monkeypatch.setattr(canary, "_open_preregistration", forbidden)
    monkeypatch.setattr(canary.openai, "OpenAI", forbidden)
    monkeypatch.setattr(canary.httpx, "HTTPTransport", forbidden)
    with pytest.raises(canary.CanaryRefusal, match=canary.CODE_PREREGISTRATION_REFUSED):
        if entry == "consume":
            canary.consume_xai_grok46_responses_preregistration(
                tmp_path, None, cap=Decimal("5.20")
            )
        elif entry == "client":
            canary.open_live_client("unused", transport=None)
        elif entry == "transport":
            canary.open_live_transport()
        else:
            getattr(canary, entry)(directory=tmp_path, policy=None, cap=Decimal("5.20"))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("consumer", [False, True])
def test_real_cli_refuses_without_consuming_record(tmp_path, module, consumer):
    name = (
        "consume_grok46_external_authorization"
        if consumer
        else "run_lifecycle_v1_xai_grok_4_6_responses_canary"
    )
    record = tmp_path / "authority.json"
    record.write_text('{"illustrative":true}\n')
    record.chmod(0o600)
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    command = [sys.executable]
    command += ["-m", "tools." + name] if module else ["tools/" + name + ".py"]
    command += ["--output-dir", str(output)]
    if not consumer:
        command += [
            "--live",
            "--cost-cap-usd",
            "5.20",
            "--input-usd-per-mtok",
            "2",
            "--output-usd-per-mtok",
            "6",
        ]
    env = {
        **os.environ,
        canary.LIVE_AUTHORIZATION_VARIABLE: canary.LIVE_AUTHORIZATION_VALUE,
        canary.PREREGISTRATION_PATH_VARIABLE: str(record),
    }
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["ok"] is False
    assert report["credential_read"] is False
    assert report["network_access"] is False
    assert list(output.iterdir()) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["authority.json", "output"]
    assert record.read_text() == '{"illustrative":true}\n'


def test_illustrative_envelope_identity_is_independent():
    assert Decimal("7.80") == canary.FIXED_AGGREGATE_CAP_USD
    assert canary.FIXED_CANARY_POLICY_ID == "illustrative_offline_grok46_v1"
    assert canary.PREREGISTRATION_SCHEMA == "operatebench.illustrative-offline-grok46.v1"
    assert canary.PREREGISTRATION_TYPE == "illustrative-offline-only-not-live-authority"
    assert canary.LIVE_AUTHORIZATION_VALUE.startswith("illustrative-offline-only:")


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "operatebench.grok-4.6-live-preregistration.v2"),
        ("type", "grok-4.6-one-shot-live-authorization"),
        ("envelope_profile", "another-envelope"),
        ("pricing_policy_id", "operator_pinned_v1"),
        ("fleet_total_ceiling_usd", "0"),
        ("fleet_xai_ceiling_usd", "0"),
        ("fleet_reservation_usd", "0"),
    ],
)
def test_mock_consumption_rejects_other_authority_before_marker(
    tmp_path, monkeypatch, field, value
):
    import httpx

    from tests.test_lifecycle_v1_xai_grok46_responses_canary_gate import (
        policy,
        prepare,
        source_identity,
        write_preregistration,
    )

    output = tmp_path / "out"
    prepare(output)
    record_path = write_preregistration(tmp_path, output, nonce="a" * 64)
    record = json.loads(record_path.read_text())
    record[field] = value
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    monkeypatch.setenv(canary.PREREGISTRATION_PATH_VARIABLE, str(record_path))
    monkeypatch.setattr(canary, "_runtime_source_identity", source_identity)
    with pytest.raises(canary.CanaryRefusal):
        canary.consume_xai_grok46_responses_preregistration(
            output,
            policy(),
            cap=Decimal("5.20"),
            before_credential=True,
            mock_transport=httpx.MockTransport(lambda request: httpx.Response(200)),
        )
    assert list(record_path.parent.iterdir()) == [record_path]
    assert list(output.iterdir()) == []


def test_non_mock_injection_refused_before_output_or_client(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("non-mock injection reached a client")

    monkeypatch.setattr(canary.openai, "OpenAI", forbidden)
    with pytest.raises(canary.CanaryRefusal):
        canary.run_offline_preflight(
            directory=tmp_path,
            policy=None,
            cap=Decimal("5.20"),
            environ={},
            mock_transport=object(),
        )
    with pytest.raises(canary.CanaryRefusal):
        canary.consume_xai_grok46_responses_preregistration(
            tmp_path, None, cap=Decimal("5.20"), mock_transport=object()
        )
    assert list(tmp_path.iterdir()) == []
