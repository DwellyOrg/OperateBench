"""Frozen mock SDK bounds and unshimmed current-build refusal contracts."""

from __future__ import annotations

import importlib
import json
from decimal import Context, Decimal, localcontext
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.historical_canary_controls import historical_canary_gate_build  # noqa: F401

# Pinned canonical UTF-8 SDK request ceilings. Keep these numeric controls fixed.
# Engine 0.12 guidance shortens only the 23 business requests by 30 bytes.
# Retrieval requests, provider mappings and all authorization budgets are unchanged.
VECTORS: dict[str, tuple[list[int], int, str, str]] = {
    "run_lifecycle_v1_anthropic_haiku_canary": (
        [
            15454,
            20554,
            15461,
            20796,
            15375,
            20710,
            15472,
            21063,
            15536,
            21761,
            15539,
            20923,
            15447,
            21436,
            15396,
            21292,
            15526,
            22906,
            15533,
            23149,
            15513,
            23554,
            15513,
            23554,
            15439,
            23681,
            15439,
            23707,
            15493,
            24229,
            15493,
            24410,
            15375,
            24640,
            15382,
            24883,
            15376,
            24877,
            15514,
            25675,
            15514,
            25831,
            15382,
            25699,
            15382,
            25881,
        ],
        25649,
        "2.31",
        "1.832845",
    ),
    "run_lifecycle_v1_anthropic_sonnet_canary": (
        [
            15457,
            20557,
            15464,
            20799,
            15378,
            20713,
            15475,
            21066,
            15539,
            21764,
            15542,
            20926,
            15450,
            21439,
            15399,
            21295,
            15529,
            22909,
            15536,
            23152,
            15516,
            23557,
            15516,
            23557,
            15442,
            23684,
            15442,
            23710,
            15496,
            24232,
            15496,
            24413,
            15378,
            24643,
            15385,
            24886,
            15379,
            24880,
            15517,
            25678,
            15517,
            25834,
            15385,
            25702,
            15385,
            25884,
        ],
        26296,
        "4.68",
        "3.665966",
    ),
    "run_lifecycle_v1_canary": (
        [
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
        ],
        23409,
        "0.50",
        "0.4167642",
    ),
    "run_lifecycle_v1_mistral_small_2603_canary": (
        [
            16904,
            22004,
            16911,
            22246,
            16825,
            22160,
            16922,
            22513,
            16986,
            23211,
            16989,
            22373,
            16897,
            22886,
            16846,
            22742,
            16976,
            24356,
            16983,
            24599,
            16963,
            25004,
            16963,
            25004,
            16889,
            25131,
            16889,
            25157,
            16943,
            25679,
            16943,
            25860,
            16825,
            26090,
            16832,
            26333,
            16826,
            26327,
            16964,
            27125,
            16964,
            27281,
            16832,
            27149,
            16832,
            27331,
        ],
        27099,
        "0.33",
        "0.25666935",
    ),
    "run_lifecycle_v1_openai_luna_canary": (
        [
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
        ],
        27009,
        "0.52",
        "0.4167642",
    ),
    "run_lifecycle_v1_xai_grok_4_5_responses_canary": (
        [
            16851,
            21951,
            16858,
            22193,
            16772,
            22107,
            16869,
            22460,
            16933,
            23158,
            16936,
            22320,
            16844,
            22833,
            16793,
            22689,
            16923,
            24303,
            16930,
            24546,
            16910,
            24951,
            16910,
            24951,
            16836,
            25078,
            16836,
            25104,
            16890,
            25626,
            16890,
            25807,
            16772,
            26037,
            16779,
            26280,
            16773,
            26274,
            16911,
            27072,
            16911,
            27228,
            16779,
            27096,
            16779,
            27278,
        ],
        27001,
        "3.93",
        "3.04055",
    ),
    "run_lifecycle_v1_xai_grok_4_6_responses_canary": (
        [
            16852,
            21952,
            16859,
            22194,
            16773,
            22108,
            16870,
            22461,
            16934,
            23159,
            16937,
            22321,
            16845,
            22834,
            16794,
            22690,
            16924,
            24304,
            16931,
            24547,
            16911,
            24952,
            16911,
            24952,
            16837,
            25079,
            16837,
            25105,
            16891,
            25627,
            16891,
            25808,
            16773,
            26038,
            16780,
            26281,
            16774,
            26275,
            16912,
            27073,
            16912,
            27229,
            16780,
            27097,
            16780,
            27279,
        ],
        27002,
        "5.20",
        "4.171138",
    ),
}


def policy(mod: Any) -> Any:
    return mod.LifecyclePricingPolicy(
        policy_id=mod.FIXED_CANARY_POLICY_ID,
        input_usd_per_mtok=mod.FIXED_CANARY_INPUT_USD_PER_MTOK,
        output_usd_per_mtok=mod.FIXED_CANARY_OUTPUT_USD_PER_MTOK,
        rate_source=mod.FIXED_CANARY_RATE_SOURCE,
    )


@pytest.mark.usefixtures("historical_canary_gate_build")
@pytest.mark.parametrize("name", VECTORS)
def test_exact_sdk_vector_and_independent_decimal_envelope(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = importlib.import_module("tools." + name)
    expected, old_bound, _old_cap, expected_reserve = VECTORS[name]
    directory = tmp_path / "out"
    directory.mkdir(mode=0o700)
    observed = []
    initial_bounds = []

    class Capture(mod.ScriptedProvider):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            canonical = json.dumps(
                body,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            observed.append(len(canonical) + 1024)
            # Mistral SDK adds stream=false after the initial payload reservation.
            if "mistral" in name:
                assert body.pop("stream") is False
            initial_bounds.append(
                len(
                    json.dumps(
                        body,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                + 1024
            )
            return super().__call__(request)

    monkeypatch.setattr(mod, "ScriptedProvider", Capture)
    summary = mod.run_offline_preflight(
        directory=directory,
        policy=policy(mod),
        cap=mod.FIXED_CANARY_COST_CAP_USD,
        environ={},
    )
    from operatebench.artifact import OPERATEBENCH_VERSION, read_artifact

    assert OPERATEBENCH_VERSION == "0.13.0"
    assert read_artifact(directory / mod.ARTIFACT_NAME)["engine_version"] == "0.13.0"
    # Check every actual SDK request, not just the episode sum or maximum.
    assert len(observed) == len(expected) == 46
    assert all(wire <= bound for wire, bound in zip(observed, expected, strict=True)), (
        "canonical SDK request exceeds pinned per-request vector"
    )
    assert observed == [bound - (30 if i % 2 else 0) for i, bound in enumerate(expected)]
    assert summary["provider_calls"] == 46
    assert max(observed) <= mod.LARGEST_REQUEST_TOKEN_BOUND
    assert max(expected) == mod.LARGEST_REQUEST_TOKEN_BOUND != old_bound
    with localcontext(Context(prec=80)):
        inp = mod.FIXED_CANARY_INPUT_USD_PER_MTOK
        out = mod.FIXED_CANARY_OUTPUT_USD_PER_MTOK
        worst = (
            (Decimal(max(expected)) * inp + mod.MAX_OUTPUT_TOKENS * out) * 50 / 1_000_000
        )
        reserve = (
            Decimal(sum(observed)) * inp + 46 * mod.MAX_OUTPUT_TOKENS * out
        ) / 1_000_000
    assert worst == Decimal(summary["budget"]["worst_case_usd"])
    assert worst <= mod.FIXED_CANARY_COST_CAP_USD
    assert reserve == Decimal(expected_reserve) - Decimal(23 * 30) * inp / 1_000_000
    with localcontext(Context(prec=80)):
        initial_reserve = (
            Decimal(sum(initial_bounds)) * inp + 46 * mod.MAX_OUTPUT_TOKENS * out
        ) / 1_000_000
    assert initial_reserve == Decimal(summary["ledger_audit"]["totals"]["reserved_usd"])
    if "mistral" in name:
        assert [
            wire - initial for wire, initial in zip(observed, initial_bounds, strict=True)
        ] == [15] * 46
    else:
        assert initial_reserve == reserve
    assert 50 * (max(expected) + mod.MAX_OUTPUT_TOKENS) == mod.TOKEN_HARD_CAP
    assert summary["bundle_audit"]["ok"]
    assert summary["replay"]["ok"]
    assert summary["replay"]["provider_calls"] == 0


@pytest.mark.usefixtures("historical_canary_gate_build")
@pytest.mark.parametrize("name", VECTORS)
def test_guidance_growth_cannot_admit_an_understated_wire_budget(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from operatebench.domains.lettings.maintenance import operation

    original = operation.action_schema_view

    def oversized_guidance():
        schema = original()
        # Real observation -> canonical request -> actual SDK -> MockTransport.
        # The corrected guidance has 30 bytes of per-business-request headroom.
        # Reintroduce one byte beyond that ceiling without touching any budget.
        schema["send_message"]["field_guidance"]["recipient_actor_id"] += "x" * 31
        return schema

    monkeypatch.setattr(operation, "action_schema_view", oversized_guidance)
    with pytest.raises(
        AssertionError, match="canonical SDK request exceeds pinned per-request vector"
    ):
        test_exact_sdk_vector_and_independent_decimal_envelope(
            name, tmp_path, monkeypatch
        )


@pytest.mark.usefixtures("historical_canary_gate_build")
@pytest.mark.parametrize("name", VECTORS)
@pytest.mark.parametrize("kind", ["file", "symlink", "reservation"])
def test_occupied_partial_name_refuses_before_client_or_current_files(
    name: str, kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = importlib.import_module("tools." + name)
    directory = tmp_path / "out"
    directory.mkdir(mode=0o700)
    partial = "execution_ledger.partial.ndjson"
    assert (mod.LEDGER_NAME, mod.ARTIFACT_NAME, partial) == mod.OUTPUT_NAMES
    held = directory / (
        mod.reservation_name_for(partial) if kind == "reservation" else partial
    )
    if kind == "symlink":
        target = tmp_path / "target"
        target.write_text("older evidence")
        held.symlink_to(target)
    else:
        held.write_text("older evidence")

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("client constructed before complete name reservation")

    monkeypatch.setattr(mod, "open_offline_client", forbidden)
    with pytest.raises(mod.CanaryRefusal):
        mod.run_offline_preflight(
            directory=directory,
            policy=policy(mod),
            cap=mod.FIXED_CANARY_COST_CAP_USD,
            environ={},
        )
    assert held.read_text() == "older evidence"
    expected_entries = {held.name}
    if kind == "reservation" and name != "run_lifecycle_v1_canary":
        expected_entries.update(
            mod.reservation_name_for(n) for n in (mod.LEDGER_NAME, mod.ARTIFACT_NAME)
        )
    assert {p.name for p in directory.iterdir()} == expected_entries
    assert not (directory / mod.LEDGER_NAME).exists()
    assert not (directory / mod.ARTIFACT_NAME).exists()


@pytest.mark.usefixtures("historical_canary_gate_build")
@pytest.mark.parametrize("name", VECTORS)
def test_original_numeric_controls_cannot_be_reused(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = importlib.import_module("tools." + name)
    _, old_bound, old_cap, _ = VECTORS[name]
    directory = tmp_path / "out"
    directory.mkdir(mode=0o700)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("client constructed with historical controls")

    monkeypatch.setattr(mod, "open_offline_client", forbidden)
    if Decimal(old_cap) != mod.FIXED_CANARY_COST_CAP_USD:
        with pytest.raises(
            mod.CanaryRefusal, match=mod.CODE_FIXED_CANARY_PRICING_ENVELOPE_REQUIRED
        ):
            mod.run_offline_preflight(
                directory=directory, policy=policy(mod), cap=Decimal(old_cap), environ={}
            )
    monkeypatch.setattr(mod, "LARGEST_REQUEST_TOKEN_BOUND", old_bound)
    with pytest.raises(mod.CanaryRefusal, match="pinned_token_ceiling_changed"):
        mod.run_offline_preflight(
            directory=directory,
            policy=policy(mod),
            cap=mod.FIXED_CANARY_COST_CAP_USD,
            environ={},
        )
    assert list(directory.iterdir()) == []


@pytest.mark.usefixtures("historical_canary_gate_build")
@pytest.mark.parametrize("name", VECTORS)
def test_fault_retains_well_formed_non_scored_sidecar(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from operatebench.artifact import ArtifactError, read_artifact
    from operatebench.execution_ledger import read_execution_ledger

    mod = importlib.import_module("tools." + name)
    directory = tmp_path / "out"
    directory.mkdir(mode=0o700)

    class Fault(mod.ScriptedProvider):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            if self.calls == 3:
                return httpx.Response(500, json={"error": {"message": "synthetic"}})
            return super().__call__(request)

    monkeypatch.setattr(mod, "ScriptedProvider", Fault)
    code = mod.main(
        [
            "--offline-preflight",
            "--output-dir",
            str(directory),
            "--cost-cap-usd",
            str(mod.FIXED_CANARY_COST_CAP_USD),
            "--input-usd-per-mtok",
            str(mod.FIXED_CANARY_INPUT_USD_PER_MTOK),
            "--output-usd-per-mtok",
            str(mod.FIXED_CANARY_OUTPUT_USD_PER_MTOK),
        ]
    )
    assert code != 0
    assert not (directory / mod.ARTIFACT_NAME).exists()
    ledger = read_execution_ledger(directory / mod.LEDGER_NAME)
    assert ledger.scored is False
    partial = directory / "execution_ledger.partial.ndjson"
    rows = [json.loads(line) for line in partial.read_text().splitlines()]
    assert rows[0]["schema"] == "operatebench.partial-execution.v1"
    assert any(row["kind"] == "decision" for row in rows)
    assert rows[-1]["kind"] == "closure"
    assert rows[-1]["classification"] == "excluded"
    with pytest.raises(ArtifactError):
        read_artifact(partial)


@pytest.mark.parametrize("name", VECTORS)
@pytest.mark.parametrize("mode", ["--offline-preflight", "--live"])
def test_unshimmed_current_build_refuses_before_keys_claims_or_sdk(
    name: str,
    mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from types import SimpleNamespace

    from operatebench.version import OPERATEBENCH_VERSION

    mod = importlib.import_module("tools." + name)
    assert mod.OPERATEBENCH_VERSION == OPERATEBENCH_VERSION == "0.13.0"
    with pytest.raises(mod.CanaryRefusal, match="pinned_exact_cell_envelope_changed"):
        mod.check_pinned_controls()
    directory = tmp_path / "out"
    directory.mkdir(mode=0o700)
    reached = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        reached.append("keys, claims or SDK")
        pytest.fail("current build crossed frozen operator gate")

    class UnreadEnvironment(dict):
        def __getitem__(self, key):
            if key == mod.CREDENTIAL_VARIABLE:
                forbidden()
            return super().__getitem__(key)

        def get(self, key, default=None):
            if key == mod.CREDENTIAL_VARIABLE:
                forbidden()
            return super().get(key, default)

    # An isolated synthetic environment; never read the process's credentials.
    monkeypatch.setattr(
        mod,
        "os",
        SimpleNamespace(environ=UnreadEnvironment() if mode == "--live" else {}),
    )
    for boundary in ("reserve_output_names", "open_live_client", "open_offline_client"):
        monkeypatch.setattr(mod, boundary, forbidden)
    monkeypatch.setattr(mod, "ScriptedProvider", forbidden)
    if "anthropic" in name:
        monkeypatch.setattr(mod.anthropic, "Anthropic", forbidden)
    elif "mistral" in name:
        monkeypatch.setattr(mod, "build_mistral_client", forbidden)
    else:
        monkeypatch.setattr(mod.openai, "OpenAI", forbidden)
    code = mod.main(
        [
            mode,
            "--output-dir",
            str(directory),
            "--cost-cap-usd",
            str(mod.FIXED_CANARY_COST_CAP_USD),
            "--input-usd-per-mtok",
            str(mod.FIXED_CANARY_INPUT_USD_PER_MTOK),
            "--output-usd-per-mtok",
            str(mod.FIXED_CANARY_OUTPUT_USD_PER_MTOK),
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    expected_code = (
        "xai_grok46_responses_preregistration_refused"
        if mode == "--live" and name == "run_lifecycle_v1_xai_grok_4_6_responses_canary"
        else "pinned_exact_cell_envelope_changed"
    )
    assert report["failure"]["code"] == expected_code
    if name not in {"run_lifecycle_v1_canary", "run_lifecycle_v1_anthropic_haiku_canary"}:
        assert report["provider_calls_dispatched"] == 0
        assert report["credential_read"] is False
        assert report["network_access"] is False
    assert reached == []
    assert list(directory.iterdir()) == []
