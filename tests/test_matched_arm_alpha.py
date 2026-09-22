"""Fail-closed tests for the fixed matched-arm Alpha operator.

No test opens a socket or reads a real credential.  The offline preflight goes
through the real OpenAI SDK, WireCaptureTransport, provider recorders, Full
Engine executions, ledgers, artifacts, bundle audits and provider-free replay.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator, Mapping, MutableMapping
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from boundarybench.jsonsafe import canonical_json_bytes
from operatebench.agents.openai_responses import OpenAIResponsesTransport
from operatebench.domains.lettings.maintenance.semantic_arms_v1 import (
    ADMISSIBLE,
    ARM_BOUNDARY,
    ARM_FULL,
    ARM_ORDERED,
    ARM_STATIC,
    ERROR,
    NOT_ESTABLISHED,
)
from operatebench.execution_ledger import ledger_digest, row_digest
from operatebench.providers.openai_responses import REJECTED_VARIABLES
from tests.historical_runtime import validate_historical_mock_result
from tools import matched_arm_alpha as alpha


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    path = tmp_path / "alpha"
    path.mkdir(mode=0o700)
    return path


@pytest.fixture(autouse=True)
def clean_environment(
    monkeypatch: pytest.MonkeyPatch, archived_provider_identity: dict[str, Any]
) -> None:
    # Frozen Alpha source imports this identity from its original provider.
    # Keep the historical reader coherent without relabelling the current runtime.
    monkeypatch.setattr(
        alpha,
        "LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION",
        "lifecycle_openai_responses_model_request_v7",
    )
    monkeypatch.setattr(
        alpha,
        "_expected_provider_identity",
        lambda: copy.deepcopy(archived_provider_identity),
    )
    for name in (
        alpha.CREDENTIAL_VARIABLE,
        alpha.LIVE_AUTHORIZATION_VARIABLE,
        alpha.EXPECTED_SOURCE_REVISION_VARIABLE,
        *REJECTED_VARIABLES,
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def archived_provider_identity(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    from tests.historical_runtime import run_historical_mock

    return json.loads(
        run_historical_mock("alpha-identity", tmp_path_factory.mktemp("alpha-identity"))
    )


@pytest.fixture(scope="session")
def captured_preflight(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate mock evidence in the exact historical offline subprocess.

    Current validators still audit every original tamper case. No current
    engine executes under an old version label or bypasses its build check.
    """
    from tests.historical_runtime import run_historical_mock

    directory = tmp_path_factory.mktemp("matched-alpha-preflight") / "captured"
    directory.mkdir(mode=0o700)
    run_historical_mock("alpha", directory)
    return directory


@pytest.fixture()
def tamperable_preflight(
    captured_preflight: Path, tmp_path: Path
) -> tuple[dict[str, Any], Path]:
    directory = tmp_path / "captured"
    shutil.copytree(captured_preflight, directory)
    result = json.loads((directory / alpha.RESULT_NAME).read_text())
    return result, directory


def refresh_standalone_manifest(result: dict[str, Any], directory: Path) -> None:
    path = directory / alpha.MANIFEST_NAME
    document = {"schema": alpha.MANIFEST_SCHEMA, "entries": result["evidence_manifest"]}
    path.write_bytes(canonical_json_bytes(document, "test manifest") + b"\n")
    result["evidence_manifest_digest_sha256"] = alpha._sha_file(path)


def test_current_alpha_causal_audit_refuses_old_engine_mock(
    captured_preflight: Path,
) -> None:
    from operatebench.agents.playback import ObservationMismatchError
    from operatebench.artifact import ArtifactError

    result = json.loads((captured_preflight / alpha.RESULT_NAME).read_text())
    with pytest.raises(
        alpha.AlphaRefusal, match="evidence_causal_replay_mismatch"
    ) as exc:
        alpha.validate_result(result, directory=captured_preflight)
    assert isinstance(exc.value.__cause__, ArtifactError)
    # The first historical control tape precedes the Full artifact in the
    # authored manifest. Current observation binding rejects it before the
    # later Full replay can reach its independent engine-version guard.
    assert isinstance(exc.value.__cause__, ObservationMismatchError)
    assert "a decision is bound to what produced it" in str(exc.value.__cause__)


def test_cli_has_only_mode_and_output_selection() -> None:
    options = {
        option
        for action in alpha.build_parser()._actions
        for option in action.option_strings
    }
    assert {"--offline-preflight", "--live", "--output-dir"} <= options
    for forbidden in (
        "--model",
        "--profile",
        "--provider",
        "--fixture",
        "--schedule",
        "--plan",
        "--factory",
        "--scenario",
        "--max-calls",
        "--rate",
        "--cost-cap-usd",
        "--input-usd-per-mtok",
        "--output-usd-per-mtok",
    ):
        assert forbidden not in options


def test_schedule_and_descriptive_expectations_are_fixed() -> None:
    assert alpha.ENGINE_VERSION == "0.9.0"
    assert alpha.plan_payload()["operatebench_version"] == alpha.ENGINE_VERSION
    assert alpha.AUTHORED_SCHEDULE == (
        (1, ARM_STATIC),
        (1, ARM_BOUNDARY),
        (1, ARM_ORDERED),
        (1, ARM_FULL),
        (2, ARM_BOUNDARY),
        (2, ARM_ORDERED),
        (2, ARM_FULL),
        (2, ARM_STATIC),
        (3, ARM_ORDERED),
        (3, ARM_FULL),
        (3, ARM_STATIC),
        (3, ARM_BOUNDARY),
    )
    assert alpha.EXPECTED_CALLS == 228
    assert alpha.FULL_EXPECTED_CALLS == 46
    assert alpha.HARD_CALL_CAP == 510
    assert alpha.TURN_DEADLINE_SECONDS == 30.0
    assert alpha.WALL_CLOCK_DEADLINE_SECONDS == 18_000.0
    assert alpha.SDK_MAX_RETRIES == 0
    assert alpha.OPENAI_SDK_VERSION == "2.53.0"
    assert alpha.plan_payload()["openai_sdk_version"] == "2.53.0"


def test_operator_has_no_tracked_internal_source_revision_or_oracle() -> None:
    source = Path(alpha.__file__).read_text()
    assert not hasattr(alpha, "SOURCE_REVISION")
    assert not hasattr(alpha, "COMPILER_SOURCE_REVISION_URL")
    assert re.search(r"\bSOURCE_REVISION\b", source) is None
    assert re.search(r"\bCOMPILER_SOURCE(?:_REVISION)?_URL\b", source) is None
    assert re.search(r"(?<![0-9A-Za-z_-])[0-9a-f]{40}(?![0-9A-Za-z_-])", source) is None
    assert not ({"source_revision", "compiler_source_url"} & alpha.plan_payload().keys())


def test_offline_preflight_runs_real_complete_shape(captured_preflight: Path) -> None:
    output_dir = captured_preflight
    result = json.loads((output_dir / alpha.RESULT_NAME).read_text())
    assert output_dir.stat().st_mode & 0o777 == 0o700
    assert result["schema"] == alpha.RESULT_SCHEMA
    assert result["mode"] == alpha.MODE_OFFLINE
    assert result["external_provider_calls"] == 0
    assert len(result["trials"]) == 12
    assert result["schedule"] == [list(item) for item in alpha.AUTHORED_SCHEDULE]
    assert result["accounting"]["provider_attempts"] > 0
    assert result["accounting"]["retries"] == 0
    assert result["execution_integrity"] == {
        "complete": True,
        "no_replacement": True,
        "control_replays_ok": True,
        "full_bundle_audits_ok": True,
        "full_replays_ok": True,
    }
    assert all(row["vector"] == [ADMISSIBLE] * 5 for row in result["trials"])
    assert len(result["evidence_manifest"]) == 48
    validate_historical_mock_result(result, directory=output_dir)
    expected = set(alpha.OUTPUT_NAMES)
    assert {p.name for p in output_dir.iterdir()} == expected
    assert all((p.stat().st_mode & 0o777) == 0o600 for p in output_dir.iterdir())
    public = json.loads((output_dir / alpha.PUBLIC_SUMMARY_NAME).read_text())
    rendered = json.dumps(public)
    for private in (
        "request_digest",
        "operation_instance",
        "provider_prose",
        "execution_run_id",
    ):
        assert private not in rendered


def test_expected_call_metadata_is_not_an_actual_equality_gate(
    output_dir: Path,
) -> None:
    result = alpha._offline_test_result_fixture()
    assert result["accounting"]["subject_calls"] != result["plan"]["expected_calls"]
    validate_historical_mock_result(result, directory=None, verify_files=False)


def test_historical_source_digest_remains_valid_without_relabelling() -> None:
    result = alpha._offline_test_result_fixture()
    result["operator_source_digest_sha256"] = (
        alpha.HISTORICAL_OPERATOR_SOURCE_DIGEST_SHA256
    )
    source_revision = result["external_source_revision"]
    historical_authorization = alpha._authorization_value_for_operator_digest(
        source_revision, alpha.HISTORICAL_OPERATOR_SOURCE_DIGEST_SHA256
    )
    result["preregistration"]["authorization_digest_sha256"] = alpha._sha_bytes(
        historical_authorization.encode()
    )

    validate_historical_mock_result(result, directory=None, verify_files=False)


def test_source_digest_and_preregistration_authorization_cannot_be_cross_bound() -> None:
    for stated_digest, wrong_authorization_digest in (
        (
            alpha.OPERATOR_SOURCE_DIGEST_SHA256,
            alpha.HISTORICAL_OPERATOR_SOURCE_DIGEST_SHA256,
        ),
        (
            alpha.HISTORICAL_OPERATOR_SOURCE_DIGEST_SHA256,
            alpha.OPERATOR_SOURCE_DIGEST_SHA256,
        ),
    ):
        result = alpha._offline_test_result_fixture()
        result["operator_source_digest_sha256"] = stated_digest
        source_revision = result["external_source_revision"]
        wrong_authorization = alpha._authorization_value_for_operator_digest(
            source_revision, wrong_authorization_digest
        )
        result["preregistration"]["authorization_digest_sha256"] = alpha._sha_bytes(
            wrong_authorization.encode()
        )

        with pytest.raises(
            alpha.AlphaRefusal, match=r"^result_preregistration_mismatch$"
        ):
            validate_historical_mock_result(result, directory=None, verify_files=False)


def test_foreign_self_consistent_source_digest_is_refused() -> None:
    result = alpha._offline_test_result_fixture()
    foreign_digest = "f" * 64
    assert foreign_digest not in {
        alpha.OPERATOR_SOURCE_DIGEST_SHA256,
        alpha.HISTORICAL_OPERATOR_SOURCE_DIGEST_SHA256,
    }
    result["operator_source_digest_sha256"] = foreign_digest
    source_revision = result["external_source_revision"]
    authorization = alpha._authorization_value_for_operator_digest(
        source_revision, foreign_digest
    )
    result["preregistration"]["authorization_digest_sha256"] = alpha._sha_bytes(
        authorization.encode()
    )

    with pytest.raises(alpha.AlphaRefusal, match=r"^result_source_binding_mismatch$"):
        validate_historical_mock_result(result, directory=None, verify_files=False)


def test_plan_expected_calls_is_pinned_even_with_recomputed_digest() -> None:
    result = alpha._offline_test_result_fixture()
    result["plan"]["expected_calls"] += 1
    result["plan_digest_sha256"] = alpha.compute_plan_digest(result["plan"])

    with pytest.raises(alpha.AlphaRefusal, match="result_plan_identity_mismatch"):
        validate_historical_mock_result(result, directory=None, verify_files=False)


def test_manifest_retry_and_aggregate_forgery_is_refused(
    tamperable_preflight: tuple[dict[str, Any], Path],
) -> None:
    result, directory = tamperable_preflight
    result["evidence_manifest"][0]["retries"] += 1
    result["accounting"]["retries"] += 1
    refresh_standalone_manifest(result, directory)

    with pytest.raises(alpha.AlphaRefusal, match="ledger_accounting_binding_mismatch"):
        validate_historical_mock_result(result, directory=directory)


def test_trial_calls_and_evidence_digest_forgery_is_refused(
    tamperable_preflight: tuple[dict[str, Any], Path],
) -> None:
    result, directory = tamperable_preflight
    result["trials"][0]["calls"] += 1
    result["trials"][0]["evidence_digests"].reverse()

    with pytest.raises(alpha.AlphaRefusal, match="trial_evidence_binding_mismatch"):
        validate_historical_mock_result(result, directory=directory)


def test_standalone_manifest_extra_field_with_refreshed_digest_is_refused(
    tamperable_preflight: tuple[dict[str, Any], Path],
) -> None:
    result, directory = tamperable_preflight
    path = directory / alpha.MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["forged"] = True
    path.write_bytes(canonical_json_bytes(manifest, "test manifest") + b"\n")
    result["evidence_manifest_digest_sha256"] = alpha._sha_file(path)

    with pytest.raises(alpha.AlphaRefusal, match="manifest_file_binding_mismatch"):
        validate_historical_mock_result(result, directory=directory)


def test_all_derived_vectors_statuses_and_verdict_cannot_be_rewritten(
    tamperable_preflight: tuple[dict[str, Any], Path],
) -> None:
    result, directory = tamperable_preflight
    replacement = [NOT_ESTABLISHED] * 5
    for row in result["trials"]:
        row["vector"] = replacement.copy()
        row["pass"] = False
    for item in result["evidence_manifest"]:
        item["status"] = NOT_ESTABLISHED
        if item["kind"] == "full":
            item["vector"] = replacement.copy()
    refresh_standalone_manifest(result, directory)
    result["pass_counts"] = dict.fromkeys(
        (ARM_STATIC, ARM_BOUNDARY, ARM_ORDERED, ARM_FULL), 0
    )
    result["differential_verdict"] = alpha.NO_SUPPORTED_LONGITUDINAL_DIFFERENTIAL

    with pytest.raises(alpha.AlphaRefusal, match="evidence_causal_replay_mismatch"):
        validate_historical_mock_result(result, directory=directory)


def test_public_summary_tamper_with_valid_json_is_refused(
    tamperable_preflight: tuple[dict[str, Any], Path],
) -> None:
    result, directory = tamperable_preflight
    path = directory / alpha.PUBLIC_SUMMARY_NAME
    summary = json.loads(path.read_text())
    summary["differential_verdict"] = alpha.UNINTERPRETABLE
    path.write_bytes(canonical_json_bytes(summary, "test summary") + b"\n")

    with pytest.raises(alpha.AlphaRefusal, match="public_summary_file_binding_mismatch"):
        validate_historical_mock_result(result, directory=directory)


def test_relabelled_duplicate_full_bundle_is_refused(
    tamperable_preflight: tuple[dict[str, Any], Path],
) -> None:
    result, directory = tamperable_preflight
    entries = result["evidence_manifest"]
    first = next(
        item for item in entries if item["kind"] == "full" and item["trial"] == 1
    )
    second_index = next(
        index
        for index, item in enumerate(entries)
        if item["kind"] == "full" and item["trial"] == 2
    )
    second = entries[second_index]
    replacement = copy.deepcopy(first)
    replacement.update(
        trial=2,
        ledger=second["ledger"],
        artifact=second["artifact"],
    )
    shutil.copyfile(directory / first["ledger"], directory / second["ledger"])
    shutil.copyfile(directory / first["artifact"], directory / second["artifact"])
    replacement["ledger_digest_sha256"] = alpha._sha_file(directory / second["ledger"])
    replacement["artifact_digest_sha256"] = alpha._sha_file(
        directory / second["artifact"]
    )
    entries[second_index] = replacement
    row = next(
        row for row in result["trials"] if row["arm"] == ARM_FULL and row["trial"] == 2
    )
    row["calls"] = replacement["provider_calls"]
    row["vector"] = copy.deepcopy(replacement["vector"])
    row["pass"] = alpha._trial_pass(row["vector"])
    row["evidence_digests"] = [replacement["artifact_digest_sha256"]]
    for key in (
        "provider_calls",
        "provider_attempts",
        "input_tokens",
        "output_tokens",
        "retries",
    ):
        result["accounting"][key] = sum(item[key] for item in entries)
    result["accounting"]["subject_calls"] = result["accounting"]["provider_calls"]
    result["accounting"]["measured_cost_usd"] = alpha.usd_text(
        sum((Decimal(item["measured_cost_usd"]) for item in entries), Decimal(0))
    )
    refresh_standalone_manifest(result, directory)
    summary_path = directory / alpha.PUBLIC_SUMMARY_NAME
    summary_path.write_bytes(
        canonical_json_bytes(alpha._historical_public_summary(result), "mutated summary")
        + b"\n"
    )
    result["public_summary_digest_sha256"] = alpha._sha_file(summary_path)

    with pytest.raises(alpha.AlphaRefusal, match="duplicate_evidence"):
        validate_historical_mock_result(result, directory=directory)


def test_rechained_ledger_with_wrong_profile_and_controls_is_refused(
    tamperable_preflight: tuple[dict[str, Any], Path],
) -> None:
    result, directory = tamperable_preflight
    entry = next(
        item for item in result["evidence_manifest"] if item["kind"] == "control"
    )
    ledger_path = directory / entry["ledger"]
    ledger_rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    settings = ledger_rows[0]["header"]["provider"]["settings"]
    settings["reasoning"] = {"effort": "low"}
    ledger_rows[0]["header"]["provider"]["settings_digest_sha256"] = ledger_digest(
        settings
    )
    controls = ledger_rows[0]["header"]["controls"]
    controls["turn_deadline_seconds"] = 31.0
    controls["wall_clock_deadline_seconds"] = 18_001.0
    controls["max_provider_calls"] = 9
    previous = None
    for index, ledger_row in enumerate(ledger_rows):
        if index:
            ledger_row["previous_row_digest_sha256"] = previous
        ledger_row["row_digest_sha256"] = row_digest(ledger_row)
        previous = ledger_row["row_digest_sha256"]
    ledger_path.write_bytes(
        b"".join(
            canonical_json_bytes(ledger_row, "mutated ledger row") + b"\n"
            for ledger_row in ledger_rows
        )
    )
    entry["settings_digest_sha256"] = ledger_rows[0]["header"]["provider"][
        "settings_digest_sha256"
    ]
    entry["ledger_digest_sha256"] = alpha._sha_file(ledger_path)
    refresh_standalone_manifest(result, directory)

    with pytest.raises(
        alpha.AlphaRefusal,
        match=r"evidence_provider_identity_mismatch|ledger_header_binding_mismatch",
    ):
        validate_historical_mock_result(result, directory=directory)


def test_rechained_ledger_with_wrong_controls_is_refused(
    tamperable_preflight: tuple[dict[str, Any], Path],
) -> None:
    result, directory = tamperable_preflight
    entry = next(item for item in result["evidence_manifest"] if item["kind"] == "full")
    ledger_path = directory / entry["ledger"]
    ledger_rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    controls = ledger_rows[0]["header"]["controls"]
    controls["max_provider_calls"] = 51
    controls["turn_deadline_seconds"] = 31.0
    controls["wall_clock_deadline_seconds"] = 18_001.0
    previous = None
    for index, ledger_row in enumerate(ledger_rows):
        if index:
            ledger_row["previous_row_digest_sha256"] = previous
        ledger_row["row_digest_sha256"] = row_digest(ledger_row)
        previous = ledger_row["row_digest_sha256"]
    ledger_path.write_bytes(
        b"".join(
            canonical_json_bytes(ledger_row, "mutated ledger row") + b"\n"
            for ledger_row in ledger_rows
        )
    )
    entry["ledger_digest_sha256"] = alpha._sha_file(ledger_path)
    refresh_standalone_manifest(result, directory)

    with pytest.raises(alpha.AlphaRefusal, match="ledger_header_binding_mismatch"):
        validate_historical_mock_result(result, directory=directory)


def test_global_deadline_shortens_turn_and_refuses_late_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    moments = iter((130.0,))

    def clock() -> float:
        return next(moments)

    transport = object.__new__(alpha.AlphaOpenAIResponsesTransport)
    transport._alpha_clock = clock
    transport._alpha_deadline_at = 100.0
    transport._deadline_seconds = 30.0
    transport._clock = clock

    deadline = transport._deadline(99.0)
    assert deadline.remaining_seconds <= 1.0
    monkeypatch.setattr(OpenAIResponsesTransport, "send", lambda self, request: object())
    with pytest.raises(alpha.AlphaFailure, match="wall_deadline_exceeded"):
        transport.send(object())


class HostileCallerMapping(Mapping[str, Any]):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def _fail(self, hook: str) -> Any:
        self.calls.append(hook)
        raise AssertionError(f"caller data traversed through {hook}")

    def __getitem__(self, key: str) -> Any:
        return self._fail("__getitem__")

    def __iter__(self) -> Iterator[str]:
        return self._fail("__iter__")

    def __len__(self) -> int:
        return self._fail("__len__")

    def get(self, key: str, default: Any = None) -> Any:
        return self._fail("get")

    def items(self) -> Any:
        return self._fail("items")

    def keys(self) -> Any:
        return self._fail("keys")

    def values(self) -> Any:
        return self._fail("values")


def test_public_differential_verdict_refuses_without_traversing_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vectors = HostileCallerMapping()
    monkeypatch.setattr(alpha, "_trial_pass", lambda vector: pytest.fail("science read"))

    with pytest.raises(alpha.AlphaRefusal) as raised:
        alpha.differential_verdict(vectors)

    assert raised.value.code == "alpha_operator_quarantined"
    assert vectors.calls == []


def test_public_summary_refuses_without_traversing_arguments() -> None:
    result = HostileCallerMapping()

    with pytest.raises(alpha.AlphaRefusal) as raised:
        alpha.public_summary(result)

    assert raised.value.code == "alpha_operator_quarantined"
    assert result.calls == []


def test_verdict_is_vector_only_and_any_error_is_uninterpretable() -> None:
    passing = [ADMISSIBLE] * 5
    failing = [NOT_ESTABLISHED, *passing[1:]]
    vectors = {
        ARM_STATIC: [passing, passing, failing],
        ARM_BOUNDARY: [passing, passing, failing],
        ARM_ORDERED: [passing, passing, failing],
        ARM_FULL: [failing, failing, passing],
    }
    assert (
        alpha._historical_differential_verdict(vectors)
        == alpha.SUPPORTED_LONGITUDINAL_DIFFERENTIAL
    )
    vectors[ARM_STATIC][0] = [ERROR, *passing[1:]]
    assert alpha._historical_differential_verdict(vectors) == alpha.UNINTERPRETABLE


class UnreadCredentialEnvironment(MutableMapping[str, str]):
    def __init__(self, values: Mapping[str, str]) -> None:
        self.data = dict(values)

    def __getitem__(self, key: str) -> str:
        if key == alpha.CREDENTIAL_VARIABLE:
            raise AssertionError("credential read before refusal")
        return self.data[key]

    def __setitem__(self, key: str, value: str) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MODEL", "moved"),
        ("PROFILE_ID", "moved"),
        ("POLICY_ID", "moved"),
        ("RATE_SOURCE", "moved"),
        ("INPUT_USD_PER_MTOK", Decimal("0.21")),
        ("OUTPUT_USD_PER_MTOK", Decimal("1.21")),
        ("CELL_CAP_USD", Decimal("5.01")),
        ("HARD_CALL_CAP", 511),
        ("EXPECTED_CALLS", 229),
        ("MAX_POINT_CALLS", 9),
        ("MAX_ATTEMPTS_PER_CALL", 2),
        ("SDK_MAX_RETRIES", 1),
        ("OPENAI_SDK_VERSION", "2.54.0"),
        ("TURN_DEADLINE_SECONDS", 31.0),
        ("WALL_CLOCK_DEADLINE_SECONDS", 18001.0),
        ("SPEC_DIGEST_SHA256", "0" * 64),
        ("COMPILER_ID", "moved"),
        ("SCAFFOLD_ID", "moved"),
    ],
)
def test_each_mutated_build_control_refuses_before_credential_or_construction(
    output_dir: Path, monkeypatch: pytest.MonkeyPatch, name: str, value: Any
) -> None:
    monkeypatch.setattr(alpha, name, value)
    monkeypatch.setattr(
        alpha, "open_live_transport", lambda: pytest.fail("transport built")
    )
    monkeypatch.setattr(
        alpha, "open_live_client", lambda *a, **k: pytest.fail("client built")
    )
    env = UnreadCredentialEnvironment(
        {
            alpha.LIVE_AUTHORIZATION_VARIABLE: "wrong",
            alpha.CREDENTIAL_VARIABLE: "sk-nev...0000",
        }
    )
    with pytest.raises(alpha.AlphaRefusal):
        alpha._run_live(directory=output_dir, environ=env)
    assert not any(output_dir.iterdir())


class CloseCountingClient:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


@pytest.mark.parametrize("failure_stage", [None, "execution", "validation"])
def test_offline_client_closes_exactly_once(
    output_dir: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str | None
) -> None:
    """The quarantined boundary constructs no client, irrespective of old stages."""
    monkeypatch.setattr(
        alpha.openai, "OpenAI", lambda **kwargs: pytest.fail("client constructed")
    )
    monkeypatch.setattr(alpha, "_execute", lambda **kwargs: pytest.fail("executed"))

    with pytest.raises(alpha.AlphaRefusal, match=r"^alpha_operator_quarantined$"):
        alpha.run_offline_preflight(directory=output_dir, environ={})
    assert not any(output_dir.iterdir())


@pytest.mark.parametrize("failure_stage", [None, "execution", "validation"])
def test_live_client_closes_exactly_once(
    output_dir: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str | None
) -> None:
    """The quarantined live boundary constructs no client to require closing."""
    monkeypatch.setattr(
        alpha, "open_live_transport", lambda: pytest.fail("transport constructed")
    )
    monkeypatch.setattr(
        alpha,
        "open_live_client",
        lambda *args, **kwargs: pytest.fail("client constructed"),
    )
    monkeypatch.setattr(alpha, "_execute", lambda **kwargs: pytest.fail("executed"))

    with pytest.raises(alpha.AlphaRefusal, match=r"^alpha_operator_quarantined$"):
        alpha._run_live(directory=output_dir, environ=UnreadCredentialEnvironment({}))
    assert not any(output_dir.iterdir())


def test_installed_openai_sdk_drift_refuses_before_credential_or_construction(
    output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(alpha.openai, "__version__", "2.54.0")
    monkeypatch.setattr(
        alpha, "check_pinned_build", lambda: pytest.fail("build inspected")
    )
    monkeypatch.setattr(
        alpha, "open_live_transport", lambda: pytest.fail("transport built")
    )
    monkeypatch.setattr(
        alpha, "open_live_client", lambda *a, **k: pytest.fail("client built")
    )
    env = UnreadCredentialEnvironment({alpha.CREDENTIAL_VARIABLE: "«redacted:sk-…»"})

    with pytest.raises(alpha.AlphaRefusal, match=r"^alpha_operator_quarantined$"):
        alpha._run_live(directory=output_dir, environ=env)
    assert not any(output_dir.iterdir())


@pytest.mark.parametrize(
    "git_error",
    [
        OSError("git missing: secret detail"),
        subprocess.CalledProcessError(
            128,
            ["git", "rev-parse", "HEAD"],
            output="secret stdout",
            stderr="secret stderr",
        ),
    ],
)
def test_git_failures_are_stable_safe_refusals(
    monkeypatch: pytest.MonkeyPatch, git_error: BaseException
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise git_error

    monkeypatch.setattr(alpha.subprocess, "run", fail)

    with pytest.raises(
        alpha.AlphaRefusal, match=r"^source_revision_unavailable$"
    ) as caught:
        alpha._git_revision_and_clean()
    rendered = str(caught.value)
    assert "secret" not in rendered
    assert "rev-parse" not in rendered


def test_cli_reports_stable_git_refusal_without_command_output(
    output_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("git command reached")

    monkeypatch.setattr(alpha.subprocess, "run", fail)

    assert alpha.main(["--live", "--output-dir", str(output_dir)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["failure"] == {"code": "alpha_operator_quarantined"}
    assert "secret" not in json.dumps(report)
    assert not any(output_dir.iterdir())


def test_untracked_source_refuses_before_credential_or_client(
    output_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "source"
    operator = repo / "tools" / "matched_arm_alpha.py"
    operator.parent.mkdir(parents=True)
    operator.write_text("# untracked historical operator fixture\n")
    monkeypatch.setattr(alpha, "REPO_ROOT", repo)
    monkeypatch.setattr(alpha, "__file__", str(operator))
    monkeypatch.setattr(alpha, "_git_revision_and_clean", lambda: pytest.fail("git read"))
    monkeypatch.setattr(alpha, "open_live_client", lambda *a, **k: pytest.fail("client"))
    env = UnreadCredentialEnvironment({alpha.CREDENTIAL_VARIABLE: "«redacted:sk-…»"})

    with pytest.raises(alpha.AlphaRefusal, match=r"^alpha_operator_quarantined$"):
        alpha._run_live(directory=output_dir, environ=env)
    assert not any(output_dir.iterdir())


def test_wrong_external_source_refuses_before_credential_or_client(
    output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(alpha, "_git_revision_and_clean", lambda: pytest.fail("git read"))
    monkeypatch.setattr(alpha, "open_live_transport", lambda: pytest.fail("transport"))
    monkeypatch.setattr(alpha, "open_live_client", lambda *a, **k: pytest.fail("client"))
    env = UnreadCredentialEnvironment(
        {
            alpha.EXPECTED_SOURCE_REVISION_VARIABLE: "f" * 40,
            alpha.CREDENTIAL_VARIABLE: "sk-nev...0000",
        }
    )

    with pytest.raises(alpha.AlphaRefusal, match=r"^alpha_operator_quarantined$"):
        alpha._run_live(directory=output_dir, environ=env)
    assert not any(output_dir.iterdir())


def test_output_collision_refuses_before_credential(
    output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collision = output_dir / alpha.RESULT_NAME
    collision.symlink_to("missing")
    monkeypatch.setattr(
        alpha, "reserve_output_names", lambda *a, **k: pytest.fail("reservation reached")
    )
    env = UnreadCredentialEnvironment({alpha.CREDENTIAL_VARIABLE: "sk-nev...0000"})

    with pytest.raises(alpha.AlphaRefusal, match=r"^alpha_operator_quarantined$"):
        alpha._run_live(directory=output_dir, environ=env)
    assert collision.is_symlink()


def test_closed_result_schema_rejects_unknown_missing_bool_and_nonfinite() -> None:
    base = alpha._offline_test_result_fixture()
    mutations = []
    unknown = json.loads(json.dumps(base))
    unknown["unknown"] = 1
    mutations.append(unknown)
    missing = json.loads(json.dumps(base))
    del missing["schema"]
    mutations.append(missing)
    boolean = json.loads(json.dumps(base))
    boolean["accounting"]["subject_calls"] = True
    mutations.append(boolean)
    negative = json.loads(json.dumps(base))
    negative["accounting"]["input_tokens"] = -1
    mutations.append(negative)
    nonfinite = json.loads(json.dumps(base))
    nonfinite["accounting"]["measured_cost_usd"] = "NaN"
    mutations.append(nonfinite)
    for mutation in mutations:
        with pytest.raises(alpha.AlphaRefusal):
            validate_historical_mock_result(mutation, directory=None, verify_files=False)


def test_live_cli_is_quarantined_before_credentials_providers_or_output(
    output_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sentinel = output_dir / "preserve.txt"
    sentinel.write_text("historical bytes\n")
    before = {path.name: path.read_bytes() for path in output_dir.iterdir()}

    class HostileEnvironment(MutableMapping[str, str]):
        def __init__(self, values: Mapping[str, str]) -> None:
            self.data = dict(values)
            self.sensitive = {
                alpha.CREDENTIAL_VARIABLE,
                alpha.LIVE_AUTHORIZATION_VARIABLE,
                alpha.EXPECTED_SOURCE_REVISION_VARIABLE,
                *REJECTED_VARIABLES,
            }

        def __getitem__(self, key: str) -> str:
            if key in self.sensitive:
                raise AssertionError(f"environment read: {key}")
            return self.data[key]

        def __setitem__(self, key: str, value: str) -> None:
            if key in self.sensitive:
                raise AssertionError(f"environment write: {key}")
            self.data[key] = value

        def __delitem__(self, key: str) -> None:
            if key in self.sensitive:
                raise AssertionError(f"environment delete: {key}")
            del self.data[key]

        def __iter__(self) -> Iterator[str]:
            return iter(self.data)

        def __len__(self) -> int:
            return len(self.data)

    monkeypatch.setattr(alpha.os, "environ", HostileEnvironment(alpha.os.environ))
    monkeypatch.setattr(alpha, "check_pinned_build", lambda: pytest.fail("build read"))
    monkeypatch.setattr(alpha, "compile_pinned_arms", lambda: pytest.fail("compiled"))
    monkeypatch.setattr(
        alpha, "reserve_output_names", lambda *a, **k: pytest.fail("reserved")
    )
    monkeypatch.setattr(
        alpha, "open_live_transport", lambda: pytest.fail("network built")
    )
    monkeypatch.setattr(
        alpha, "open_live_client", lambda *a, **k: pytest.fail("provider built")
    )

    assert alpha.main(["--live", "--output-dir", str(output_dir)]) == 1

    report = json.loads(capsys.readouterr().out)
    assert report == {
        "failure": {"code": "alpha_operator_quarantined"},
        "mode": "historical_quarantine",
        "ok": False,
    }
    assert "SUPPORTED_LONGITUDINAL_DIFFERENTIAL" not in json.dumps(report)
    assert "NO_SUPPORTED_LONGITUDINAL_DIFFERENTIAL" not in json.dumps(report)
    assert {path.name: path.read_bytes() for path in output_dir.iterdir()} == before
    assert not (output_dir / alpha.PUBLIC_SUMMARY_NAME).exists()


def test_all_cli_forms_refuse_and_help_states_permanent_uninterpretability(
    tmp_path: Path,
) -> None:
    env = {
        **os.environ,
        alpha.CREDENTIAL_VARIABLE: "hostile credential sentinel",
        alpha.LIVE_AUTHORIZATION_VARIABLE: "hostile authorization sentinel",
        alpha.EXPECTED_SOURCE_REVISION_VARIABLE: "f" * 40,
        "PYTHONPATH": f"{alpha.REPO_ROOT / 'src'}:{alpha.REPO_ROOT}",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    expected = {
        "failure": {"code": "alpha_operator_quarantined"},
        "mode": "historical_quarantine",
        "ok": False,
    }
    invocations = ((), ("--live",), ("--offline-preflight",))
    for index, arguments in enumerate(invocations):
        directory = tmp_path / f"output-{index}"
        directory.mkdir()
        sentinel = directory / "sealed-history.json"
        sentinel.write_bytes(b'{"verdict":"UNINTERPRETABLE"}\n')
        command = [
            sys.executable,
            "-m",
            "tools.matched_arm_alpha",
            *arguments,
            "--output-dir",
            str(directory),
        ]
        completed = subprocess.run(
            command,
            cwd=alpha.REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 1
        assert completed.stderr == ""
        assert json.loads(completed.stdout) == expected
        assert list(directory.iterdir()) == [sentinel]
        assert sentinel.read_bytes() == b'{"verdict":"UNINTERPRETABLE"}\n'
        assert not (directory / alpha.PUBLIC_SUMMARY_NAME).exists()
        assert alpha.SUPPORTED_LONGITUDINAL_DIFFERENTIAL not in completed.stdout
        assert alpha.NO_SUPPORTED_LONGITUDINAL_DIFFERENTIAL not in completed.stdout

    help_result = subprocess.run(
        [sys.executable, "-m", "tools.matched_arm_alpha", "--help"],
        cwd=alpha.REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_result.returncode == 0
    help_text = help_result.stdout.lower()
    for required in (
        "historical",
        "diagnostic",
        "quarantined",
        "permanently uninterpretable",
        "no rerun or regrade",
    ):
        assert required in help_text


def test_execution_entry_points_are_quarantined_before_environment_or_output(
    output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class UnreadEnvironment(MutableMapping[str, str]):
        def __init__(self, values: Mapping[str, str]) -> None:
            self.data = dict(values)
            self.sensitive = {
                alpha.CREDENTIAL_VARIABLE,
                alpha.LIVE_AUTHORIZATION_VARIABLE,
                alpha.EXPECTED_SOURCE_REVISION_VARIABLE,
                *REJECTED_VARIABLES,
            }

        def __getitem__(self, key: str) -> str:
            if key in self.sensitive:
                raise AssertionError(f"environment read: {key}")
            return self.data[key]

        def __setitem__(self, key: str, value: str) -> None:
            if key in self.sensitive:
                raise AssertionError(f"environment write: {key}")
            self.data[key] = value

        def __delitem__(self, key: str) -> None:
            if key in self.sensitive:
                raise AssertionError(f"environment delete: {key}")
            del self.data[key]

        def __iter__(self) -> Iterator[str]:
            return iter(self.data)

        def __len__(self) -> int:
            return len(self.data)

    environ = UnreadEnvironment(alpha.os.environ)
    monkeypatch.setattr(alpha.os, "environ", environ)
    for invocation in (
        lambda: alpha._run_live(directory=output_dir, environ=environ),
        lambda: alpha.run_live(directory=output_dir),
        lambda: alpha.run_offline_preflight(directory=output_dir, environ=environ),
    ):
        with pytest.raises(alpha.AlphaRefusal, match=r"^alpha_operator_quarantined$"):
            invocation()
        assert not any(output_dir.iterdir())


def test_public_historical_methodology_preserves_non_relabeling() -> None:
    methodology = (alpha.REPO_ROOT / "docs" / "HISTORICAL_RUNTIME.md").read_text()
    assert "No historical artifact is relabelled or repinned." in methodology
    assert "never a new real historical run or current-engine substitution" in methodology
    assert not (alpha.REPO_ROOT / "docs" / "PRODUCT_DECISION.md").exists()
    assert not (alpha.REPO_ROOT / "docs" / "PHASE_A_APPROVAL_RECORD.md").exists()
