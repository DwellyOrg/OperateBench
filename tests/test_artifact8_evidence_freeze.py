"""B3.1 byte-exact Artifact 8 and execution-ledger 2 evidence freeze.

These are static historical oracles.  Tests read and audit them; regeneration is
an explicit maintenance operation and never test setup.
"""

# ruff: noqa: E501 -- immutable SHA-256 oracle literals are intentionally unsplit.

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import shutil
import socket
import stat
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest

from operatebench.artifact import read_artifact
from operatebench.execution_bundle import audit_execution_bundle_files
from operatebench.execution_ledger import (
    IncompleteExecutionLedgerError,
    read_execution_ledger,
)
from tests.test_public_release import _canonical_distribution
from tools import check_public_release as release_checks

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "b3"
MANIFEST_NAME = "artifact8-evidence-freeze-v1.json"
DETERMINISTIC_NAME = "artifact8-deterministic-reference-v1.json"
MODEL_NAME = "artifact8-fake-openai-reference-v1.json"
LEDGER_NAME = "artifact8-fake-openai-reference-v1.execution-ledger-v2.ndjson"

PINS = {
    MANIFEST_NAME: (
        2777,
        "6ec8409ca55dc4fbb0f5d9078f25004429edd6525209d1e8080b163b36a2b8c9",
    ),
    DETERMINISTIC_NAME: (
        220582,
        "45f2f3bec377230a0fc464c5ea1b64798a053beeb588c3ec4bdb477eaf7cfab5",
    ),
    MODEL_NAME: (
        228801,
        "115bd5c058f4b4057cf7b4cd18d7cf0b367c0234b4367fbac92396aa26a24b3f",
    ),
    LEDGER_NAME: (
        88863,
        "29fc1c33d601529151b9340bc253992c3b55c28ba4c55849f070df13ead7afdb",
    ),
}
CHAIN = "73f0c6b77ff6e774ded0c325ee007fe62f62999627ffcd8ae915474809f41b2f"
SETTINGS = "5b4a120cfaf3821807d60a3b9ffb848acd90f28cc9b2be496ac054abcbf764e2"
PRICING = "3988f30b258bc99a96205bed3d3cef2425efb7229416a1a5e80715a7eae0906e"
EXPECTED_GENERATION = {
    "execution_run_id": "exec_0123456789abcdef0123456789abcdef",
    "ledger_instant": "2035-01-02T03:04:05Z",
    "operation_fixture": "examples/operatebench/maintenance_v0_1.yaml",
    "operation_fixture_sha256": "ea01e3c208614f69f2dacd4f8749beb4e5724871dc21b17505ad79fd809a946e",
    "operation_instance_id": "opinst_0123456789abcdef0123456789abcdef",
    "uv_lock_sha256": "a3c1d226486cb93344206f4da959b0e2d52926c4d72e777ba1a9a1c758848baf",
}
EXPECTED_FIXTURES = {
    "deterministic": {
        "artifact": "artifact8-deterministic-reference-v1.json",
        "artifact_bytes": 220582,
        "artifact_sha256": "45f2f3bec377230a0fc464c5ea1b64798a053beeb588c3ec4bdb477eaf7cfab5",
        "ledger": None,
    },
    "fake_model": {
        "artifact": "artifact8-fake-openai-reference-v1.json",
        "artifact_bytes": 228801,
        "artifact_sha256": "115bd5c058f4b4057cf7b4cd18d7cf0b367c0234b4367fbac92396aa26a24b3f",
        "ledger": "artifact8-fake-openai-reference-v1.execution-ledger-v2.ndjson",
        "ledger_bytes": 88863,
        "ledger_chain_sha256": "73f0c6b77ff6e774ded0c325ee007fe62f62999627ffcd8ae915474809f41b2f",
        "ledger_file_sha256": "29fc1c33d601529151b9340bc253992c3b55c28ba4c55849f070df13ead7afdb",
    },
}
EXPECTED_IDENTITY = {
    "compatibility_registry_resource_sha256": "bcb84fdac5a545e34f002113eed1ac7231d7387c0cd61a6043acc6895282ba44",
    "component_registry_resource_sha256": "852690707e5864ba3d15591094a9c1872dbce2ec4458483c2523ea554553db5f",
    "components": {
        "analysis_contract_content": "506519362dc50e3dbd506e311e2bf33062f95d1e474a6b002a4b9dc94715ff90",
        "arm_protocol_content": "21cf80a871163365202943612dfe8c3be0f752e8d0da1bae3a383eb9c47c3826",
        "build_provenance_content": "3bd17b1465a01f24831230441bcd32993d7b7adcc6963d9641bdbba3a9bb45c7",
        "card_schema_content": "6ea65e919b821f70bcd52825e58b398d1ff532143fef21122ded8434a499a99b",
        "evaluator_bundle_content": "e197fea41f20c7780d0592f8ddde5cbae0a823b1ead09603fc6608752af2402d",
        "operation_core_content": "fb1f8e065c89c00aab83ebf201dc22e78d168617ff51c12d54a95d08b404fac0",
        "provider_run_plan_content": "9bcc863d148afe1a81d558ad0cec1a22b8ec2bc17efacade5ec3abc6ce092b4d",
        "runtime_contract_content": "4e4cf36e4eb708e52185dd61ac2988ac0f83c73c9cbfd545d65b34d972e1e2d0",
        "scaffold_content": "d383710341102d6549c236803329acd202e31ad4f646ccef43098f35c4451060",
        "semantic_scenario_content": "1b689ec623535ecdbd16ca5d9ee1f51f0aa6759ae217eb3bcb986eb2ca6e9bf8",
        "variant_content": "0add47d249cd6e8a55dfb79d29356dd0f6ece3a4e372236d98fb3a9b48146781",
    },
    "composites": {
        "experiment_bundle": "dea1f4bed6860d3a5e78ee2e2e7386d6aa67b84878a6eb8be8d7d9beb6efe80e",
        "runtime_bundle": "251e36a96f2c4cf474d06cef7d3ff3d39aab75211b4b9d191f1c4c4647b9a1bc",
    },
    "golden_resource_sha256": "01bceeb07cf195415d519f570e13c697b80f0c4ab4ae7945763fcc1a83225948",
    "manifest_digest_sha256": "44a47e0c8229e14ed96b176223d6e798885ed5289e4375aa3d9e6b1cda4d444c",
}


def _canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _pinned(path: Path, pin: tuple[int, str]) -> bytes:
    raw = path.read_bytes()
    assert len(raw) == pin[0]
    assert _sha(raw) == pin[1]
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    return raw


def _audit(directory: Path = FIXTURES) -> dict[str, Any]:
    manifest_raw = _pinned(directory / MANIFEST_NAME, PINS[MANIFEST_NAME])
    manifest = json.loads(manifest_raw)
    assert manifest_raw == _canonical(manifest)
    assert manifest["schema"] == "operatebench.b3.artifact8_evidence_freeze.v1"
    assert manifest["schema_version"] == 1
    assert manifest["fixtures"] == EXPECTED_FIXTURES
    assert manifest["generation"] == EXPECTED_GENERATION
    assert manifest["identity"] == EXPECTED_IDENTITY

    deterministic_fixture = manifest["fixtures"]["deterministic"]
    model_fixture = manifest["fixtures"]["fake_model"]
    assert deterministic_fixture["artifact"] == DETERMINISTIC_NAME
    assert deterministic_fixture["ledger"] is None
    assert model_fixture["artifact"] == MODEL_NAME
    assert model_fixture["ledger"] == LEDGER_NAME

    operation = ROOT / manifest["generation"]["operation_fixture"]
    assert (
        _sha(operation.read_bytes()) == manifest["generation"]["operation_fixture_sha256"]
    )
    assert (
        _sha((ROOT / "uv.lock").read_bytes()) == manifest["generation"]["uv_lock_sha256"]
    )

    resources = ROOT / "src" / "operatebench" / "resources" / "identity"
    identity = manifest["identity"]
    for key, name in (
        ("golden_resource_sha256", "identity-manifest-digest-v1.golden.json"),
        ("component_registry_resource_sha256", "identity-component-registry-v1.json"),
        (
            "compatibility_registry_resource_sha256",
            "identity-compatibility-v1.registry.json",
        ),
    ):
        assert _sha((resources / name).read_bytes()) == identity[key]

    deterministic_raw = _pinned(directory / DETERMINISTIC_NAME, PINS[DETERMINISTIC_NAME])
    model_raw = _pinned(directory / MODEL_NAME, PINS[MODEL_NAME])
    ledger_raw = _pinned(directory / LEDGER_NAME, PINS[LEDGER_NAME])
    assert deterministic_raw == _canonical(json.loads(deterministic_raw))
    assert model_raw == _canonical(json.loads(model_raw))
    for line in ledger_raw.splitlines(keepends=True):
        assert line.endswith(b"\n")
        row = json.loads(line)
        assert (
            line == (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode()
        )

    deterministic = read_artifact(directory / DETERMINISTIC_NAME)
    model = read_artifact(directory / MODEL_NAME)
    ledger = read_execution_ledger(directory / LEDGER_NAME, require_complete=True)
    bundle = audit_execution_bundle_files(directory / MODEL_NAME, directory / LEDGER_NAME)

    assert (len(deterministic_raw), _sha(deterministic_raw)) == (
        deterministic_fixture["artifact_bytes"],
        deterministic_fixture["artifact_sha256"],
    )
    assert (len(model_raw), _sha(model_raw)) == (
        model_fixture["artifact_bytes"],
        model_fixture["artifact_sha256"],
    )
    assert (len(ledger_raw), _sha(ledger_raw)) == (
        model_fixture["ledger_bytes"],
        model_fixture["ledger_file_sha256"],
    )

    assert deterministic["artifact_version"] == model["artifact_version"] == 8
    assert deterministic["engine_version"] == model["engine_version"] == "0.9.0"
    assert (
        deterministic["operation_instance_id"]
        == model["operation_instance_id"]
        == manifest["generation"]["operation_instance_id"]
    )
    assert deterministic["scenario_id"] == model["scenario_id"] == "V1"
    assert deterministic["agent_id"] == "reference"
    assert deterministic["agent_execution"]["outcome_source"] == "deterministic"
    assert deterministic["provider_execution"] is None
    assert model["agent_id"] == "openai-gpt-5-6-luna"
    assert model["agent_execution"]["outcome_source"] == "model"
    assert model["agent_execution"]["max_output_tokens"] == 4096
    assert (
        model["provider_execution"]["execution_run_id"]
        == manifest["generation"]["execution_run_id"]
    )

    assert ledger.rows_verified == 48
    assert ledger.ledger_version == bundle.execution_ledger_version == 2
    assert bundle.summary()["execution_ledger_version"] == 2
    assert len(ledger.calls) == bundle.provider_calls == 46
    assert ledger.scored and ledger.complete
    assert ledger.ledger_digest_sha256 == model_fixture["ledger_chain_sha256"]
    assert ledger.ledger_digest_sha256 == CHAIN
    assert ledger.header.provider.settings_digest_sha256 == SETTINGS
    assert ledger.header.controls.pricing_digest_sha256 == PRICING
    assert bundle.ok
    assert b"sk-test" not in model_raw + ledger_raw
    return manifest


def _copy_fixture_tree(tmp_path: Path) -> Path:
    target = tmp_path / "b3"
    shutil.copytree(FIXTURES, target)
    return target


def _load_with_source_pins(
    replacements: dict[str, tuple[int, str]],
) -> dict[str, Any]:
    """Load this module after replacing only literal entries in source PINS."""

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=__file__)
    pins = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "PINS"
            for target in node.targets
        )
    )
    assert isinstance(pins.value, ast.Dict)
    for index, key in enumerate(pins.value.keys):
        assert isinstance(key, ast.Name)
        if key.id in replacements:
            size, digest = replacements[key.id]
            pins.value.values[index] = ast.Tuple(
                elts=[ast.Constant(size), ast.Constant(digest)], ctx=ast.Load()
            )

    namespace: dict[str, Any] = {
        "__file__": __file__,
        "__name__": "tests._artifact8_evidence_freeze_source_repin",
    }
    exec(compile(ast.fix_missing_locations(tree), __file__, "exec"), namespace)
    return namespace


def test_missing_freeze_manifest_is_red() -> None:
    assert (FIXTURES / MANIFEST_NAME).is_file()


def test_frozen_source_tree_bytes_and_production_audits() -> None:
    _audit()


@pytest.mark.parametrize("name", PINS)
def test_each_frozen_file_tamper_is_refused(tmp_path: Path, name: str) -> None:
    directory = _copy_fixture_tree(tmp_path)
    path = directory / name
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 1
    path.write_bytes(raw)
    with pytest.raises(AssertionError):
        _audit(directory)


@pytest.mark.parametrize("name", PINS)
def test_wrong_size_or_final_lf_is_refused(tmp_path: Path, name: str) -> None:
    directory = _copy_fixture_tree(tmp_path)
    path = directory / name
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(AssertionError):
        _audit(directory)


@pytest.mark.parametrize("name", [MANIFEST_NAME, DETERMINISTIC_NAME, MODEL_NAME])
def test_noncanonical_json_is_refused(tmp_path: Path, name: str) -> None:
    directory = _copy_fixture_tree(tmp_path)
    path = directory / name
    path.write_text(
        json.dumps(json.loads(path.read_bytes()), indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(AssertionError):
        _audit(directory)


def test_missing_model_ledger_is_refused(tmp_path: Path) -> None:
    directory = _copy_fixture_tree(tmp_path)
    (directory / LEDGER_NAME).unlink()
    with pytest.raises(FileNotFoundError):
        _audit(directory)


def test_pairing_is_closed_by_exact_artifact_and_ledger_hashes(tmp_path: Path) -> None:
    directory = _copy_fixture_tree(tmp_path)
    (directory / MODEL_NAME).write_bytes((directory / DETERMINISTIC_NAME).read_bytes())
    with pytest.raises(AssertionError):
        _audit(directory)


def test_manifest_fixture_pin_refuses_one_literal_artifact_repin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _copy_fixture_tree(tmp_path)
    path = directory / MODEL_NAME
    payload = json.loads(path.read_bytes())
    payload["scenario_label"] = "repinned historical label"
    raw = _canonical(payload)
    path.write_bytes(raw)
    monkeypatch.setitem(PINS, MODEL_NAME, (len(raw), _sha(raw)))

    with pytest.raises(AssertionError):
        _audit(directory)


@pytest.mark.parametrize(
    ("fixture_key", "artifact_name_constant"),
    [
        ("deterministic", "DETERMINISTIC_NAME"),
        ("fake_model", "MODEL_NAME"),
    ],
)
def test_coherent_artifact_manifest_and_source_pin_repin_is_refused(
    tmp_path: Path, fixture_key: str, artifact_name_constant: str
) -> None:
    directory = _copy_fixture_tree(tmp_path)
    artifact_name = globals()[artifact_name_constant]
    artifact_path = directory / artifact_name
    artifact = json.loads(artifact_path.read_bytes())
    artifact["scenario_label"] = "repinned historical label"
    artifact_raw = _canonical(artifact)
    artifact_path.write_bytes(artifact_raw)

    manifest_path = directory / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_bytes())
    manifest_fixture = manifest["fixtures"][fixture_key]
    manifest_fixture["artifact_bytes"] = len(artifact_raw)
    manifest_fixture["artifact_sha256"] = _sha(artifact_raw)
    manifest_raw = _canonical(manifest)
    manifest_path.write_bytes(manifest_raw)

    repinned = _load_with_source_pins(
        {
            artifact_name_constant: (len(artifact_raw), _sha(artifact_raw)),
            "MANIFEST_NAME": (len(manifest_raw), _sha(manifest_raw)),
        }
    )
    with pytest.raises(AssertionError):
        repinned["_audit"](directory)


def test_expected_fixtures_are_authored_only_from_independent_literals() -> None:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    expected = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "EXPECTED_FIXTURES"
            for target in node.targets
        )
    )
    references = {
        node.id for node in ast.walk(expected.value) if isinstance(node, ast.Name)
    }
    assert references == set()


def test_incomplete_ledger_is_refused_even_if_file_pin_is_temporarily_updated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _copy_fixture_tree(tmp_path)
    path = directory / LEDGER_NAME
    raw = b"".join(path.read_bytes().splitlines(keepends=True)[:-1])
    path.write_bytes(raw)
    monkeypatch.setitem(PINS, LEDGER_NAME, (len(raw), _sha(raw)))
    with pytest.raises(IncompleteExecutionLedgerError):
        _audit(directory)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        (None, "engine_version", "9.9.9"),
        ("operation", "operation_id", "unknown"),
        (None, "scenario_id", "V2"),
        ("agent_execution", "protocol_version", "unknown"),
        ("agent_execution", "model", "unknown"),
    ],
)
def test_historical_label_drift_is_refused_by_closed_byte_pin(
    tmp_path: Path, section: str | None, field: str, value: str
) -> None:
    directory = _copy_fixture_tree(tmp_path)
    path = directory / MODEL_NAME
    payload = json.loads(path.read_bytes())
    target = payload if section is None else payload[section]
    target[field] = value
    path.write_bytes(_canonical(payload))
    with pytest.raises(AssertionError):
        _audit(directory)


@pytest.mark.parametrize(
    "path",
    [
        ("generation", "operation_fixture_sha256"),
        ("generation", "uv_lock_sha256"),
        ("identity", "golden_resource_sha256"),
        ("identity", "component_registry_resource_sha256"),
        ("identity", "compatibility_registry_resource_sha256"),
        ("identity", "manifest_digest_sha256"),
        ("identity", "components", "evaluator_bundle_content"),
        ("identity", "composites", "runtime_bundle"),
    ],
)
def test_manifest_provenance_or_b2_drift_is_refused(
    tmp_path: Path, path: tuple[str, ...]
) -> None:
    directory = _copy_fixture_tree(tmp_path)
    manifest_path = directory / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_bytes())
    target = manifest
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = "0" * 64
    manifest_path.write_bytes(_canonical(manifest))
    with pytest.raises(AssertionError):
        _audit(directory)


def test_manifest_public_provenance_excludes_git_revision_and_keeps_independent_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _copy_fixture_tree(tmp_path)
    manifest_path = directory / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_bytes())
    assert "source_revision" not in manifest["generation"]
    commit_identifier = re.compile(r"(?<![0-9A-Za-z_-])[0-9A-Fa-f]{40}(?![0-9A-Za-z_-])")
    assert commit_identifier.search(manifest_path.read_text(encoding="utf-8")) is None

    for section, field in (
        ("generation", "operation_fixture_sha256"),
        ("identity", "golden_resource_sha256"),
    ):
        changed = json.loads((FIXTURES / MANIFEST_NAME).read_bytes())
        changed[section][field] = "0" * 64
        raw = _canonical(changed)
        manifest_path.write_bytes(raw)
        monkeypatch.setitem(PINS, MANIFEST_NAME, (len(raw), _sha(raw)))
        with pytest.raises(AssertionError):
            _audit(directory)


def test_model_fixture_requires_the_production_bundle_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refused(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("bundle audit did not pass")

    monkeypatch.setattr(
        "tests.test_artifact8_evidence_freeze.audit_execution_bundle_files",
        refused,
    )
    with pytest.raises(RuntimeError, match="bundle audit did not pass"):
        _audit()


def test_acceptance_constructs_no_provider_transport_socket_or_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        calls.append("called")
        raise AssertionError("acceptance touched credentials, a provider, or a socket")

    class ForbiddenEnvironment(dict[str, str]):
        def __getitem__(self, key: str) -> str:
            return forbidden(key)

        def get(self, key: str, default: Any = None) -> Any:
            return forbidden(key, default)

    monkeypatch.setattr(os, "environ", ForbiddenEnvironment())
    monkeypatch.setattr(os, "getenv", forbidden)
    monkeypatch.setattr(openai, "OpenAI", forbidden)
    monkeypatch.setattr(httpx, "Client", forbidden)
    monkeypatch.setattr(httpx, "MockTransport", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(
        "operatebench.agents.openai_responses.OpenAIResponsesTransport", forbidden
    )
    monkeypatch.setattr(
        "operatebench.agents.evidence.EvidenceRecordingModelAgent", forbidden
    )
    monkeypatch.setattr("operatebench.agents.evidence.WireCaptureTransport", forbidden)
    _audit()
    assert calls == []


def _expected_wheel_package_files(root: Path = ROOT) -> dict[str, Path]:
    expected: dict[str, Path] = {}
    excluded_directories = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}

    def visit(directory: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                mode = path.lstat().st_mode
                assert not stat.S_ISLNK(mode), (
                    f"wheel source oracle encountered symlink: {path}"
                )
                if stat.S_ISDIR(mode):
                    if path.name not in excluded_directories:
                        visit(path)
                    continue
                assert stat.S_ISREG(mode), (
                    f"wheel source oracle encountered non-regular entry: {path}"
                )
                if path.suffix in {".pyc", ".pyo"}:
                    continue
                expected[path.relative_to(root / "src").as_posix()] = path

    for package in ("operatebench", "boundarybench"):
        package_root = root / "src" / package
        assert package_root.exists(), f"wheel package source is missing: {package_root}"
        mode = package_root.lstat().st_mode
        assert stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), (
            f"wheel package source must be a regular directory: {package_root}"
        )
        visit(package_root)
    return expected


def _write_test_distribution(directory: Path) -> None:
    _, sdist = _canonical_distribution(directory)
    replacement = sdist.with_name("replacement.tar.gz")
    with (
        tarfile.open(sdist, "r:gz") as source,
        tarfile.open(replacement, "w:gz") as archive,
    ):
        for member in source.getmembers():
            extracted = source.extractfile(member) if member.isfile() else None
            archive.addfile(member, extracted)
        for name, pin in PINS.items():
            raw = (FIXTURES / name).read_bytes()
            assert (len(raw), _sha(raw)) == pin
            member = tarfile.TarInfo(f"operatebench-0.1.0/tests/fixtures/b3/{name}")
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    replacement.replace(sdist)


def test_distribution_directory_unset_skips_with_build_job_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPERATEBENCH_DIST_DIR", raising=False)
    with pytest.raises(pytest.skip.Exception, match="dedicated CI build job"):
        _audit_distribution_from_environment()


def test_prebuilt_distribution_passes_exact_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "dist"
    _write_test_distribution(out)
    monkeypatch.setenv("OPERATEBENCH_DIST_DIR", str(out))

    _audit_distribution_from_environment()


def test_distribution_directory_from_environment_must_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing"
    monkeypatch.setenv("OPERATEBENCH_DIST_DIR", str(missing))
    with pytest.raises(AssertionError, match="existing directory"):
        _distribution_archives_from_environment()


@pytest.mark.parametrize("duplicate_suffix", [".whl", ".tar.gz"])
def test_distribution_directory_requires_exactly_one_archive_of_each_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, duplicate_suffix: str
) -> None:
    out = tmp_path / "dist"
    _write_test_distribution(out)
    (out / f"duplicate{duplicate_suffix}").write_bytes(b"wrong archive")
    monkeypatch.setenv("OPERATEBENCH_DIST_DIR", str(out))
    with pytest.raises(AssertionError, match="exactly one wheel and one sdist"):
        _distribution_archives_from_environment()


def test_distribution_directory_refuses_wrong_archive_kinds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "dist"
    out.mkdir()
    (out / "operatebench.zip").write_bytes(b"not a wheel")
    (out / "operatebench.tar").write_bytes(b"not an sdist")
    monkeypatch.setenv("OPERATEBENCH_DIST_DIR", str(out))
    with pytest.raises(AssertionError, match="exactly one wheel and one sdist"):
        _distribution_archives_from_environment()


def _rewrite_test_wheel(
    wheel: Path,
    *,
    remove: set[str] | None = None,
    alter: str | None = None,
    symlink: str | None = None,
) -> None:
    replacement = wheel.with_suffix(".replacement.whl")
    with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(replacement, "w") as target:
        for member in source.infolist():
            if remove is not None and member.filename in remove:
                continue
            raw = source.read(member)
            if member.filename == alter:
                raw += b"tampered"
            if member.filename == symlink:
                member = zipfile.ZipInfo(member.filename)
                member.create_system = 3
                member.external_attr = (stat.S_IFLNK | 0o777) << 16
                raw = b"outside"
            target.writestr(member, raw)
    replacement.replace(wheel)


def _append_test_sdist_member(
    sdist: Path, member: tarfile.TarInfo, raw: bytes = b""
) -> None:
    replacement = sdist.with_suffix(".replacement.tar.gz")
    with (
        tarfile.open(sdist, "r:gz") as source,
        tarfile.open(replacement, "w:gz") as target,
    ):
        for existing in source.getmembers():
            extracted = source.extractfile(existing) if existing.isfile() else None
            target.addfile(existing, extracted)
        member.size = len(raw)
        target.addfile(member, io.BytesIO(raw) if raw else None)
    replacement.replace(sdist)


@pytest.mark.parametrize(
    ("kind", "linkname"),
    [(tarfile.SYMTYPE, "../../../../outside"), (tarfile.LNKTYPE, "outside")],
)
def test_sdist_refuses_duplicate_fixture_link_replacement(
    tmp_path: Path, kind: bytes, linkname: str
) -> None:
    out = tmp_path / "dist"
    _write_test_distribution(out)
    wheel, sdist = next(out.glob("*.whl")), next(out.glob("*.tar.gz"))
    member = tarfile.TarInfo(f"operatebench-0.9.0/tests/fixtures/b3/{MODEL_NAME}")
    member.type = kind
    member.linkname = linkname
    _append_test_sdist_member(sdist, member)

    with pytest.raises(AssertionError):
        _audit_distribution(wheel, sdist)


def test_sdist_refuses_duplicate_regular_member(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    _write_test_distribution(out)
    wheel, sdist = next(out.glob("*.whl")), next(out.glob("*.tar.gz"))
    member = tarfile.TarInfo(f"operatebench-0.9.0/tests/fixtures/b3/{MODEL_NAME}")
    _append_test_sdist_member(sdist, member, (FIXTURES / MODEL_NAME).read_bytes())

    with pytest.raises(AssertionError):
        _audit_distribution(wheel, sdist)


@pytest.mark.parametrize(
    "name",
    [
        "/absolute/file",
        "operatebench-0.9.0/../outside",
        "operatebench-0.9.0\\outside",
        "operatebench-0.9.0//outside",
    ],
)
def test_sdist_refuses_unsafe_member_name(tmp_path: Path, name: str) -> None:
    out = tmp_path / "dist"
    _write_test_distribution(out)
    wheel, sdist = next(out.glob("*.whl")), next(out.glob("*.tar.gz"))
    _append_test_sdist_member(sdist, tarfile.TarInfo(name), b"unsafe")

    with pytest.raises(AssertionError):
        _audit_distribution(wheel, sdist)


def test_sdist_refuses_second_top_level_root(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    _write_test_distribution(out)
    wheel, sdist = next(out.glob("*.whl")), next(out.glob("*.tar.gz"))
    _append_test_sdist_member(sdist, tarfile.TarInfo("other-root/file"), b"unsafe")

    with pytest.raises(AssertionError):
        _audit_distribution(wheel, sdist)


def test_sdist_refuses_non_file_non_directory_member(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    _write_test_distribution(out)
    wheel, sdist = next(out.glob("*.whl")), next(out.glob("*.tar.gz"))
    member = tarfile.TarInfo("operatebench-0.9.0/pipe")
    member.type = tarfile.FIFOTYPE
    _append_test_sdist_member(sdist, member)

    with pytest.raises(AssertionError):
        _audit_distribution(wheel, sdist)


def test_wheel_refuses_omission_of_boundarybench_package(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    _write_test_distribution(out)
    wheel, sdist = next(out.glob("*.whl")), next(out.glob("*.tar.gz"))
    with zipfile.ZipFile(wheel) as archive:
        remove = {
            name for name in archive.namelist() if name.startswith("boundarybench/")
        }
    _rewrite_test_wheel(wheel, remove=remove)

    with pytest.raises(AssertionError):
        _audit_distribution(wheel, sdist)


@pytest.mark.parametrize(
    "name",
    [
        "operatebench/version.py",
        "operatebench/resources/identity/identity-component-registry-v1.json",
        "operatebench/py.typed",
    ],
)
def test_wheel_refuses_omission_of_any_operatebench_source_member(
    tmp_path: Path, name: str
) -> None:
    out = tmp_path / "dist"
    _write_test_distribution(out)
    wheel, sdist = next(out.glob("*.whl")), next(out.glob("*.tar.gz"))
    _rewrite_test_wheel(wheel, remove={name})

    with pytest.raises(AssertionError):
        _audit_distribution(wheel, sdist)


def test_wheel_refuses_changed_package_member_bytes(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    _write_test_distribution(out)
    wheel, sdist = next(out.glob("*.whl")), next(out.glob("*.tar.gz"))
    _rewrite_test_wheel(wheel, alter="operatebench/version.py")

    with pytest.raises(AssertionError):
        _audit_distribution(wheel, sdist)


def test_wheel_refuses_duplicate_member_name(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    _write_test_distribution(out)
    wheel, sdist = next(out.glob("*.whl")), next(out.glob("*.tar.gz"))
    with (
        pytest.warns(UserWarning, match="Duplicate name"),
        zipfile.ZipFile(wheel, "a") as archive,
    ):
        archive.writestr("operatebench/__init__.py", b"replacement")

    with pytest.raises(AssertionError):
        _audit_distribution(wheel, sdist)


def test_wheel_refuses_symlink_mode_package_member(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    _write_test_distribution(out)
    wheel, sdist = next(out.glob("*.whl")), next(out.glob("*.tar.gz"))
    _rewrite_test_wheel(wheel, symlink="operatebench/__init__.py")

    with pytest.raises(AssertionError):
        _audit_distribution(wheel, sdist)


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_wheel_source_oracle_refuses_non_regular_entries(
    tmp_path: Path, kind: str
) -> None:
    source_root = tmp_path / "src"
    for package in ("operatebench", "boundarybench"):
        package_root = source_root / package
        package_root.mkdir(parents=True)
        (package_root / "__init__.py").write_bytes(b"")
    hazard = source_root / "operatebench" / "hazard"
    if kind == "symlink":
        hazard.symlink_to(source_root / "operatebench" / "__init__.py")
    else:
        os.mkfifo(hazard)

    with pytest.raises(AssertionError):
        _expected_wheel_package_files(tmp_path)


def _distribution_archives_from_environment() -> tuple[Path, Path]:
    configured = os.getenv("OPERATEBENCH_DIST_DIR")
    if configured is None:
        pytest.skip(
            "OPERATEBENCH_DIST_DIR is unset; exact distribution audit runs in the "
            "dedicated CI build job"
        )
    assert configured is not None

    directory = Path(configured)
    assert directory.is_dir(), "OPERATEBENCH_DIST_DIR must name an existing directory"
    wheels = list(directory.glob("*.whl"))
    sdists = list(directory.glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1, (
        "OPERATEBENCH_DIST_DIR must contain exactly one wheel and one sdist; "
        f"found {len(wheels)} wheel(s) and {len(sdists)} sdist(s)"
    )
    return wheels[0], sdists[0]


def _audit_distribution(wheel: Path, sdist: Path) -> None:
    problems = release_checks.check_built_distributions(
        ROOT, distribution_dir=wheel.parent
    )
    assert problems == [], "generic distribution integrity failed:\n" + "\n".join(
        problems
    )
    with zipfile.ZipFile(wheel) as archive:
        assert not any("tests/fixtures/b3/" in name for name in archive.namelist())

    with tarfile.open(sdist, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        roots = {name.split("/", 1)[0] for name in members}
        assert len(roots) == 1
        root = next(iter(roots))
        fixture_prefix = f"{root}/tests/fixtures/b3/"
        shipped = {
            name.removeprefix(fixture_prefix)
            for name in members
            if name.startswith(fixture_prefix)
        }
        assert shipped == set(PINS), (
            "sdist must carry exactly the four pinned B3 fixtures"
        )
        for name, pin in PINS.items():
            fixture_path = f"{root}/tests/fixtures/b3/{name}"
            member = members[fixture_path]
            assert member.isfile()
            extracted = archive.extractfile(member)
            assert extracted is not None
            raw = extracted.read()
            assert (len(raw), _sha(raw)) == pin


def _audit_distribution_from_environment() -> None:
    wheel, sdist = _distribution_archives_from_environment()
    _audit_distribution(wheel, sdist)


def test_distribution_membership_and_sdist_bytes() -> None:
    _audit_distribution_from_environment()


def test_the_committed_bundle_is_exactly_what_this_build_regenerates() -> None:
    """Legacy test name retained; regeneration belongs to the exact old build.

    Current readers audit the frozen bytes independently. The subprocess runs
    the original 0.9.0 source, never 0.10.0 with a substituted version label.
    Missing local history is an explicit failure, not a skip or successful audit.
    """
    from tests.historical_runtime import run_historical_mock

    _audit()
    run_historical_mock("b3", FIXTURES)
