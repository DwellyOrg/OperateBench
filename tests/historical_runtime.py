"""Test-only exact-source, offline mock generation; never real historical runs.

No downloads or fallback to the current engine. See docs/HISTORICAL_RUNTIME.md.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_SOURCE = "226eee9574f300a0ef8bf921ac5bdcabeb2635c06f4f2b951606571c5ac13d38"[
    :40
]
HISTORICAL_SOURCE_SHA256 = (
    "00bef1dddfc85d376812aa75d5f1a9ffb18259c2336dc35330404bb115f1f47f"
)
HISTORICAL_MANIFEST_SHA256 = (
    "226eee9574f300a0ef8bf921ac5bdcabeb2635c06f4f2b951606571c5ac13d38"
)


class HistoricalRuntimeUnavailable(RuntimeError):
    """The exact local source is absent; byte audits are not regeneration."""


def run_historical_mock(
    kind: str, directory: Path, payload: dict[str, Any] | None = None
) -> str:
    if kind not in {"alpha", "b3", "alpha-audit", "alpha-identity"}:
        raise ValueError("unknown historical mock fixture")
    if hashlib.sha256(HISTORICAL_SOURCE.encode("utf-8")).hexdigest() != (
        HISTORICAL_SOURCE_SHA256
    ):
        raise HistoricalRuntimeUnavailable(
            "historical_runtime_unavailable: provision the exact source input "
            "documented in docs/HISTORICAL_RUNTIME.md"
        )
    # Extraction filters were backported to maintained Python 3.11 releases.
    # Older interpreters must refuse, never fall back to unrestricted extraction.
    if "filter" not in inspect.signature(
        tarfile.TarFile.extractall
    ).parameters or not callable(getattr(tarfile, "data_filter", None)):
        raise HistoricalRuntimeUnavailable(
            "historical_runtime_unavailable: tar_data_filter_required; "
            "use a maintained Python with tarfile data-filter support"
        )
    # An allowlist rather than deleting known credential names from the parent.
    env = {"PATH": os.defpath, "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1"}
    fixture = ROOT / "tests/fixtures/historical_engine_0_9"
    manifest_path = fixture / "manifest.json"
    if (
        not manifest_path.is_file()
        or hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        != HISTORICAL_MANIFEST_SHA256
    ):
        raise HistoricalRuntimeUnavailable(
            "historical_runtime_unavailable: manifest pin mismatch"
        )
    pins = json.loads(manifest_path.read_text())["files"]
    source_root = fixture / "source"
    paths = list(source_root.rglob("*"))
    if any(p.is_symlink() for p in paths) or {
        p.relative_to(source_root).as_posix() for p in paths if p.is_file()
    } != set(pins):
        raise HistoricalRuntimeUnavailable(
            "historical_runtime_unavailable: source closure mismatch"
        )
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as bundle:
        for relative, expected in pins.items():
            path = source_root / relative
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise HistoricalRuntimeUnavailable(
                    "historical_runtime_unavailable: source content mismatch"
                )
            entry = tarfile.TarInfo(relative)
            entry.size = len(data)
            entry.mode = 0o644
            bundle.addfile(entry, io.BytesIO(data))
    archive.seek(0)
    with tempfile.TemporaryDirectory(prefix="operatebench-historical-") as raw:
        source = Path(raw)
        with tarfile.open(fileobj=archive) as bundle:
            bundle.extractall(source, filter="data")
        # The current Alpha validator must remain exactly the archived one:
        # otherwise auditing there would not cover the current assertion bodies.
        assert (source / "tools/matched_arm_alpha.py").read_bytes() == (
            ROOT / "tools/matched_arm_alpha.py"
        ).read_bytes()
        env["PYTHONPATH"] = f"{source / 'src'}:{source}"
        env["HOME"] = raw
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                _OFFLINE_SCRIPT,
                kind,
                str(directory.resolve()),
                HISTORICAL_SOURCE,
            ],
            cwd=source,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
            input=json.dumps(payload),
        )
        if result.returncode:
            raise AssertionError(
                f"historical mock {kind} failed ({result.returncode}):\n"
                f"{result.stdout}\n{result.stderr}"
            )
        return result.stdout


def validate_historical_mock_result(
    raw: dict[str, Any], *, directory: Path | None, verify_files: bool = True
) -> dict[str, Any]:
    """Current static validation plus exact-source causal audit of test mocks."""
    from tools import matched_arm_alpha as alpha

    value = alpha.validate_result(raw, directory=None, verify_files=False)
    if verify_files:
        if directory is None:
            raise alpha.AlphaRefusal("evidence_directory_required")
        response = json.loads(run_historical_mock("alpha-audit", directory, raw))
        if response["refusal"] is not None:
            raise alpha.AlphaRefusal(response["refusal"])
    return value


_OFFLINE_SCRIPT = r"""
import json
import socket
import sys
from pathlib import Path

def no_network(*args, **kwargs):
    raise AssertionError("historical mock attempted network access")

socket.socket.connect = no_network
socket.socket.connect_ex = no_network
socket.create_connection = no_network
socket.getaddrinfo = no_network

from operatebench.version import OPERATEBENCH_VERSION
assert OPERATEBENCH_VERSION == "0.9.0"
kind, target, revision = sys.argv[1:]
directory = Path(target)
if kind == "b3":
    from tools.regenerate_b3_evidence import check
    assert check(directory) == []
elif kind == "alpha-identity":
    from tools import matched_arm_alpha as alpha
    assert alpha.LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION == (
        "lifecycle_openai_responses_model_request_v7"
    )
    print(json.dumps(alpha._expected_provider_identity()))
elif kind == "alpha-audit":
    from tools import matched_arm_alpha as alpha
    try:
        alpha.validate_result(json.load(sys.stdin), directory=directory)
    except alpha.AlphaRefusal as exc:
        print(json.dumps({"refusal": exc.code}))
    else:
        print(json.dumps({"refusal": None}))
else:
    from tools import matched_arm_alpha as alpha
    from operatebench.agents.evidence import WireCaptureTransport
    from tools.preflight_lifecycle_provider_evidence import ScriptedProvider
    alpha.check_pinned_build()
    alpha.compile_pinned_arms()
    with alpha.reserve_output_names(directory):
        wire = WireCaptureTransport()
        provider = ScriptedProvider(max_calls=alpha.HARD_CALL_CAP)
        wire.attach(alpha.httpx.MockTransport(provider))
        client = alpha.openai.OpenAI(
            api_key=alpha.PLACEHOLDER_KEY,
            base_url=alpha.OPENAI_BASE_URL,
            max_retries=0,
            http_client=alpha.httpx.Client(
                transport=wire, timeout=alpha.TURN_DEADLINE_SECONDS
            ),
        )
        try:
            result = alpha._execute(
                directory=directory, client=client, wire=wire,
                mode=alpha.MODE_OFFLINE, external_source_revision=revision,
            )
            assert provider.calls == result["accounting"]["provider_calls"]
        finally:
            client.close()
"""
