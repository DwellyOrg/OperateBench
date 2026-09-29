"""Synthetic external custody only. Never opens a configured credential store."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
from pathlib import Path

import httpx
import httpx2
import pytest

from tools import three_flow_live as live
from tools.three_flow_admission import (
    GUARANTEES,
    RESOURCE_POLICY,
    VARIABLES,
    digest,
    registry,
    runtime_identity,
)
from tools.three_flow_campaign import assignments
from tools.three_flow_http import ENDPOINTS
from tools.three_flow_mock import SDKMockTransport

DUMMY = "benign-dummy-fixture"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


class Network:
    def __init__(self, fault=None):
        self.calls = []
        self.fault = fault

    def http(self, slot):
        self.calls.append(slot["trial_id"])
        inner = SDKMockTransport(slot["provider"], slot["flow"])

        def response(request):
            assert request.extensions["timeout"] == dict.fromkeys(
                ("connect", "read", "write", "pool"), 17
            )
            if self.fault is None:
                return inner.handle_request(request)
            if self.fault == "interrupt":
                raise KeyboardInterrupt()
            if self.fault == "timeout":
                raise httpx.ReadTimeout("benign private error", request=request)
            result = inner.handle_request(request)
            body = json.loads(result.read())
            if self.fault == "cache":
                body["usage"].update(
                    cache_creation_input_tokens=5, cache_read_input_tokens=7
                )
            else:
                body.pop("usage", None)
            return httpx.Response(200, json=body)

        return httpx.MockTransport(response)

    def river(self, slot):
        from tools.three_flow_river_mock import RiverMockChannel

        self.calls.append(slot["trial_id"])
        return RiverMockChannel(slot["model"], slot["flow"])


def write(path, value):
    path.write_text(json.dumps(value))
    path.chmod(0o600)


def fixture(tmp_path, provider="openai"):
    custody = tmp_path / "custody"
    custody.mkdir(mode=0o700)
    credential = custody / "dummy.env"
    credential.write_text(VARIABLES[provider] + "=" + DUMMY + "\n")
    credential.chmod(0o600)
    slot = next(s for s in assignments("test-series") if s["provider"] == provider)
    p = {
        "endpoint": ENDPOINTS.get(provider, "api.river.ai:443"),
        "input_max": 262144,
        "output_max": 32768 if provider == "river" else 128000,
        "context_max": 1000000,
        "input_usd_per_million": "1",
        "output_usd_per_million": "2",
        "network_timeout_seconds": 120 if provider == "river" else 17,
        "rate_evidence_sha256": "a" * 64,
        "bounds_evidence_sha256": "b" * 64,
        "guarantees": dict.fromkeys(GUARANTEES, True),
        "pricing_policy": "conservative-all-tier-envelope-v1",
    }
    assets = None
    if provider == "river":
        from tools.three_flow_river_assets import catalog

        assets = {
            "root": os.environ["THREE_FLOW_RIVER_ASSETS"],
            "catalog_sha256": digest(catalog()),
        }
    a = {
        "schema": "three-flow-admission-v1",
        "approved": False,
        "mode": "synthetic",
        "source_sha": live.source_identity(),
        "runtime": runtime_identity(),
        "registry": registry(),
        "campaign_id": "test-series",
        "attempt_id": "first",
        "campaign_root": str(tmp_path / "campaign"),
        "custody_root": str(custody),
        "credential_file": str(credential),
        "cap_usd": "1000",
        "resource_policy": RESOURCE_POLICY,
        "selected": [slot["trial_id"]],
        "profiles": {slot["model"]: p},
        "assets": assets,
        "operation": "initial",
        "original_claim_sha256": None,
        "journal_sha256": None,
        "pending": [slot["trial_id"]],
        "corrections": {},
        "parent_gates": True,
        "no_intervening_spend": True,
    }
    path = custody / "synthetic.json"
    write(path, a)
    return path, a, slot


def cli(path, a, network):
    return live.main(
        ["run", "--expected-source", a["source_sha"], "--admission", str(path)],
        network=network,
    )


@pytest.mark.parametrize("provider", ["openai", "anthropic", "mistral", "river"])
def test_actual_consuming_cli_sdk_full_trial_replay(tmp_path, provider, capsys):
    if provider == "river" and "THREE_FLOW_RIVER_ASSETS" not in os.environ:
        pytest.skip("native lane runs separately under qualified existing interpreter")
    path, a, slot = fixture(tmp_path, provider)
    network = Network()
    assert cli(path, a, network) == 0
    root = Path(a["campaign_root"])
    record = json.loads((root / slot["trial_id"] / "record.json").read_text())
    assert record["evidence_origin"] == "SDK_MOCK_NOT_LLM"
    assert record["provider"]["settings"]["admission"] is True
    contract = next(r for r in registry() if r["model"] == slot["model"])
    assert record["provider"]["request_mapping"] == contract["mapping"]
    for key, value in contract["request_settings"].items():
        assert record["provider"]["settings"][key] == value
    report = json.loads((root / "scorecard.json").read_text())
    assert report["rows"][0]["classification"] == "scored"
    assert report["rows"][0]["eligible_for_live_results"] is False
    events = [
        json.loads(line)
        for line in (root / "campaign-budget.jsonl").read_text().splitlines()
    ]
    assert (
        sum(e["kind"] == "raw_rpc" for e in events) == report["rows"][0]["network_rpcs"]
    )
    from tools.three_flow_runtime import Trial, replay_trial

    trial = Trial(**json.loads((root / (slot["trial_id"] + ".binding.json")).read_text()))
    replay_trial(trial, record, expected_record_digest=record["record_digest"])
    before = list(network.calls)
    assert cli(path, a, network) == 2
    assert network.calls == before
    output = capsys.readouterr()
    assert DUMMY not in output.out + output.err
    for file in root.rglob("*"):
        if file.is_file():
            assert DUMMY.encode() not in file.read_bytes()


@pytest.mark.parametrize(
    "mutation", ["price", "coverage", "source", "endpoint", "settings", "runtime"]
)
def test_admission_refuses_before_credential_pin(tmp_path, monkeypatch, mutation):
    path, a, slot = fixture(tmp_path)
    p = a["profiles"][slot["model"]]
    if mutation == "price":
        p["input_usd_per_million"] = "NaN"
    if mutation == "coverage":
        p["guarantees"]["all_input_cache_write_read_tiers_covered"] = False
    if mutation == "source":
        a["source_sha"] = "0" * 40
    if mutation == "endpoint":
        p["endpoint"] = "https://invalid.example/"
    if mutation == "settings":
        p["output_max"] = 10
    if mutation == "runtime":
        a["runtime"]["python"] = "changed"
    write(path, a)
    original = live.private_fd

    def pin(value, **kwargs):
        assert value != a["credential_file"]
        return original(value, **kwargs)

    monkeypatch.setattr(live, "private_fd", pin)
    assert cli(path, a, Network()) == 2
    assert not (path.parent / "SERIES-CONSUMED").exists()
    assert not Path(a["campaign_root"]).exists()


def test_preflight_has_no_custody_or_writes(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("custody forbidden during preflight")

    monkeypatch.setattr(live, "private_fd", forbidden)
    monkeypatch.setattr(live, "publish", forbidden)
    assert live.main(["preflight", "--admission", str(tmp_path / "unopened")]) == 0
    value = json.loads(capsys.readouterr().out)
    assert len(value["slots"]) == 616 and len(value["registry"]) == 14
    assert value["admitted"] is False
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fail", [False, True])
def test_claim_file_and_directory_durable_before_read(tmp_path, monkeypatch, fail):
    path, a, _ = fixture(tmp_path)
    fsync, read = os.fsync, live.read_pinned
    completed = []

    def sync(fd):
        kind = "dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
        if fail and kind == "dir":
            raise OSError("injected durability failure")
        fsync(fd)
        completed.append(kind)

    def guarded(fd, cap):
        if cap == 65536:
            assert completed[:4] == ["file", "dir", "file", "dir"]
            assert (path.parent / "SERIES-CONSUMED").exists()
        return read(fd, cap)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(live, "read_pinned", guarded)
    assert cli(path, a, Network()) == (2 if fail else 0)
    assert (path.parent / "SERIES-CONSUMED").exists()
    if fail:
        assert not Path(a["campaign_root"]).exists()


@pytest.mark.parametrize("kind", ["mode", "hardlink", "symlink", "oversize", "nonnormal"])
def test_private_custody(tmp_path, kind):
    p = tmp_path / "dummy"
    p.write_text("dummy")
    p.chmod(0o600)
    value = str(p)
    if kind == "mode":
        p.chmod(0o644)
    if kind == "hardlink":
        os.link(p, tmp_path / "link")
    if kind == "symlink":
        (tmp_path / "link").symlink_to(p)
        value = str(tmp_path / "link")
    if kind == "oversize":
        p.write_bytes(b"a" * 65537)
    if kind == "nonnormal":
        value = str(tmp_path) + "/./dummy"
    with pytest.raises(ValueError):
        live.private_fd(value, cap=65536)


@pytest.mark.parametrize(
    "raw", [b"X=$(echo)\n", b"X=\n", b"X=a\nX=b\n", b"X='a b'\n", b"Y=ok\n"]
)
def test_strict_env_rejects(raw):
    with pytest.raises(ValueError):
        live.parse_env(raw, {"X"})


def continuation(path, a, *, operation="resume"):
    a["operation"] = operation
    a["attempt_id"] = "second"
    a["pending"] = []
    a["original_claim_sha256"] = hashlib.sha256(
        (path.parent / "SERIES-CONSUMED").read_bytes()
    ).hexdigest()
    a["journal_sha256"] = hashlib.sha256(
        (Path(a["campaign_root"]) / "campaign-budget.jsonl").read_bytes()
    ).hexdigest()
    write(path, a)


def test_resume_same_journal_no_redispatch(tmp_path):
    path, a, slot = fixture(tmp_path)
    network = Network()
    assert cli(path, a, network) == 0
    root = Path(a["campaign_root"])
    retained = (root / (slot["trial_id"] + ".closure.json")).read_bytes()
    journal = (root / "campaign-budget.jsonl").read_bytes()
    continuation(path, a)
    assert cli(path, a, network) == 0
    assert len(network.calls) == 1
    assert (root / "campaign-budget.jsonl").read_bytes() == journal
    assert (root / (slot["trial_id"] + ".closure.json")).read_bytes() == retained
    assert (path.parent / "ATTEMPT-second-CONSUMED").exists()


def test_changed_journal_refuses_before_secret(tmp_path, monkeypatch):
    path, a, _ = fixture(tmp_path)
    assert cli(path, a, Network()) == 0
    continuation(path, a)
    a["journal_sha256"] = "0" * 64
    write(path, a)
    original = live.private_fd

    def pin(value, **kwargs):
        assert value != a["credential_file"]
        return original(value, **kwargs)

    monkeypatch.setattr(live, "private_fd", pin)
    assert cli(path, a, Network()) == 2
    assert not (path.parent / "ATTEMPT-second-CONSUMED").exists()


@pytest.mark.parametrize("fault", ["missing_usage", "timeout", "interrupt"])
def test_unknown_interrupted_resume_and_confirmed_correction(tmp_path, fault):
    path, a, slot = fixture(tmp_path)
    broken = Network(fault)
    if fault == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            cli(path, a, broken)
    else:
        assert cli(path, a, broken) == 2
    root = Path(a["campaign_root"])
    journal = root / "campaign-budget.jsonl"
    before = journal.read_bytes()
    events = [json.loads(line) for line in before.splitlines()]
    assert sum(e["kind"] == "reserve" for e in events) == 1
    assert sum(e["kind"] == "unknown" for e in events) == 1
    continuation(path, a)
    network = Network()
    assert cli(path, a, network) == 2
    assert not network.calls
    original = root / (slot["trial_id"] + ".closure.json")
    retained = original.read_bytes()
    assert float(json.loads(retained)["unknown_liability_usd"]) > 0
    a["operation"] = "correction"
    a["attempt_id"] = "confirmed-fix"
    a["corrections"] = {
        slot["trial_id"]: {"reason": "runtime_fix", "evidence_sha256": "c" * 64}
    }
    a["journal_sha256"] = hashlib.sha256(journal.read_bytes()).hexdigest()
    write(path, a)
    assert cli(path, a, network) == 0
    assert network.calls == [slot["trial_id"] + "-fix-confirmed-fix"]
    assert original.read_bytes() == retained
    assert journal.read_bytes().startswith(before)
    all_events = [json.loads(line) for line in journal.read_bytes().splitlines()]
    assert sum(e["kind"] == "genesis" for e in all_events) == 1
    assert sum(e["kind"] == "unknown" for e in all_events) == 1
    assert (root / (slot["trial_id"] + "-fix-confirmed-fix.closure.json")).exists()


def test_concurrent_claim_has_one_owner(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    path, a, _ = fixture(tmp_path)
    network = Network()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: cli(path, a, network), range(2)))
    assert sorted(results) == [0, 2]
    assert len(network.calls) == 1


def test_constructor_failure_releases_lane_and_http_progress(tmp_path):
    from threading import Event, Lock

    from tools.three_flow_campaign import run_campaign

    ready = Event()
    mutex = Lock()
    constructed = []
    active = 0
    peak = 0
    roster = assignments("resource-test")
    selected = [s for s in roster if s["provider"] == "river"][:3]
    selected += [next(s for s in roster if s["provider"] == "openai")]

    def factory(slot, budget):
        nonlocal active, peak
        if slot["provider"] == "openai":
            ready.set()
        with mutex:
            active += 1
            peak = max(peak, active)
            constructed.append(slot["trial_id"])
        try:
            assert ready.wait(5), "native barrier starved HTTP lane"
            raise ValueError("synthetic constructor failure")
        finally:
            with mutex:
                active -= 1

    rows = run_campaign(
        tmp_path / "resource",
        campaign_id="resource-test",
        slots=selected,
        transport_factory=factory,
    )
    assert len(rows) == 4 and len(constructed) == 4
    assert all(r["classification"] == "aborted" for r in rows)
    assert peak <= 2 and active == 0


def test_source_rejected_before_repository_import_in_fresh_process(tmp_path):
    import subprocess

    result = subprocess.run(
        [
            os.sys.executable,
            "-I",
            "-B",
            str(live.ROOT / "tools/three_flow_live.py"),
            "run",
            "--expected-source",
            "0" * 40,
            "--admission",
            str(tmp_path / "absent"),
        ],
        env={"PATH": "/usr/bin:/bin"},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "refused_or_incomplete" in result.stderr
    assert not list(tmp_path.iterdir())


def test_cache_input_categories_settle_under_envelope(tmp_path):
    path, a, slot = fixture(tmp_path, "anthropic")
    assert cli(path, a, Network("cache")) == 0
    record = json.loads(
        (Path(a["campaign_root"]) / slot["trial_id"] / "record.json").read_text()
    )
    for turn in record["provider_turns"]:
        wire = json.loads(turn["wire"]["response_utf8"])
        assert turn["input_tokens"] == wire["usage"]["input_tokens"] + 12
        assert wire["usage"]["cache_creation_input_tokens"] == 5


def test_bad_credentials_consumed_series_can_resume_only_pending(tmp_path):
    path, a, _ = fixture(tmp_path)
    credential = Path(a["credential_file"])
    credential.write_text("invalid grammar\n")
    assert cli(path, a, Network()) == 2
    root = Path(a["campaign_root"])
    assert (root / "assignment.json").exists()
    assert (path.parent / "SERIES-CONSUMED").exists()
    credential.write_text("OPENAI_API_KEY=" + DUMMY + "\n")
    continuation(path, a)
    a["pending"] = a["selected"]
    write(path, a)
    network = Network()
    assert cli(path, a, network) == 0
    assert len(network.calls) == 1
    assert (
        len(
            [
                json.loads(line)
                for line in (root / "campaign-budget.jsonl").read_text().splitlines()
                if json.loads(line)["kind"] == "genesis"
            ]
        )
        == 1
    )


@pytest.mark.parametrize("provider", ["openai", "anthropic", "mistral"])
@pytest.mark.parametrize("body_kind", ["literal", "escaped", "nested", "nonsecret"])
def test_cli_error_projection_before_publication(
    tmp_path, monkeypatch, provider, body_kind, capsys, caplog
):
    import base64
    from decimal import Decimal

    from tools.three_flow_http import HTTPCampaignTransport

    path, admission, slot = fixture(tmp_path, provider)
    status = 503 if body_kind == "nonsecret" else 401
    message = "Service unavailable" if body_kind == "nonsecret" else DUMMY
    body = {"error": {"type": "authentication_error", "message": message}}
    if body_kind == "nested":
        body = {"error": {"details": [{"content": [{"text": DUMMY}]}]}}
    raw = json.dumps(body).encode()
    escaped = "".join(f"\\u{ord(c):04x}" for c in DUMMY).encode()
    if body_kind == "escaped":
        raw = raw.replace(DUMMY.encode(), escaped)
    calls, sdk_errors = [], []
    original_dispatch = HTTPCampaignTransport._dispatch

    def dispatch(self, payload):
        try:
            return original_dispatch(self, payload)
        except Exception as exc:
            # Inspect the actual SDK exception, not a synthetic parser. The
            # SDK must still receive the untouched response in memory.
            response = getattr(exc, "response", getattr(exc, "raw_response", None))
            response_type = httpx2.Response if provider == "anthropic" else httpx.Response
            assert isinstance(response, response_type)
            assert response.content == raw
            assert response.status_code == status
            sdk_errors.append(type(exc).__module__)
            raise

    monkeypatch.setattr(HTTPCampaignTransport, "_dispatch", dispatch)

    class ErrorNetwork:
        def http(self, slot):
            def response(request):
                calls.append(str(request.url))
                return httpx.Response(
                    status,
                    content=raw,
                    headers={
                        "content-type": "application/json",
                        "x-secret-canary": "header-canary-" + DUMMY,
                    },
                )

            return httpx.MockTransport(response)

    assert cli(path, admission, ErrorNetwork()) == 2
    assert calls == [ENDPOINTS[provider]]
    assert len(sdk_errors) == 1 and sdk_errors[0].startswith(provider)
    root = Path(admission["campaign_root"])
    rows = [
        json.loads(line)
        for line in (root / slot["trial_id"] / "partial.ndjson").read_bytes().splitlines()
    ]
    responses = [row for row in rows if row["kind"] == "wire_response"]
    assert len(responses) == 1
    witness = responses[0]
    assert witness["status_code"] == status
    projection = witness["response_projection"]
    assert projection["representation"] == "sanitized_http_error_v2"
    assert projection["http_status"] == status
    assert projection["reason"] == "unknown"
    assert projection["body_retained"] is False
    assert projection["automatic_resubmission"] is False
    assert "response_utf8" not in witness and "response_sha256" not in witness
    requests = [row for row in rows if row["kind"] == "wire_request"]
    assert len(requests) == 1
    assert witness["request_sha256"] == requests[0]["request_sha256"]
    assert rows[-1]["classification"] == "excluded"
    closure = json.loads((root / (slot["trial_id"] + ".closure.json")).read_bytes())
    assert closure["classification"] == "excluded"
    turn = next(row["telemetry"] for row in rows if row["kind"] == "provider_turn")
    assert closure["fault"] == turn["classification"]
    assert closure["fault"] != "provider_transport"
    assert closure["raw_rpc_attempts"] == 1
    assert Decimal(closure["unknown_liability_usd"]) > 0
    events = [
        json.loads(line)
        for line in (root / "campaign-budget.jsonl").read_bytes().splitlines()
    ]
    assert sum(e["kind"] == "unknown" for e in events) == 1
    assert sum(e["kind"] == "reserve" for e in events) == 1
    assert not (root / slot["trial_id"] / "record.json").exists()
    forbidden = [
        DUMMY.encode(),
        escaped,
        base64.b64encode(raw),
        raw.hex().encode(),
        hashlib.sha256(raw).hexdigest().encode(),
        b"header-canary-",
        b"x-secret-canary",
        b"authentication_error",
        b"Service unavailable",
    ]
    # All outputs, including custody receipts, closure, partials and logs. The
    # sole excluded file is the deliberately supplied dummy input credential.
    for file in tmp_path.rglob("*"):
        if file.is_file() and file != Path(admission["credential_file"]):
            data = file.read_bytes()
            assert all(value not in data for value in forbidden), file
    output = capsys.readouterr()
    assert all(
        value not in (output.out + output.err + caplog.text).encode()
        for value in forbidden
    )


def test_cli_rejects_nonnormal_admission_path(tmp_path):
    path, a, _ = fixture(tmp_path)
    assert (
        live.main(
            [
                "run",
                "--expected-source",
                a["source_sha"],
                "--admission",
                str(path.parent) + "/./" + path.name,
            ],
            network=Network(),
        )
        == 2
    )
    assert not (path.parent / "SERIES-CONSUMED").exists()
