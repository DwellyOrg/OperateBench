"""Offline Carbon consumers; no account evidence or provider access."""

import copy
import datetime
import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from tools import three_flow_live as live
from tools.three_flow_admission import (
    RESOURCE_POLICY,
    registry,
    runtime_identity,
    validate,
)
from tools.three_flow_campaign import CONFIGURATIONS, assignments, preflight
from tools.three_flow_gemini import (
    CARBON_MODELS,
    MODELS,
    GeminiMockTransport,
    offline_limit_policy,
    reported_tariff,
)
from tools.three_flow_profiles import (
    CARBON_ROSTER_PROFILE,
    CARBON_UNCAPPED_PROFILE,
    GEMINI_ROSTER_PROFILE,
    selected_roster,
)


def set_carbon_clock(monkeypatch, *, expired=False):
    class FixtureDate(datetime.date):
        @classmethod
        def today(cls):
            return cls(2026, 11, 30)

    class FixtureDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz is datetime.UTC
            return (
                cls(2026, 12, 1, tzinfo=tz)
                if expired
                else cls(2026, 11, 30, 23, 59, 59, tzinfo=tz)
            )

    monkeypatch.setattr(datetime, "date", FixtureDate)
    monkeypatch.setattr(datetime, "datetime", FixtureDatetime)


@pytest.fixture(autouse=True)
def carbon_fixture_clock(monkeypatch):
    """Pin UTC positives; local November 30 also exposes the UTC cutoff mismatch."""
    set_carbon_clock(monkeypatch)


def admission(tmp_path: Path, *, synthetic: bool = True) -> dict:
    sha = "a" * 40
    model = "gemini-4-argon"
    slot = next(
        s
        for s in assignments("gemini-test", GEMINI_ROSTER_PROFILE)
        if s["model"] == model
    )
    policy = {**offline_limit_policy(), "source_sha": sha}
    return {
        "schema": "three-flow-admission-v1",
        "approved": not synthetic,
        "mode": "synthetic" if synthetic else "paid",
        "source_sha": sha,
        "runtime": runtime_identity(),
        "registry": registry(
            roster_profile=GEMINI_ROSTER_PROFILE, mode_profile="gemini-low-v1"
        ),
        "campaign_id": "gemini-test",
        "attempt_id": "first",
        "campaign_root": str(tmp_path / "campaign"),
        "custody_root": str(tmp_path / "custody"),
        "credential_file": str(tmp_path / "custody" / "dummy.env"),
        "cap_usd": None,
        "resource_policy": RESOURCE_POLICY,
        "selected": [slot["trial_id"]],
        "profiles": {
            model: MODELS[model].settings(65536)
            if synthetic
            else MODELS[model].live_settings(65536)
        },
        "assets": None,
        "operation": "initial",
        "original_claim_sha256": None,
        "journal_sha256": None,
        "pending": [slot["trial_id"]],
        "corrections": {},
        "parent_gates": True,
        "no_intervening_spend": True,
        "roster_profile": GEMINI_ROSTER_PROFILE,
        "mode_profile": "gemini-low-v1",
        "limit_policy": policy,
    }


def carbon_admission(tmp_path):
    value = admission(tmp_path)
    profile = CARBON_MODELS["gemini-4-carbon"]
    value.update(
        roster_profile=CARBON_ROSTER_PROFILE,
        registry=registry(
            roster_profile=CARBON_ROSTER_PROFILE, mode_profile="gemini-low-v1"
        ),
        profiles={profile.model: profile.settings(profile.output)},
    )
    value["limit_policy"]["mode"] = CARBON_UNCAPPED_PROFILE
    return value


def test_carbon_finite_roster_and_legacy_defaults():
    slots = assignments("carbon", CARBON_ROSTER_PROFILE)
    assert len(slots) == len(CONFIGURATIONS) == 44
    assert {(s["flow"], s["scenario"]) for s in slots} == set(CONFIGURATIONS)
    assert selected_roster(CARBON_ROSTER_PROFILE) == (("gemini", "gemini-4-carbon"),)
    assert len(assignments("old", GEMINI_ROSTER_PROFILE)) == 88
    assert tuple(MODELS) == ("gemini-4-argon", "gemini-3.8-flash")
    assert all(p.output == 65536 for p in MODELS.values())
    assert preflight(CARBON_ROSTER_PROFILE)["initial_assignments"] == 44


def test_carbon_exact_settings_and_usage_tariff():
    p = CARBON_MODELS["gemini-4-carbon"]
    s = p.settings(32768)
    assert s["generationConfig"] == {
        "maxOutputTokens": 32768,
        "thinkingConfig": {"thinkingLevel": "low"},
    }
    assert s["context_request_bound"] == 1048576
    assert s["pricing_basis"]["cached_input_usd_per_mtok"] == "0.1"
    assert reported_tariff(
        {
            "promptTokenCount": 100,
            "cachedContentTokenCount": 40,
            "toolUsePromptTokenCount": 3,
            "candidatesTokenCount": 5,
            "thoughtsTokenCount": 7,
        }
    ) == Decimal("0.00025")
    with pytest.raises(ValueError):
        p.settings(65536)


def test_carbon_source_bound_separate_consent(tmp_path):
    value = carbon_admission(tmp_path)
    assert validate(value, value["source_sha"], synthetic=True) == value
    for key, replacement in [
        ("mode", "three-flow-optional-limits-v1"),
        ("source_sha", "b" * 40),
    ]:
        bad = copy.deepcopy(value)
        bad["limit_policy"][key] = replacement
        with pytest.raises(ValueError):
            validate(bad, value["source_sha"], synthetic=True)


@pytest.mark.parametrize("historical", [False, True])
@pytest.mark.parametrize("direct", [False, True])
def test_carbon_factory_native_wire(tmp_path, monkeypatch, historical, direct):
    from tools.three_flow_budget import CampaignBudget
    from tools.three_flow_gemini import GeminiMockTransport
    from tools.three_flow_live_factory import live_factory

    if historical:
        from tools.three_flow_gemini import GeminiProfile

        current_settings = GeminiProfile.settings

        def original_settings(self, maximum):
            settings = current_settings(self, maximum)
            if self.model == "gemini-4-carbon":
                settings["network_timeout_seconds"]["write"] = None
            return settings

        monkeypatch.setattr(GeminiProfile, "settings", original_settings)
    value = carbon_admission(tmp_path)
    if direct:
        profile = CARBON_MODELS["gemini-4-carbon"]
        value["profiles"][profile.model] = profile.live_settings(profile.output)
    root = tmp_path / "budget"
    root.mkdir(mode=0o700)
    slot = assignments("carbon", CARBON_ROSTER_PROFILE)[0]

    import json

    import httpx

    from tools.three_flow_runtime import Trial, provider_identity, replay_trial, run_trial

    mock = GeminiMockTransport(slot["model"], slot["flow"])

    class Network:
        def http(self, slot):
            def handle(request):
                assert request.extensions["timeout"] == {
                    "connect": 10,
                    "pool": 10,
                    "read": None,
                    "write": 10,
                }
                body = json.loads(mock.handle(request).content)
                body["usageMetadata"] = {
                    "promptTokenCount": 100,
                    "cachedContentTokenCount": 40,
                    "toolUsePromptTokenCount": 3,
                    "candidatesTokenCount": 5,
                    "thoughtsTokenCount": 7,
                    "totalTokenCount": 115,
                }
                return httpx.Response(200, json=body)

            return httpx.MockTransport(handle)

    budget = CampaignBudget(
        root,
        campaign_id="carbon",
        cap=None,
        limit_policy=value["limit_policy"],
        roster_profile=CARBON_ROSTER_PROFILE,
        create=True,
    )
    try:
        if direct:
            monkeypatch.setattr(httpx, "HTTPTransport", lambda **kw: Network().http(slot))
        transport = live_factory(
            value,
            {"GEMINI_API_KEY": "synthetic"},
            network=None if direct else Network(),
        )(slot, budget)
        if not direct:
            set_carbon_clock(monkeypatch, expired=True)
        assert transport.max_output_tokens == 32768
        trial = Trial(
            "carbon",
            slot["trial_id"],
            slot["flow"],
            slot["scenario"],
            slot["model"],
            32768,
            provider_identity(transport),
        )
        record = run_trial(trial, transport, output=tmp_path / "trial")
        assert record["evaluation"]["reliable"]
        assert all(
            t["reported_tariff_estimate_usd"] == "0.00025"
            for t in record["provider_turns"]
        )
        assert (
            mock.calls[1]["contents"][1]["parts"][0]["thoughtSignature"]
            == "offline-signature"
        )
        if historical:
            monkeypatch.setattr(GeminiProfile, "settings", current_settings)
        set_carbon_clock(monkeypatch, expired=True)
        retained = copy.deepcopy(record)
        assert (
            replay_trial(trial, record, expected_record_digest=record["record_digest"])[
                "provider_calls"
            ]
            == 0
        )
        assert record == retained
        if historical and direct:
            from tools.three_flow_gemini import validate_replay

            bad = copy.deepcopy(record)
            bad["provider"]["settings"]["admission_profile_sha256"] = "b" * 64
            with pytest.raises(ValueError, match="provenance"):
                validate_replay(bad)
        transport.close()
    finally:
        budget.close()


def test_carbon_cli_preflight_is_non_authorizing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(live, "source_identity", lambda expected=None: "a" * 40)
    assert live.main(["preflight", "--roster-profile", CARBON_ROSTER_PROFILE]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["admitted"] is False
    assert value["credentials_accessed"] is False
    assert len(value["slots"]) == 44
    assert not any("offline-only" in b for b in value["blockers"])
    assert any("source-bound" in b for b in value["blockers"])
    assert list(tmp_path.iterdir()) == []


def test_carbon_consuming_cli_claims_once_before_mock_run(tmp_path, monkeypatch, capsys):
    value = carbon_admission(tmp_path)
    custody = Path(value["custody_root"])
    custody.mkdir(mode=0o700)
    key = "synthetic-key-not-real"
    credential = Path(value["credential_file"])
    credential.write_text("GEMINI_API_KEY=" + key + "\n")
    credential.chmod(0o600)
    path = custody / "synthetic.json"
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    monkeypatch.setattr(
        live, "source_identity", lambda expected=None: value["source_sha"]
    )
    calls = []

    class Network:
        def http(self, slot):
            calls.append(slot["trial_id"])
            return httpx.MockTransport(
                GeminiMockTransport(slot["model"], slot["flow"]).handle
            )

    original_read = live.read_pinned

    def ordered_read(fd, cap):
        import os

        if os.fstat(fd).st_ino == credential.stat().st_ino:
            assert (custody / "SERIES-CONSUMED").is_file()
        return original_read(fd, cap)

    monkeypatch.setattr(live, "read_pinned", ordered_read)
    network = Network()
    command = ["run", "--expected-source", value["source_sha"], "--admission", str(path)]
    assert live.main(command, network=network) == 0
    root = Path(value["campaign_root"])
    assert len(calls) == 1
    assert (custody / "SERIES-CONSUMED").is_file()
    assert (custody / "RECEIPT-first").is_file()
    assert (
        json.loads((root / "scorecard.json").read_text())["rows"][0]["classification"]
        == "scored"
    )
    assert live.main(command, network=network) == 2
    assert len(calls) == 1
    assert key not in capsys.readouterr().out
    for p in (*root.rglob("*"), *custody.rglob("*")):
        if p.is_file():
            assert key.encode() not in p.read_bytes() or p == credential


def test_carbon_complete_context_intersection(monkeypatch):
    from tools import three_flow_gemini as native
    from tools.three_flow_gemini import GeminiCampaignTransport

    mapping = object.__new__(GeminiCampaignTransport)
    mapping.profile = CARBON_MODELS["gemini-4-carbon"]
    mapping.max_output_tokens = 32768
    monkeypatch.setattr(
        GeminiCampaignTransport, "payload", lambda self, request: {"generationConfig": {}}
    )
    monkeypatch.setattr(native, "request_input_token_bound", lambda *args: 1048576 - 123)
    payload, bound, maximum = mapping.bounded_payload(None)
    assert maximum == payload["generationConfig"]["maxOutputTokens"] == 123
    assert bound + maximum == 1048576


def test_carbon_strict_native_response_and_quality_classification():
    from operatebench.agents.model import ModelAgent, ModelReturnedNoToolCall
    from operatebench.agents.transport import ForbiddenTransport
    from operatebench.providers.faults import AdapterProviderError
    from tools.three_flow_gemini import parse_response

    body = {
        "modelVersion": "gemini-4-carbon",
        "usageMetadata": {
            "promptTokenCount": 100,
            "cachedContentTokenCount": 40,
            "toolUsePromptTokenCount": 3,
            "candidatesTokenCount": 5,
            "thoughtsTokenCount": 7,
            "totalTokenCount": 115,
        },
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {"role": "model", "parts": [{"text": "no call"}]},
            }
        ],
    }
    response, counts, _ = parse_response(json.dumps(body), "gemini-4-carbon")
    assert counts["cachedContentTokenCount"] == 40
    assert isinstance(
        ModelAgent(ForbiddenTransport(), model="gemini-4-carbon")._parse(response),
        ModelReturnedNoToolCall,
    )
    body["candidates"][0]["content"]["parts"] = [
        {"functionCall": {"name": "done", "args": {}}}
    ]
    with pytest.raises(AdapterProviderError):
        parse_response(json.dumps(body), "gemini-4-carbon")


def test_carbon_rejects_expired_introductory_tariff(tmp_path, monkeypatch):
    value = carbon_admission(tmp_path)
    set_carbon_clock(monkeypatch, expired=True)
    assert datetime.date.today().isoformat() == "2026-11-30"
    with pytest.raises(ValueError, match="tariff expired"):
        validate(value, value["source_sha"], synthetic=True)


def test_carbon_admitted_live_send_crosses_utc_cutoff(tmp_path, monkeypatch):
    from operatebench.agents.transport import ProviderFailure
    from operatebench.providers.faults import AdapterProviderError
    from tools.three_flow_budget import CampaignBudget
    from tools.three_flow_live_factory import live_factory
    from tools.three_flow_runtime import Trial, provider_identity, run_trial

    value = carbon_admission(tmp_path)
    validate(value, value["source_sha"], synthetic=True)
    profile = CARBON_MODELS["gemini-4-carbon"]
    value["profiles"][profile.model] = profile.live_settings(profile.output)
    slot = assignments("carbon", CARBON_ROSTER_PROFILE)[0]
    root = tmp_path / "budget"
    root.mkdir(mode=0o700)
    calls = []
    mock = GeminiMockTransport(slot["model"], slot["flow"])

    def handle(request):
        calls.append(request)
        return mock.handle(request)

    monkeypatch.setattr(httpx, "HTTPTransport", lambda **kw: httpx.MockTransport(handle))
    budget = CampaignBudget(
        root,
        campaign_id="carbon",
        cap=None,
        limit_policy=value["limit_policy"],
        roster_profile=CARBON_ROSTER_PROFILE,
        create=True,
    )
    try:
        transport = live_factory(value, {"GEMINI_API_KEY": "synthetic"})(slot, budget)
        original_send = transport.send
        attempted = []

        def crossing_send(request):
            # A valid prepared request reaches send only after admission/creation.
            set_carbon_clock(monkeypatch, expired=True)
            authorize = transport.guard.authorize
            reservations = []

            def counted_authorize(**kwargs):
                reservations.append(kwargs)
                return authorize(**kwargs)

            monkeypatch.setattr(transport.guard, "authorize", counted_authorize)
            with pytest.raises(AdapterProviderError, match="tariff expired") as error:
                original_send(request)
            assert error.value.fault == "provider_request_rejected"
            assert transport.stopped
            assert calls == reservations == []
            with pytest.raises(AdapterProviderError, match="stopped"):
                original_send(request)
            assert calls == reservations == []
            attempted.append(True)
            raise error.value

        monkeypatch.setattr(transport, "send", crossing_send)
        trial = Trial(
            "carbon",
            slot["trial_id"],
            slot["flow"],
            slot["scenario"],
            slot["model"],
            32768,
            provider_identity(transport),
        )
        with pytest.raises(ProviderFailure):
            run_trial(trial, transport, output=tmp_path / "trial")
        assert attempted == [True]
        transport.close()
    finally:
        budget.close()


def test_carbon_live_admission_rejects_original_timeout_metadata(tmp_path):
    value = carbon_admission(tmp_path)
    value["profiles"]["gemini-4-carbon"]["network_timeout_seconds"]["write"] = None
    with pytest.raises(ValueError):
        validate(value, value["source_sha"], synthetic=True)


def test_carbon_cutoff_uses_profile_validity(monkeypatch):
    from dataclasses import replace

    from tools.three_flow_gemini import carbon_tariff_current

    profile = CARBON_MODELS["gemini-4-carbon"]
    assert carbon_tariff_current(profile)
    assert not carbon_tariff_current(replace(profile, price_valid_through="2026-11-29"))
    set_carbon_clock(monkeypatch, expired=True)
    assert not carbon_tariff_current(profile)
