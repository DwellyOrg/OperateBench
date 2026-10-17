"""Gemini one-shot admission and mock-only launch qualification."""

from __future__ import annotations

import copy
import json
import socket
from pathlib import Path

import httpx
import pytest

from tools import three_flow_live as live
from tools.three_flow_admission import (
    RESOURCE_POLICY,
    digest,
    registry,
    runtime_identity,
    validate,
)
from tools.three_flow_budget import CampaignBudget
from tools.three_flow_campaign import assignments, run_campaign
from tools.three_flow_gemini import (
    LIMIT_PROFILE,
    MODELS,
    ROSTER_PROFILE,
    GeminiMockTransport,
    offline_limit_policy,
)
from tools.three_flow_live_factory import live_factory


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def admission(tmp_path: Path, *, synthetic: bool = True) -> dict:
    sha = "a" * 40
    model = "gemini-4-argon"
    slot = next(
        s for s in assignments("gemini-test", ROSTER_PROFILE) if s["model"] == model
    )
    policy = {**offline_limit_policy(), "source_sha": sha}
    return {
        "schema": "three-flow-admission-v1",
        "approved": not synthetic,
        "mode": "synthetic" if synthetic else "paid",
        "source_sha": sha,
        "runtime": runtime_identity(),
        "registry": registry(roster_profile=ROSTER_PROFILE, mode_profile="gemini-low-v1"),
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
        "roster_profile": ROSTER_PROFILE,
        "mode_profile": "gemini-low-v1",
        "limit_policy": policy,
    }


@pytest.mark.parametrize("synthetic", (True, False))
def test_exact_gemini_initial_admission_is_source_bound(tmp_path, synthetic):
    value = admission(tmp_path, synthetic=synthetic)
    assert value["limit_policy"]["mode"] == LIMIT_PROFILE
    assert validate(value, value["source_sha"], synthetic=synthetic) == value
    changed = copy.deepcopy(value)
    changed["source_sha"] = "b" * 40
    with pytest.raises(ValueError):
        validate(changed, value["source_sha"], synthetic=synthetic)


@pytest.mark.parametrize("synthetic", (True, False))
def test_gemini_admission_refuses_boolean_integer_profile_alias(tmp_path, synthetic):
    value = admission(tmp_path, synthetic=synthetic)
    value["profiles"]["gemini-4-argon"]["automatic_resubmission"] = 0
    with pytest.raises(ValueError, match="Gemini model/output/price/profile"):
        validate(value, value["source_sha"], synthetic=synthetic)


def test_gemini_live_factory_uses_native_mock_and_guard(tmp_path):
    value = admission(tmp_path)
    root = tmp_path / "campaign"
    root.mkdir(mode=0o700)
    budget = CampaignBudget(
        root,
        campaign_id=value["campaign_id"],
        cap=None,
        limit_policy=value["limit_policy"],
        roster_profile=ROSTER_PROFILE,
        create=True,
    )
    slot = next(
        s
        for s in assignments(value["campaign_id"], ROSTER_PROFILE)
        if s["trial_id"] == value["selected"][0]
    )

    class Network:
        def http(self, s):
            assert s == slot
            return httpx.MockTransport(GeminiMockTransport(s["model"], s["flow"]).handle)

    transport = None
    try:
        transport = live_factory(
            value, {"GEMINI_API_KEY": "synthetic-key"}, network=Network()
        )(slot, budget)
        assert transport.provider == "gemini"
        assert transport.model == slot["model"]
        assert transport.settings["evidence_origin"] == "SDK_MOCK_NOT_LLM"
        assert transport.settings["admission"] is False
        assert "admission_sha256" not in transport.settings
        assert "admission_profile_sha256" not in transport.settings
        assert transport.guard.budget is budget
    finally:
        if transport is not None:
            transport.close()
        budget.close()


def test_synthetic_gemini_refuses_null_mock_before_direct_transport(
    tmp_path, monkeypatch
):
    value = admission(tmp_path)
    root = tmp_path / "campaign"
    root.mkdir(mode=0o700)
    budget = CampaignBudget(
        root,
        campaign_id=value["campaign_id"],
        cap=None,
        limit_policy=value["limit_policy"],
        roster_profile=ROSTER_PROFILE,
        create=True,
    )
    slot = next(
        s
        for s in assignments(value["campaign_id"], ROSTER_PROFILE)
        if s["trial_id"] == value["selected"][0]
    )

    class Network:
        def http(self, s):
            assert s == slot
            return None

    def forbidden(**kwargs):
        raise AssertionError("direct transport constructed from synthetic admission")

    monkeypatch.setattr(httpx, "HTTPTransport", forbidden)
    try:
        with pytest.raises(ValueError, match="synthetic Gemini requires MockTransport"):
            live_factory(value, {"GEMINI_API_KEY": "synthetic-key"}, network=Network())(
                slot, budget
            )
        assert budget.records == {}
    finally:
        budget.close()


def test_gemini_admitted_mock_campaign_reuses_existing_runner(tmp_path):
    value = admission(tmp_path)
    root = tmp_path / "campaign"
    root.mkdir(mode=0o700)
    slot = next(
        s
        for s in assignments(value["campaign_id"], ROSTER_PROFILE)
        if s["trial_id"] == value["selected"][0]
    )
    budget = CampaignBudget(
        root,
        campaign_id=value["campaign_id"],
        cap=None,
        limit_policy=value["limit_policy"],
        roster_profile=ROSTER_PROFILE,
        create=True,
    )

    class Network:
        def http(self, s):
            return httpx.MockTransport(GeminiMockTransport(s["model"], s["flow"]).handle)

    rows = run_campaign(
        root,
        campaign_id=value["campaign_id"],
        slots=[slot],
        roster_profile=ROSTER_PROFILE,
        campaign_budget=budget,
        transport_factory=live_factory(
            value, {"GEMINI_API_KEY": "synthetic-key"}, network=Network()
        ),
        evidence_origin="SDK_MOCK_NOT_LLM",
        mode="synthetic",
    )
    assert len(rows) == 1 and rows[0]["classification"] == "scored"
    assert rows[0]["evidence_origin"] == "SDK_MOCK_NOT_LLM"
    assert rows[0]["eligible_for_live_results"] is False


def test_gemini_provider_origin_requires_explicit_factory_and_replays(
    tmp_path, monkeypatch
):
    from operatebench.agents.transport import content_digest
    from tools.three_flow_runtime import Trial, replay_trial

    value = admission(tmp_path, synthetic=False)
    root = tmp_path / "campaign"
    root.mkdir(mode=0o700)
    slot = next(
        s
        for s in assignments(value["campaign_id"], ROSTER_PROFILE)
        if s["trial_id"] == value["selected"][0]
    )
    budget = CampaignBudget(
        root,
        campaign_id=value["campaign_id"],
        cap=None,
        limit_policy=value["limit_policy"],
        roster_profile=ROSTER_PROFILE,
        create=True,
    )
    calls = []

    def direct(**kwargs):
        assert kwargs == {"trust_env": False, "retries": 0}
        mock = GeminiMockTransport(slot["model"], slot["flow"])

        def response(request):
            calls.append(request)
            assert request.headers["x-goog-api-key"] == "synthetic-key"
            return mock.handle(request)

        return httpx.MockTransport(response)

    monkeypatch.setattr(httpx, "HTTPTransport", direct)
    rows = run_campaign(
        root,
        campaign_id=value["campaign_id"],
        slots=[slot],
        roster_profile=ROSTER_PROFILE,
        campaign_budget=budget,
        transport_factory=live_factory(value, {"GEMINI_API_KEY": "synthetic-key"}),
        evidence_origin="PROVIDER_CANDIDATE",
        mode="paid",
    )
    assert len(calls) == rows[0]["network_rpcs"] > 0
    assert rows[0]["classification"] == "scored"
    assert rows[0]["evidence_origin"] == "PROVIDER_CANDIDATE"
    trial = Trial(**json.loads((root / (slot["trial_id"] + ".binding.json")).read_text()))
    record = json.loads((root / slot["trial_id"] / "record.json").read_text())
    expected = MODELS[slot["model"]].live_settings(65536)
    expected.update(
        admission=True,
        admission_provenance="source-bound-admission-reference-v1",
        admission_sha256=digest(value),
        admission_profile_sha256=digest(value["profiles"][slot["model"]]),
        limit_policy=value["limit_policy"],
    )
    assert trial.provider_binding["settings"] == expected
    assert record["trial"]["provider_binding"]["settings"] == expected
    assert record["provider"]["settings"] == expected
    assert expected["admission"] is True
    assert "synthetic-key" not in json.dumps(record)
    assert "synthetic-key" not in json.dumps(trial.provider_binding)
    replay_trial(trial, record, expected_record_digest=record["record_digest"])
    for field, replacement in (
        ("admission_sha256", "b" * 64),
        ("admission_profile_sha256", "b" * 64),
        ("limit_policy", {**value["limit_policy"], "source_sha": "b" * 40}),
        ("arbitrary_setting", True),
    ):
        changed = copy.deepcopy(record)
        changed["provider"]["settings"][field] = replacement
        changed["record_digest"] = content_digest(
            {k: v for k, v in changed.items() if k != "record_digest"}
        )
        with pytest.raises(ValueError, match="provider/settings binding differs"):
            replay_trial(trial, changed, expected_record_digest=changed["record_digest"])
    from tools.three_flow_gemini import validate_replay

    for field, replacement in (
        ("admission_profile_sha256", "b" * 64),
        ("admission_provenance", "unversioned"),
        ("arbitrary_setting", True),
        ("limit_policy", {**value["limit_policy"], "owner_accepted": False}),
    ):
        changed = copy.deepcopy(record)
        changed["provider"]["settings"][field] = replacement
        with pytest.raises(ValueError):
            validate_replay(changed)


@pytest.mark.parametrize("settings", ["invalid", [], None])
def test_gemini_replay_rejects_wrong_settings_root(settings):
    from tools.three_flow_gemini import validate_replay

    with pytest.raises(ValueError, match="Gemini profile provenance differs"):
        validate_replay(
            {
                "trial": {"model": "gemini-4-argon", "max_output_tokens": 65536},
                "provider": {"settings": settings},
            }
        )


def test_gemini_paid_factory_detaches_admitted_policy(tmp_path, monkeypatch):
    value = admission(tmp_path, synthetic=False)
    root = tmp_path / "campaign"
    root.mkdir(mode=0o700)
    slot = next(
        s
        for s in assignments(value["campaign_id"], ROSTER_PROFILE)
        if s["trial_id"] == value["selected"][0]
    )
    budget = CampaignBudget(
        root,
        campaign_id=value["campaign_id"],
        cap=None,
        limit_policy=value["limit_policy"],
        roster_profile=ROSTER_PROFILE,
        create=True,
    )

    def direct(**kwargs):
        assert kwargs == {"trust_env": False, "retries": 0}
        return httpx.MockTransport(
            GeminiMockTransport(slot["model"], slot["flow"]).handle
        )

    monkeypatch.setattr(httpx, "HTTPTransport", direct)
    transport = None
    try:
        transport = live_factory(value, {"GEMINI_API_KEY": "synthetic-key"})(slot, budget)
        recorded = copy.deepcopy(transport.settings)
        assert recorded["admission"] is True
        assert recorded["admission_sha256"] == digest(value)
        assert recorded["limit_policy"] == value["limit_policy"]
        value["limit_policy"]["source_sha"] = "b" * 40
        assert digest(value) != recorded["admission_sha256"]
        assert transport.settings == recorded
        assert transport.settings["limit_policy"] is not value["limit_policy"]
    finally:
        if transport is not None:
            transport.close()
        budget.close()


def test_gemini_cli_preflight_is_non_authorizing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(live, "source_identity", lambda expected=None: "a" * 40)
    assert live.main(["preflight", "--roster-profile", ROSTER_PROFILE]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["admitted"] is False
    assert value["credentials_accessed"] is False
    assert len(value["slots"]) == 88
    assert not any("offline-only" in b for b in value["blockers"])
    assert any("source-bound" in b for b in value["blockers"])
    assert list(tmp_path.iterdir()) == []


def test_gemini_consuming_cli_claims_once_before_mock_run(tmp_path, monkeypatch, capsys):
    value = admission(tmp_path)
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


@pytest.mark.parametrize(
    "mutation", ("source", "rate", "mode", "limit", "selected", "runtime")
)
def test_gemini_cli_refuses_mutations_before_credential_or_claim(
    tmp_path, monkeypatch, mutation
):
    value = admission(tmp_path)
    custody = Path(value["custody_root"])
    custody.mkdir(mode=0o700)
    credential = Path(value["credential_file"])
    credential.write_text("GEMINI_API_KEY=synthetic-key\n")
    credential.chmod(0o600)
    if mutation == "source":
        value["source_sha"] = "b" * 40
    elif mutation == "rate":
        value["profiles"]["gemini-4-argon"]["pricing_basis"]["input_usd_per_mtok"] = "0"
    elif mutation == "mode":
        value["mode_profile"] = "off-or-minimum-v1"
    elif mutation == "limit":
        value["limit_policy"]["source_sha"] = "b" * 40
    elif mutation == "selected":
        value["selected"] *= 2
    else:
        value["runtime"]["python"] = "mismatched"
    path = custody / "synthetic.json"
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    monkeypatch.setattr(live, "source_identity", lambda expected=None: "a" * 40)
    previous = live.private_fd

    def pinned(name, **kwargs):
        assert name != str(credential), "credential pin before admission"
        return previous(name, **kwargs)

    monkeypatch.setattr(live, "private_fd", pinned)
    assert (
        live.main(
            ["run", "--expected-source", "a" * 40, "--admission", str(path)],
            network=object(),
        )
        == 2
    )
    assert not (custody / "SERIES-CONSUMED").exists()
    assert not Path(value["campaign_root"]).exists()
