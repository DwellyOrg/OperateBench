"""Kimi Pro mock SDK wire and native reasoning continuation."""

import json
import socket
from decimal import Decimal

import httpx
import pytest

from operatebench.agents.pricing import LifecyclePricingPolicy
from operatebench.agents.transport import ProviderFailure
from tests.test_selected_six_http import SelectedMock
from tools.three_flow_budget import CampaignBudget
from tools.three_flow_http import HTTPCampaignTransport
from tools.three_flow_live_factory import AdmittedGuard
from tools.three_flow_profiles import LATE_NOVEMBER_PROFILE
from tools.three_flow_runtime import Trial, provider_identity, replay_trial, run_trial


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def test_kimi_pro_preserves_reasoning_in_sdk_history(tmp_path):
    model = "moonshotai/Kimi-K3.1-Pro"
    tmp_path.chmod(0o700)
    budget = CampaignBudget(
        tmp_path,
        campaign_id="offline",
        cap=Decimal("1000"),
        roster_profile=LATE_NOVEMBER_PROFILE,
        create=True,
    )
    p = {
        "endpoint": "https://api.river.ai/v1/chat/completions",
        "input_max": None,
        "output_max": 1048575,
        "context_max": 1048576,
        "owner_output_cap_tokens": 30000,
        "bounds_policy": {"mode": "test"},
    }
    guard = AdmittedGuard(
        profile=p,
        budget=budget,
        cell="trial",
        max_output_tokens=30000,
        policy=LifecyclePricingPolicy("synthetic", Decimal("2"), Decimal("10")),
    )
    mock = SelectedMock("river_http", model, "commerce")
    t = HTTPCampaignTransport(
        provider="river_http",
        model=model,
        api_key="offline-key",
        inner=mock,
        guard=guard,
        max_output_tokens=30000,
        context_tokens=1048576,
        mode_profile="off-or-minimum-v1",
        roster_profile=LATE_NOVEMBER_PROFILE,
    )
    t.settings.update(bounds_policy=p["bounds_policy"], context_request_bound=1048576)
    trial = Trial(
        "offline", "trial", "commerce", "UK_NORMAL", model, 30000, provider_identity(t)
    )
    try:
        record = run_trial(trial, t, output=tmp_path / "trial")
        assert record["evaluation"]["reliable"]
        assert len(mock.calls) > 1
        assert t.settings["reasoning_effort"] == "none"
        assert t.settings["reasoning_mode"] == "OFF"
        assert all(
            c["reasoning_effort"] == "none" and c["max_tokens"] <= 30000
            for c in mock.calls
        )
        assert mock.calls[1]["messages"][-2]["reasoning_content"] == "synthetic reasoning"
        assert mock.calls[1]["messages"][-1]["role"] == "tool"
        assert (
            replay_trial(trial, record, expected_record_digest=record["record_digest"])[
                "provider_calls"
            ]
            == 0
        )
    finally:
        t.close()
        budget.close()


class OutputUsageMock(SelectedMock):
    def __init__(self, model, output):
        super().__init__("river_http", model, "commerce")
        self.output = output

    def handle(self, request):
        response = super().handle(request)
        body = json.loads(response.content)
        output = self.output if len(self.calls) == 1 else 7
        body["usage"]["completion_tokens"] = output
        body["usage"]["total_tokens"] = 11 + output
        return httpx.Response(200, json=body)


@pytest.mark.parametrize("reported", [30000, 30001, 30003])
def test_kimi_sdk_output_ceiling_and_restart(tmp_path, reported):
    model = "moonshotai/Kimi-K3.1-Pro"
    tmp_path.chmod(0o700)
    kwargs = {
        "campaign_id": "offline-ceiling",
        "cap": Decimal("1000"),
        "roster_profile": LATE_NOVEMBER_PROFILE,
    }
    budget = CampaignBudget(tmp_path, create=True, **kwargs)
    profile = {
        "endpoint": "https://api.river.ai/v1/chat/completions",
        "input_max": None,
        "output_max": 1048575,
        "context_max": 1048576,
        "owner_output_cap_tokens": 30000,
        "bounds_policy": {"mode": "test"},
    }
    guard = AdmittedGuard(
        profile=profile,
        budget=budget,
        cell="trial",
        max_output_tokens=30000,
        policy=LifecyclePricingPolicy("synthetic", Decimal("2"), Decimal("10")),
    )
    mock = OutputUsageMock(model, reported)
    transport = HTTPCampaignTransport(
        provider="river_http",
        model=model,
        api_key="offline-key",
        inner=mock,
        guard=guard,
        max_output_tokens=30000,
        context_tokens=1048576,
        mode_profile="off-or-minimum-v1",
        roster_profile=LATE_NOVEMBER_PROFILE,
    )
    transport.settings.update(
        bounds_policy=profile["bounds_policy"], context_request_bound=1048576
    )
    trial = Trial(
        "offline",
        "trial",
        "commerce",
        "UK_NORMAL",
        model,
        30000,
        provider_identity(transport),
    )
    try:
        if reported == 30000:
            record = run_trial(trial, transport, output=tmp_path / "trial")
            assert record["evaluation"]["reliable"]
            assert (
                replay_trial(
                    trial, record, expected_record_digest=record["record_digest"]
                )["provider_calls"]
                == 0
            )
            assert budget.totals()["known"] > 0
        else:
            with pytest.raises(ProviderFailure) as failure:
                run_trial(trial, transport, output=tmp_path / "trial")
            assert failure.value.fault == "provider_budget"
            assert len(mock.calls) == 1
            rows = [
                json.loads(line)
                for line in (tmp_path / "trial" / "partial.ndjson")
                .read_text()
                .splitlines()
            ]
            assert not any(row["kind"] == "decision" for row in rows)
            assert not (tmp_path / "trial" / "record.json").exists()
            assert rows[-1]["classification"] == "excluded"
            assert rows[-1]["exclusion_code"] == "provider_budget"
            wires = [row for row in rows if row["kind"] == "wire_response"]
            assert len(wires) == 1
            assert (
                json.loads(wires[0]["response_utf8"])["usage"]["completion_tokens"]
                == reported
            )
            assert budget.totals()["unknown"] > 0
            assert budget.totals()["known"] == 0
        assert all(call["max_tokens"] <= 30000 for call in mock.calls)
        assert mock.calls[0]["max_tokens"] == 30000
    finally:
        transport.close()
        budget.close()
    recovered = CampaignBudget(tmp_path, **kwargs)
    try:
        if reported > 30000:
            assert recovered.totals()["unknown"] > 0
            assert recovered.totals()["known"] == 0
            assert len(recovered.records) == 1
            assert next(iter(recovered.records.values()))["state"] == "unknown"
        else:
            assert recovered.totals()["known"] > 0
    finally:
        recovered.close()


def test_kimi_factory_paced_wire_equals_previous_body(tmp_path, monkeypatch):
    from operatebench.core import instance
    from tools import three_flow_http as http
    from tools.three_flow_campaign import assignments
    from tools.three_flow_errors import ProviderPacer
    from tools.three_flow_live_factory import live_factory
    from tools.three_flow_profiles import registry

    monkeypatch.setattr(instance.secrets, "token_hex", lambda n: "a" * (2 * n))
    model = "moonshotai/Kimi-K3.1-Pro"
    slot = next(
        s
        for s in assignments("pacing", LATE_NOVEMBER_PROFILE)
        if s["model"] == model
        and s["flow"] == "commerce"
        and s["scenario"] == "UK_NORMAL"
    )
    profile = {
        "endpoint": "https://api.river.ai/v1/chat/completions",
        "input_max": None,
        "output_max": 1048575,
        "context_max": 1048576,
        "owner_output_cap_tokens": 30000,
        "input_usd_per_million": "2",
        "output_usd_per_million": "10",
        "network_timeout_seconds": None,
        "bounds_policy": {"mode": "test"},
    }

    class Captured(SelectedMock):
        def __init__(self):
            self.raw = []
            super().__init__("river_http", model, "commerce")

        def handle(self, request):
            self.raw.append(request.content)
            return super().handle(request)

    class Network:
        def __init__(self):
            self.mock = Captured()

        def http(self, _slot):
            return self.mock

    now = [0.0]
    waits = []

    def sleep(delay):
        waits.append(delay)
        now[0] += delay

    pacer = ProviderPacer(clock=lambda: now[0], sleep=sleep)
    monkeypatch.setattr(http, "RIVER_HTTP_PACER", pacer)
    admission = {
        "roster_profile": LATE_NOVEMBER_PROFILE,
        "operation": "initial",
        "campaign_id": "pacing",
        "selected": [slot["trial_id"]],
        "cap_usd": "1000",
        "profiles": {model: profile},
    }
    results = []
    for paced in (False, True):
        root = tmp_path / str(paced)
        root.mkdir(mode=0o700)
        budget = CampaignBudget(
            root,
            campaign_id="pacing",
            cap=Decimal("1000"),
            roster_profile=LATE_NOVEMBER_PROFILE,
            create=True,
        )
        network = Network()
        if paced:
            transport = live_factory(
                admission, {"RIVER_API_KEY": "offline-key"}, network=network
            )(slot, budget)
            assert transport.wire.pacer is pacer
            assert transport.settings["transport_policy"] == "paced-safe-errors-v1"
            row = next(
                r
                for r in registry(
                    roster_profile=LATE_NOVEMBER_PROFILE, mode_profile="late-november-v1"
                )
                if r["model"] == model
            )
            assert row["mapping"] == transport.request_mapping
            assert row["request_settings"]["automatic_resubmission"] is False
        else:
            guard = AdmittedGuard(
                profile=profile,
                budget=budget,
                cell=slot["trial_id"],
                max_output_tokens=30000,
                policy=LifecyclePricingPolicy("synthetic", Decimal("2"), Decimal("10")),
            )
            transport = HTTPCampaignTransport(
                provider="river_http",
                model=model,
                api_key="offline-key",
                inner=network.mock,
                guard=guard,
                max_output_tokens=30000,
                context_tokens=1048576,
                mode_profile="off-or-minimum-v1",
                roster_profile=LATE_NOVEMBER_PROFILE,
            )
        trial = Trial(
            "pacing",
            slot["trial_id"],
            "commerce",
            "UK_NORMAL",
            model,
            30000,
            provider_identity(transport),
        )
        try:
            record = run_trial(trial, transport, output=root / "trial")
            assert record["evaluation"]["reliable"]
            assert len(network.mock.raw) > 1
            assert (
                network.mock.calls[1]["messages"][-2]["reasoning_content"]
                == "synthetic reasoning"
            )
            assert all(call["max_tokens"] == 30000 for call in network.mock.calls)
            results.append(network.mock.raw)
        finally:
            transport.close()
            budget.close()
    assert results[0] == results[1]
    assert waits and all(wait == 1 for wait in waits)


def test_kimi_factory_shared_pacer_retry_after_no_resubmission(tmp_path, monkeypatch):
    import threading

    from tools import three_flow_http as http
    from tools.three_flow_campaign import assignments
    from tools.three_flow_errors import ProviderPacer
    from tools.three_flow_live_factory import live_factory

    model = "moonshotai/Kimi-K3.1-Pro"
    slots = [
        s
        for s in assignments("concurrent", LATE_NOVEMBER_PROFILE)
        if s["model"] == model and s["flow"] == "commerce"
    ][:2]
    profile = {
        "endpoint": "https://api.river.ai/v1/chat/completions",
        "input_max": None,
        "output_max": 1048575,
        "context_max": 1048576,
        "owner_output_cap_tokens": 30000,
        "input_usd_per_million": "2",
        "output_usd_per_million": "10",
        "network_timeout_seconds": None,
        "bounds_policy": {"mode": "test"},
    }
    admission = {
        "roster_profile": LATE_NOVEMBER_PROFILE,
        "operation": "initial",
        "campaign_id": "concurrent",
        "selected": [s["trial_id"] for s in slots],
        "cap_usd": "1000",
        "profiles": {model: profile},
    }
    now = [0.0]
    waits = []
    calls = []
    first_entered = threading.Event()
    second_started = threading.Event()

    def sleep(delay):
        waits.append(delay)
        now[0] += delay

    pacer = ProviderPacer(clock=lambda: now[0], sleep=sleep)
    monkeypatch.setattr(http, "RIVER_HTTP_PACER", pacer)

    class Network:
        def http(self, slot):
            mock = SelectedMock("river_http", model, slot["flow"])

            def handle(request):
                calls.append((slot["trial_id"], now[0]))
                if slot == slots[0]:
                    first_entered.set()
                    assert second_started.wait(5)
                    return httpx.Response(
                        429,
                        json={"error": {"message": "synthetic"}},
                        headers={"Retry-After": "2"},
                    )
                return mock.handle(request)

            return httpx.MockTransport(handle)

    tmp_path.chmod(0o700)
    budget = CampaignBudget(
        tmp_path,
        campaign_id="concurrent",
        cap=Decimal("1000"),
        roster_profile=LATE_NOVEMBER_PROFILE,
        create=True,
    )
    factory = live_factory(admission, {"RIVER_API_KEY": "offline-key"}, network=Network())
    transports = [factory(s, budget) for s in slots]
    assert all(t.wire.pacer is pacer for t in transports)
    outcomes = []

    def run(index):
        if index:
            assert first_entered.wait(5)
            second_started.set()
        s = slots[index]
        t = transports[index]
        trial = Trial(
            "concurrent",
            s["trial_id"],
            s["flow"],
            s["scenario"],
            model,
            30000,
            provider_identity(t),
        )
        try:
            run_trial(trial, t, output=tmp_path / ("trial-" + str(index)))
            outcomes.append("completed")
        except ProviderFailure as exc:
            outcomes.append(exc.fault)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
            assert not thread.is_alive()
        assert sorted(outcomes) == ["completed", "provider_rate_limited"]
        assert calls[:2] == [(slots[0]["trial_id"], 0), (slots[1]["trial_id"], 2)]
        assert sum(ident == slots[0]["trial_id"] for ident, _ in calls) == 1
        assert waits[0] == 2 and all(wait >= 1 for wait in waits)
        capture = transports[0].wire.captures[0]
        assert capture["status_code"] == 429 and "response_utf8" not in capture
        projection = capture["response_projection"]
        assert projection["reason"] == "unknown"
        assert projection["retry_after_seconds"] == 2
        assert projection["body_retained"] is False
        assert projection["automatic_resubmission"] is False
        assert budget.totals()["unknown"] > 0
    finally:
        for t in transports:
            t.close()
        budget.close()


def test_kimi_registry_historical_and_unaffected_parity():
    import copy

    from tools import three_flow_profiles as profiles
    from tools.three_flow_admission import digest, validate_historical_identity

    # Digests pin the pre-repair public registry without depending on Git HEAD.
    baseline = {
        "LATEST_ROSTER_PROFILE": (
            "80acca4b4d4ad23e0d8eeb203303ea3ad0d25ae2a84daa8312935d6e12c69f50"
        ),
        "SELECTED_SIX_PROFILE": (
            "2466588d6ce4e5220e77a29c081b7f9b005e66debd91ccc538671dec20c20488"
        ),
        "NOVEMBER_PROFILE": (
            "cf1367db3ac45622cd4bf87734337f1f0859a899a468f41ac9e374198a45e470"
        ),
        "FABLE52_PROFILE": (
            "827d5f5e7091e3cde90916f6fe3f9e55f0e76ce3e91e9c72c8482bb89b1d1e59"
        ),
        "OPENAI62_PROFILE": (
            "a064fe4759442104bc155dc6eaa13f116e276ac46e44be6ea7f2bdd348620aff"
        ),
        "META_ROSTER_PROFILE": (
            "22db7f61bba1cca956204b23e1ea8aab3c51e4b4c2ad35581ff341b53a764c23"
        ),
        "GEMINI_ROSTER_PROFILE": (
            "d86413e02221684647d4334da92005af550aad814d974406536ca7f0fd1e118d"
        ),
    }
    assert (
        digest(profiles.registry())
        == "cf829b743c356893e6e166e8754897b35e4d4dc6d4d80319f4393cfb25609c9c"
    )
    for name, expected in baseline.items():
        mode = "gemini-low-v1" if name == "GEMINI_ROSTER_PROFILE" else "off-or-minimum-v1"
        assert (
            digest(
                profiles.registry(
                    roster_profile=getattr(profiles, name), mode_profile=mode
                )
            )
            == expected
        )
    rows = profiles.registry(
        roster_profile=LATE_NOVEMBER_PROFILE, mode_profile="late-november-v1"
    )
    assert (
        digest(rows[:4])
        == "89eb3097379b409ffc2db3d639f60ae2dcada04f30bb73997b2bff7770e7be21"
    )
    assert rows[4]["mapping"].endswith("native-history-paced-v2")
    old_rows = copy.deepcopy(rows)
    old_rows[4]["mapping"] = "three-flow-river-http-kimi-pro-native-history-v1"
    for field in (
        "transport_policy",
        "pacing_scope",
        "minimum_spacing_seconds",
        "retry_after_max_seconds",
        "retry_eligibility",
        "automatic_resubmission",
    ):
        del old_rows[4]["request_settings"][field]
    validate_historical_identity(
        {
            "campaign_root": "/synthetic/campaign",
            "custody_root": "/synthetic/custody",
            "credential_file": "/synthetic/credential",
            "roster_profile": LATE_NOVEMBER_PROFILE,
            "runtime": {
                "python": "synthetic",
                "executable": "/synthetic/python",
                "executable_sha256": "a" * 64,
                "dependencies": [],
                "lock_sha256": "b" * 64,
                "runtime_digest": "c" * 64,
            },
            "registry": old_rows,
        }
    )


def test_kimi_pacing_constructor_keeps_profile_guards(tmp_path):
    tmp_path.chmod(0o700)
    model = "moonshotai/Kimi-K3.1-Pro"
    budget = CampaignBudget(
        tmp_path,
        campaign_id="guards",
        cap=Decimal("1000"),
        roster_profile=LATE_NOVEMBER_PROFILE,
        create=True,
    )
    profile = {
        "endpoint": "https://api.river.ai/v1/chat/completions",
        "input_max": None,
        "output_max": 1048575,
        "context_max": 1048576,
        "owner_output_cap_tokens": 30000,
        "bounds_policy": {"mode": "test"},
    }
    guard = AdmittedGuard(
        profile=profile,
        budget=budget,
        cell="trial",
        max_output_tokens=30000,
        policy=LifecyclePricingPolicy("synthetic", Decimal("2"), Decimal("10")),
    )
    kwargs = {
        "provider": "river_http",
        "model": model,
        "api_key": "offline-key",
        "inner": SelectedMock("river_http", model, "commerce"),
        "guard": guard,
        "max_output_tokens": 30000,
        "context_tokens": 1048576,
        "mode_profile": "off-or-minimum-v1",
        "roster_profile": LATE_NOVEMBER_PROFILE,
        "transport_policy": "paced-safe-errors-v1",
    }
    try:
        for change in (
            {"model": "foreign"},
            {"roster_profile": "legacy-v1"},
            {"mode_profile": "legacy-v1"},
            {"max_output_tokens": 30001},
        ):
            with pytest.raises(ValueError):
                HTTPCampaignTransport(**(kwargs | change))
        profile["owner_output_cap_tokens"] = 30001
        with pytest.raises(ValueError):
            HTTPCampaignTransport(**kwargs)
    finally:
        budget.close()
