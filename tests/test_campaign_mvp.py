"""Narrow campaign acceptance; no network or credential stores."""

import importlib.util
import json
from decimal import Decimal

import httpx
import pytest


def configuration():
    from tools.campaign import EXAMPLE

    return json.loads(EXAMPLE.read_text())


@pytest.mark.parametrize("fault", ["interrupt", "provider", "unknown"])
def test_stop_and_rebuild_without_resume(tmp_path, fault):
    from tools.campaign import module, rebuild_report, run_campaign

    config = configuration()
    root = tmp_path / "run"
    scripted = module("haiku45").ScriptedProvider(max_calls=50)

    def handler(request):
        if fault == "interrupt":
            raise KeyboardInterrupt()
        if fault == "provider":
            return httpx.Response(
                503,
                json={
                    "error": {"type": "api_error", "message": "DUMMY-SECRET-DO-NOT-LOG"}
                },
            )
        response = scripted(request)
        body = json.loads(response.content)
        body["usage"] = None
        return httpx.Response(200, json=body)

    report = run_campaign(
        config, root, handlers={"offline-two-sdk-r001-haiku45": handler}
    )
    assert report["counts"]["started"] == 1
    assert report["counts"]["scored"] == 0
    assert report["counts"]["reliable"] == 0
    assert report["fractions"]["scored"]["fraction"] is None
    assert [t["status"] for t in report["trials"]][1:] == ["not_started"] * 3
    assert report["trials"][0]["reliable"] is None
    assert rebuild_report(root) == report
    with pytest.raises(FileExistsError):
        run_campaign(config, root)
    assert all(
        b"DUMMY-SECRET-DO-NOT-LOG" not in p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    )


def test_prestart_budget_stop(tmp_path):
    from tools.campaign import run_campaign

    config = configuration()
    config["aggregate_usd"] = "0.000000001"
    report = run_campaign(config, tmp_path / "run")
    assert report["counts"]["started"] == 0
    assert report["counts"]["scored"] == 0
    assert all(t["status"] == "not_started" for t in report["trials"])
    assert report["accounting"]["exposure"] == "0"


@pytest.mark.parametrize("sdk", ["anthropic", "openai"])
@pytest.mark.parametrize("constraint", ["trial", "aggregate", "tokens"])
def test_wire_supplement_refused_before_http(tmp_path, constraint, sdk):
    import time

    from operatebench.agents.pricing import LifecyclePricingPolicy
    from operatebench.providers.cost import CostCapExceededError
    from tools.campaign import CampaignBudget, DispatchTransport

    root = tmp_path / "budget"
    root.mkdir(mode=0o700)
    limit = {
        "money": "0.001" if constraint == "trial" else "10",
        "tokens": 20 if constraint == "tokens" else 100000,
    }
    budget = CampaignBudget(
        root,
        cap=Decimal("0.001" if constraint == "aggregate" else "10"),
        namespace="offline-test-wire",
        create=True,
        limits={"t": limit},
    )
    policy = LifecyclePricingPolicy(
        policy_id="operator_pinned_v1",
        input_usd_per_mtok=Decimal("1"),
        output_usd_per_mtok=Decimal("1"),
        rate_source="operator_supplied_pinned_rates",
    )
    calls = []
    try:
        budget.reserve("t", input_bound=1, output_bound=1, policy=policy)
        wire = DispatchTransport(
            httpx.MockTransport(lambda r: calls.append(r) or httpx.Response(200)),
            budget,
            "t",
            time.monotonic() + 30,
        )
        import anthropic
        import openai

        client_type = anthropic.Anthropic if sdk == "anthropic" else openai.OpenAI
        client = client_type(
            api_key="offline-placeholder-not-a-real-key",
            max_retries=0,
            http_client=httpx.Client(transport=wire, trust_env=False),
        )
        try:
            with pytest.raises(
                (anthropic.APIConnectionError, openai.APIConnectionError)
            ) as error:
                if sdk == "anthropic":
                    client.messages.create(
                        model="offline-model",
                        max_tokens=1,
                        messages=[{"role": "user", "content": "x" * 20000}],
                    )
                else:
                    client.responses.create(
                        model="offline-model", max_output_tokens=1, input="x" * 20000
                    )
            assert isinstance(error.value.__cause__, CostCapExceededError)
        finally:
            client.close()
        assert calls == []
        assert next(iter(budget.records.values()))["state"] == "reserved"
    finally:
        budget.close()


@pytest.mark.parametrize("constraint", ["trial", "aggregate"])
def test_unknown_wire_exposure_is_held_without_double_count(tmp_path, constraint):
    from fractions import Fraction

    from operatebench.agents.pricing import LifecyclePricingPolicy
    from operatebench.providers.cost import CostCapExceededError
    from tools.campaign import CampaignBudget

    root = tmp_path / "budget"
    root.mkdir(mode=0o700)
    limits = {"t": {"money": "0.02" if constraint == "trial" else "1", "tokens": 100000}}
    budget = CampaignBudget(
        root,
        cap=Decimal("0.02" if constraint == "aggregate" else "1"),
        namespace="offline-test-unknown",
        create=True,
        limits=limits,
    )
    policy = LifecyclePricingPolicy(
        policy_id="operator_pinned_v1",
        input_usd_per_mtok=Decimal("1"),
        output_usd_per_mtok=Decimal("1"),
        rate_source="operator_supplied_pinned_rates",
    )
    try:
        ident = budget.reserve("t", input_bound=1, output_bound=1, policy=policy)
        budget.dispatch("t", {"max_tokens": 1, "padding": "x" * 10000})
        bound = Fraction(budget.records[ident]["bound"])
        assert bound > Fraction(budget.records[ident]["initial_bound"])
        budget.forfeit(ident)
        assert budget.totals()["unknown"] == budget.exposure == bound
        assert budget.totals()["known"] == budget.totals()["pending"] == 0
        with pytest.raises(CostCapExceededError):
            budget.reserve("t", input_bound=15000, output_bound=1, policy=policy)
        assert budget.exposure == bound
    finally:
        budget.close()


def test_incomplete_start_never_invents_terminal_or_resume(tmp_path):
    from tools.campaign import rebuild_report, run_campaign

    config = configuration()
    config["aggregate_usd"] = "0.005"
    root = tmp_path / "run"
    run_campaign(config, root)
    # Simulate absence of the last durable campaign transition. Existing ledger
    # evidence can still be read, but cannot manufacture a campaign terminal.
    sorted((root / "status").iterdir())[-1].unlink()
    report = rebuild_report(root)
    assert report["trials"][0]["status"] == "started"
    assert report["trials"][0]["reliable"] is None
    assert report["trials"][0]["terminal"] is None
    assert report["counts"]["scored"] == 0
    with pytest.raises(FileExistsError):
        run_campaign(config, root)


def test_campaign_namespace_is_accounting_not_authority(tmp_path):
    from tools.aggregate_budget import Budget

    root = tmp_path / "budget"
    root.mkdir(mode=0o700)
    budget = Budget(root, cap=Decimal("1"), namespace="campaign-paid-test", create=True)
    budget.close()
    with pytest.raises(ValueError, match="unsupported controller namespace"):
        Budget(root, cap=Decimal("1"), namespace="arbitrary-paid", create=True)


def test_authority_mismatch_precedes_credential_read(tmp_path, monkeypatch):
    import tools.campaign as c

    assert hasattr(c, "consume_authority"), "fresh campaign binding missing"
    config = configuration()
    plan = c.make_plan(config, tmp_path / "run")
    plan["source"]["clean"] = True
    authority = tmp_path / "authority.json"
    c.publish(authority, {"schema": "wrong", "credential_file": "/does/not/exist"})
    with pytest.raises(ValueError):
        c.consume_authority(authority, plan, mock=True)
    assert not (tmp_path / "CAMPAIGN-CONSUMED").exists()


def test_reference_alternative_and_negative_use_unchanged_grader(tmp_path):
    from operatebench.core.retrieval import RetrieveBatch
    from operatebench.domains.lettings.maintenance.agents import build_agent
    from tools.campaign import module, run_campaign

    class EarlierApproval:
        # AlternateIdentifierAgent is now a pass-through. Instead exercise a
        # genuinely different legal retrieval ordering of the existing fixture;
        # this is not claimed to be an independent business policy.
        def __init__(self):
            self.inner = build_agent("reference")
            self.changed = 0

        def decide(self, observation):
            decision = self.inner.decide(observation)
            if isinstance(decision, RetrieveBatch) and len(decision.requests) > 1:
                self.changed += 1
                return RetrieveBatch(tuple(reversed(decision.requests)))
            return decision

    for variant in ("alternate", "negative"):
        config = configuration()
        config["repeats"] = 1
        handlers = {}
        agents = []
        for profile in ("haiku45", "luna56"):
            scripted = module(profile).ScriptedProvider(max_calls=50)
            agent = (
                EarlierApproval()
                if variant == "alternate"
                else build_agent("complete_early")
            )
            scripted._reference = agent
            agents.append(agent)
            handlers[f"offline-two-sdk-r001-{profile}"] = scripted
        result = run_campaign(config, tmp_path / variant, handlers=handlers)
        assert result["counts"]["scored"] == 2
        assert result["counts"]["reliable"] == (2 if variant == "alternate" else 0)
        assert all(t["bundle_ok"] and t["replay_ok"] for t in result["trials"])
        if variant == "alternate":
            assert all(a.changed > 0 for a in agents)
        else:
            assert result["fractions"]["scored"]["fraction"] == "0/2"


def test_terminal_hash_tampering_is_refused(tmp_path):
    from tools.campaign import rebuild_report, run_campaign

    config = configuration()
    config["aggregate_usd"] = "0.005"
    root = tmp_path / "run"
    run_campaign(config, root)
    status = next(
        p
        for p in (root / "status").iterdir()
        if json.loads(p.read_text())["status"] == "non-scored"
    )
    value = json.loads(status.read_text())
    value["reason"] = "forged"
    status.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        rebuild_report(root)


def test_two_sdk_two_repeat_campaign(tmp_path):
    assert importlib.util.find_spec("tools.campaign") is not None, (
        "campaign adapter missing"
    )
    from tools.campaign import EXAMPLE, rebuild_report, run_campaign

    config = json.loads(EXAMPLE.read_text())
    root = tmp_path / "campaign"
    report = run_campaign(config, root)
    assert report["counts"] == {"assigned": 4, "started": 4, "scored": 4, "reliable": 4}
    assert len({t["trial_id"] for t in report["trials"]}) == 4
    assert [t["profile_key"] for t in report["trials"]] == ["haiku45", "luna56"] * 2
    assert all(
        t["bundle_ok"] and t["replay_ok"] and t["dimensions"] for t in report["trials"]
    )
    assert report["accounting"]["unknown"] == "0"
    assert sum(
        Decimal(p["totals"]["measured_usd"]) for p in report["profiles"].values()
    ) == Decimal(report["accounting"]["known"])
    assert report["totals"]["requests"] == sum(t["requests"] for t in report["trials"])
    assert all(t["unknown_cost"] is False for t in report["trials"])
    assert rebuild_report(root) == report
    assert (root / "report.json").exists() and (root / "report.md").exists()
    assert root.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in root.rglob("*") if p.is_file())


@pytest.mark.parametrize("fault", ["constructor", "close"])
def test_outer_trial_failure_retains_non_scored_report(tmp_path, monkeypatch, fault):
    from tools import campaign as c

    secret = "DUMMY-PRIVATE-DETAIL"
    if fault == "constructor":

        def fail(*args, **kwargs):
            raise RuntimeError(secret)

        monkeypatch.setattr(c, "client_for", fail)
    else:
        original = c.client_for

        def broken_close(*args, **kwargs):
            client = original(*args, **kwargs)
            close = client.close

            def fail():
                close()
                raise RuntimeError(secret)

            client.close = fail
            return client

        monkeypatch.setattr(c, "client_for", broken_close)
    root = tmp_path / "run"
    report = c.run_campaign(configuration(), root)
    assert report["counts"] == {"assigned": 4, "started": 1, "scored": 0, "reliable": 0}
    assert report["trials"][0]["reason"] == "infrastructure"
    assert report == c.rebuild_report(root)
    assert secret not in json.dumps(report)


def test_report_retains_input_digests_and_refuses_partial_promotion(tmp_path):
    from tools import campaign as c

    config = configuration()
    config["aggregate_usd"] = "0.005"
    root = tmp_path / "run"
    report = c.run_campaign(config, root)
    for name, sha in report["input_sha256"].items():
        assert sha == c.file_digest(root / name)
    assert "successor-events.jsonl" in report["input_sha256"]
    status = sorted((root / "status").iterdir())[-1]
    event = json.loads(status.read_text())
    assert event["status"] == "non-scored"
    event["status"] = "scored"
    event.pop("event_sha256")
    event["event_sha256"] = c.digest(event)
    status.write_bytes(c.canonical(event))
    with pytest.raises((ValueError, FileNotFoundError)):
        c.rebuild_report(root)
