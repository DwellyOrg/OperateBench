"""Bounded Claude review regressions: no live authority or network."""

import json
from decimal import Decimal

import pytest

from tools import campaign as c


def live_config():
    config = json.loads(c.EXAMPLE.read_text())
    config["mode"] = "live"
    for profile in config["profiles"]:
        module = c.module(profile["key"])
        profile["input_usd_per_mtok"] = str(module.FIXED_CANARY_INPUT_USD_PER_MTOK)
        profile["output_usd_per_mtok"] = str(module.FIXED_CANARY_OUTPUT_USD_PER_MTOK)
    return config


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("field", ["input_usd_per_mtok", "output_usd_per_mtok"])
def test_live_below_floor_refused_before_authority(tmp_path, monkeypatch, index, field):
    from tools import run_episode100 as primitive

    config = live_config()
    config["profiles"][index][field] = str(Decimal(config["profiles"][index][field]) / 2)
    accesses = []

    def forbidden(*args, **kwargs):
        accesses.append(True)
        raise AssertionError("authority/credential access before floor refusal")

    monkeypatch.setattr(c, "consume_authority", forbidden)
    monkeypatch.setattr(primitive, "private_fd", forbidden)
    monkeypatch.setattr(primitive, "read_json", forbidden)
    with pytest.raises(ValueError, match="historical profile rate floor"):
        c.run_campaign(config, tmp_path / "run", authority=tmp_path / "absent")
    assert accesses == []
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("multiplier", [1, 2])
def test_live_floor_or_higher_is_not_current_price_verification(tmp_path, multiplier):
    config = live_config()
    for profile in config["profiles"]:
        for field in ("input_usd_per_mtok", "output_usd_per_mtok"):
            profile[field] = str(Decimal(profile[field]) * multiplier)
    assert c.make_plan(config, tmp_path / "run")["config"] == config


def test_execution_uses_sealed_config_despite_caller_callback(tmp_path, monkeypatch):
    config = json.loads(c.EXAMPLE.read_text())
    original = json.loads(c.canonical(config))
    root = tmp_path / "run"
    publish = c.publish
    observed = []

    def mutate_after_seal(path, value):
        publish(path, value)
        if path.name == "plan.json":
            config["campaign_id"] = "mutated"
            config["aggregate_usd"] = "999"
            config["mode"] = "live"
            config["repeats"] = 99
            config["profiles"][0]["money"] = "999"
            config["profiles"][0]["tokens"] = 1
            config["profiles"][0]["input_usd_per_mtok"] = "999"

    def inspect(directory, trial, profile, budget, **kwargs):
        observed.append(
            (
                json.loads(c.canonical(profile)),
                budget.cap,
                budget.namespace,
                json.loads(c.canonical(budget.limits)),
                kwargs["offline"],
            )
        )
        raise KeyboardInterrupt()

    monkeypatch.setattr(c, "publish", mutate_after_seal)
    monkeypatch.setattr(c, "execute_trial", inspect)
    c.run_campaign(config, root)
    assert len(observed) == 1
    profile, cap, namespace, limits, offline = observed[0]
    assert profile == original["profiles"][0]
    assert cap == Decimal(original["aggregate_usd"])
    assert namespace == "offline-test-campaign-" + original["campaign_id"]
    assert offline is True
    sealed = json.loads((root / "plan.json").read_text())["plan"]
    assert sealed["config"] == original
    pmap = {p["key"]: p for p in original["profiles"]}
    assert limits == {t["trial_id"]: pmap[t["profile_key"]] for t in sealed["trials"]}


@pytest.mark.parametrize("ending", ["normal", "interrupt", "setup_error", "cli"])
def test_offline_guard_is_scoped_and_restores_socket(tmp_path, monkeypatch, ending):
    import socket

    config = json.loads(c.EXAMPLE.read_text())
    if ending == "normal":
        config["aggregate_usd"] = "0.000000001"
    root = tmp_path / "run"
    original_connect = socket.socket.connect
    checked = []

    def probe():
        # Closed descriptor: even a missing tripwire cannot make a real connection.
        sock = socket.socket()
        sock.close()
        with pytest.raises(RuntimeError, match="offline internet access forbidden"):
            sock.connect(("127.0.0.1", 9))
        checked.append(True)

    def interrupt(*args, **kwargs):
        probe()
        raise KeyboardInterrupt()

    publish = c.publish

    def publishing(path, value):
        if path.name == "plan.json":
            probe()
            if ending == "setup_error":
                raise OSError("synthetic publication failure")
        return publish(path, value)

    monkeypatch.setattr(c, "execute_trial", interrupt)
    monkeypatch.setattr(c, "publish", publishing)
    if ending == "setup_error":
        with pytest.raises(OSError, match="synthetic publication failure"):
            c.run_campaign(config, root)
    elif ending == "cli":
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))
        assert c.main(["run", "--config", str(config_path), "--output", str(root)]) == 1
    else:
        c.run_campaign(config, root)
    assert checked
    assert socket.socket.connect is original_connect
    sock = socket.socket()
    sock.close()
    with pytest.raises(OSError):
        sock.connect(("127.0.0.1", 9))


def test_tripwire_nested_restores_all_functions_without_audit_hook(monkeypatch):
    import socket
    import sys

    calls = []

    def sentinel(*args, **kwargs):
        calls.append(True)
        return "restored"

    def no_audit_hook(*args):
        pytest.fail("irreversible audit hook installed")

    monkeypatch.setattr(sys, "addaudithook", no_audit_hook)
    targets = [
        (socket.socket, name)
        for name in ("connect", "connect_ex", "send", "sendall", "sendto", "sendmsg")
        if hasattr(socket.socket, name)
    ] + [
        (socket, name)
        for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr")
    ]
    for owner, name in targets:
        monkeypatch.setattr(owner, name, sentinel)
    for _ in range(2):
        with c.network_tripwire():
            outer = [getattr(owner, name) for owner, name in targets]
            with c.network_tripwire():
                for family in (socket.AF_INET, socket.AF_INET6):
                    with socket.socket(family) as sock:
                        for owner, name in targets:
                            args = (sock,) if owner is socket.socket else ()
                            with pytest.raises(
                                RuntimeError, match="offline internet access forbidden"
                            ):
                                getattr(owner, name)(*args)
            assert [getattr(owner, name) for owner, name in targets] == outer
            assert not calls
        for owner, name in targets:
            assert getattr(owner, name) is sentinel
    # Restored Python routing is exercised without any DNS or internet operation.
    assert socket.getaddrinfo("unused.invalid", 9) == "restored"
    assert calls == [True]
