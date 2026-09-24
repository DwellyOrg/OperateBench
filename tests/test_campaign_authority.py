"""Fresh-authority tests use only synthetic records and public dummy keys."""

import json
import os

import pytest

from tools import campaign as c


def prepare(tmp_path, monkeypatch):
    config = json.loads(c.EXAMPLE.read_text())
    config["repeats"] = 1
    source = c.source_identity()
    source["clean"] = True
    # Helper-level clean-source simulation is explicit; a real clean-source CLI
    # rehearsal is separate. No live credential store is accessed here.
    monkeypatch.setattr(c, "source_identity", lambda: dict(source))
    root = tmp_path / "run"
    plan = c.make_plan(config, root)
    credential = tmp_path / "dummy-keys.json"
    c.publish(credential, dict.fromkeys(c.PROFILE_KEYS, c.PLACEHOLDER))
    authority = tmp_path / "dummy-authority.json"
    c.publish(
        authority,
        {
            "schema": "operatebench.campaign-authority.v1",
            "plan_sha256": c.digest(plan),
            "mode": "dummy",
            "approved": True,
            "historical_authorities_retired": True,
            "credential_file": str(credential),
        },
    )
    return config, plan, root, authority, credential, source


def test_dummy_consuming_cli_fsync_before_key_read(tmp_path, monkeypatch, capsys):
    from tools import run_episode100 as primitive

    config, _plan, root, authority, credential, _ = prepare(tmp_path, monkeypatch)
    config_path = tmp_path / "config.json"
    c.publish(config_path, config)
    original = primitive.read_json
    real_fsync = os.fsync
    synced = set()
    read = []

    def fsync(fd):
        real_fsync(fd)
        synced.add((os.fstat(fd).st_dev, os.fstat(fd).st_ino))

    def checked(fd):
        if os.fstat(fd).st_ino == credential.stat().st_ino:
            for path in (
                tmp_path / "CAMPAIGN-CONSUMED",
                root / "authority-receipt.json",
                tmp_path,
                root,
            ):
                assert path.exists()
                assert (path.stat().st_dev, path.stat().st_ino) in synced
            read.append(True)
        return original(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(primitive, "read_json", checked)
    assert (
        c.main(
            [
                "run",
                "--config",
                str(config_path),
                "--output",
                str(root),
                "--authority",
                str(authority),
                "--mock-authority",
            ]
        )
        == 0
    )
    assert read == [True]
    assert c.rebuild_report(root)["counts"]["scored"] == 2
    assert (
        c.main(
            [
                "run",
                "--config",
                str(config_path),
                "--output",
                str(root),
                "--authority",
                str(authority),
                "--mock-authority",
            ]
        )
        == 2
    )
    assert read == [True]
    assert c.PLACEHOLDER not in capsys.readouterr().out


@pytest.mark.parametrize("change", ["source", "config", "output", "plan", "mode"])
def test_authority_binding_refuses_before_credential_open(tmp_path, monkeypatch, change):
    from tools import run_episode100 as primitive

    _, plan, _root, authority, credential, source = prepare(tmp_path, monkeypatch)
    if change == "source":
        source["commit"] = "0" * 40
    elif change == "config":
        plan["config"]["aggregate_usd"] = "0.99"
    elif change == "output":
        plan["output"] = str(tmp_path / "other")
    elif change == "plan":
        plan["trials"][0]["trial_id"] += "-changed"
    else:
        plan["config"]["mode"] = "live"
    original = primitive.private_fd
    opened = []

    def guarded(path):
        assert str(path) != str(credential), "credential opened on mismatch"
        opened.append(path)
        return original(path)

    monkeypatch.setattr(primitive, "private_fd", guarded)
    with pytest.raises(ValueError):
        c.consume_authority(authority, plan, mock=True)
    assert not (tmp_path / "CAMPAIGN-CONSUMED").exists()


def test_failed_fsync_never_releases_dummy_key(tmp_path, monkeypatch):
    from tools import run_episode100 as primitive

    _, plan, root, authority, credential, _ = prepare(tmp_path, monkeypatch)
    root.mkdir(mode=0o700)
    original = primitive.read_json

    def guarded(fd):
        assert os.fstat(fd).st_ino != credential.stat().st_ino
        return original(fd)

    def fail(fd):
        raise OSError("synthetic durability fault")

    monkeypatch.setattr(primitive, "read_json", guarded)
    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError):
        c.consume_authority(authority, plan, mock=True)
    assert (tmp_path / "CAMPAIGN-CONSUMED").exists()


def test_ambient_credential_presence_refused_without_logging_value(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("OPENAI_API_KEY", "DUMMY-SECRET-NOT-REAL")
    assert c.main(["run", "--output", str(tmp_path / "run")]) == 2
    assert not (tmp_path / "run").exists()
    assert "DUMMY-SECRET-NOT-REAL" not in str(capsys.readouterr())
