import json
from decimal import Decimal

import pytest

from tools import run_episode100 as launch
from tools.aggregate_budget import CELLS, Budget
from tools.diagnose_aggregate_budget import PLACEHOLDER


def test_controller_namespace_is_not_paid_authority(tmp_path):
    tmp_path.chmod(0o700)
    budget = Budget(
        tmp_path, cap=Decimal("1"), namespace="episode100-paid-candidate", create=True
    )
    assert budget.totals()["exposure"] == 0
    budget.close()


def approval_files(tmp_path, *, malformed=False):
    tmp_path.chmod(0o700)
    credentials = tmp_path / "dummy-credentials.json"
    credentials.write_text(
        json.dumps({} if malformed else dict.fromkeys(CELLS, PLACEHOLDER))
    )
    credentials.chmod(0o600)
    output = tmp_path / "output"
    authority = tmp_path / "dummy-approval.json"
    authority.write_text(
        json.dumps(
            {
                "profile": launch.PROFILE,
                "source_sha": launch.source_sha(),
                "output": str(output),
                "cap_usd": "10",
                "cells": list(CELLS),
                "max_episode_decisions": 100,
                "retries": 0,
                "approved": True,
                "historical_authorities_retired": True,
                "mode": "dummy",
                "credential_file": str(credentials),
            }
        )
    )
    authority.chmod(0o600)
    return authority, output


def test_actual_cli_dummy_consumes_before_key_read_and_six_sdk(tmp_path, monkeypatch):
    authority, output = approval_files(tmp_path)
    original = launch.read_json
    reads = []

    def observe(fd):
        reads.append(fd)
        if len(reads) == 2:
            assert (tmp_path / "EPISODE100-CONSUMED").exists()
        return original(fd)

    monkeypatch.setattr(launch, "read_json", observe)
    assert (
        launch.main(
            [
                "--mock-authority",
                "--authority",
                str(authority),
                "--output",
                str(output),
                "--cap-usd",
                "10",
            ]
        )
        == 0
    )
    summary = json.loads((output / "summary.json").read_text())
    assert len(summary["cells"]) == 6
    assert all(c["bundle_ok"] and c["replay_ok"] for c in summary["cells"])
    assert summary["aggregate"]["pending"] == summary["aggregate"]["unknown"] == "0"
    assert len(reads) == 2


def test_malformed_dummy_key_consumption_is_permanent(tmp_path):
    authority, output = approval_files(tmp_path, malformed=True)
    argv = [
        "--mock-authority",
        "--authority",
        str(authority),
        "--output",
        str(output),
        "--cap-usd",
        "10",
    ]
    with pytest.raises(ValueError, match="authority remains consumed"):
        launch.main(argv)
    assert not output.exists()
    with pytest.raises(FileExistsError):
        launch.main(argv)


def test_wrong_source_authority_does_not_consume_or_read_keys(tmp_path):
    authority, output = approval_files(tmp_path)
    approval = json.loads(authority.read_text())
    approval["source_sha"] = "0" * 40
    authority.write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="exactly match"):
        launch.main(
            [
                "--mock-authority",
                "--authority",
                str(authority),
                "--output",
                str(output),
                "--cap-usd",
                "10",
            ]
        )
    assert not (tmp_path / "EPISODE100-CONSUMED").exists()


def test_live_without_new_authority_refuses_before_credentials(tmp_path):
    with pytest.raises(ValueError, match="new external authority"):
        launch.main(["--live", "--output", str(tmp_path / "out"), "--cap-usd", "3"])
    assert not (tmp_path / "out").exists()


def test_required_cap_never_inherits_historical_ceiling(tmp_path):
    with pytest.raises(SystemExit):
        launch.main(["--offline", "--output", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
