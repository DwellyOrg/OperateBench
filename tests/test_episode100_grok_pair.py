# SPDX-License-Identifier: Apache-2.0
"""Exact closed subset and non-resetting series admission, no paid calls."""

import json
from decimal import Decimal

import pytest

from tools import run_episode100 as launch
from tools.diagnose_aggregate_budget import PLACEHOLDER


def series_file(root):
    root.chmod(0o700)
    value = {
        "schema": "episode100-series-v1",
        "series_id": "offline-synthetic-series",
        "prior_evidence_sha256": ["a" * 64],
        "authorized_total_usd": "11",
        "committed_exposure_usd": "1",
        "new_cap_usd": "10",
    }
    path = root / "series.json"
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    return path, value


def pair_files(root):
    path, series = series_file(root)
    output = root / "pair"
    credentials = root / "dummy-credentials.json"
    credentials.write_text(json.dumps(dict.fromkeys(launch.GROK_CELLS, PLACEHOLDER)))
    credentials.chmod(0o600)
    approval = {
        "profile": launch.GROK_PROFILE,
        "source_sha": launch.source_sha(),
        "output": str(output),
        "cap_usd": "10",
        "cells": list(launch.GROK_CELLS),
        "max_episode_decisions": 100,
        "retries": 0,
        "approved": True,
        "historical_authorities_retired": True,
        "mode": "dummy",
        "credential_file": str(credentials),
        "series_manifest_sha256": launch.series_digest(series),
    }
    authority = root / "dummy-approval.json"
    authority.write_text(json.dumps(approval))
    authority.chmod(0o600)
    argv = [
        "--mock-authority",
        "--cells",
        "grok45,grok46",
        "--cap-usd",
        "10",
        "--output",
        str(output),
        "--authority",
        str(authority),
        "--series-manifest",
        str(path),
    ]
    return argv, authority, output, series


def test_exact_source_pair_real_sdk_audit_replay_and_durable_carry(tmp_path, monkeypatch):
    argv, authority, output, series = pair_files(tmp_path)
    original = launch.read_json
    reads = []

    def observe(fd):
        value = original(fd)
        reads.append(value)
        if isinstance(value, dict) and set(value) == set(launch.GROK_CELLS):
            receipt = json.loads((tmp_path / "EPISODE100-CONSUMED").read_text())
            assert receipt["cells"] == list(launch.GROK_CELLS)
            assert receipt["series_manifest_sha256"] == launch.series_digest(series)
        return value

    monkeypatch.setattr(launch, "read_json", observe)
    assert launch.main(argv) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["controller_profile"] == launch.GROK_PROFILE
    assert [c["cell"] for c in summary["cells"]] == list(launch.GROK_CELLS)
    assert all(c["bundle_ok"] and c["replay_ok"] for c in summary["cells"])
    assert all(c["api_requests"] == c["decision_turns"] == 46 for c in summary["cells"])
    assert all(c["retries"] == 0 for c in summary["cells"])
    assert json.loads((output / "series-manifest.json").read_text()) == series
    assert summary["committed_exposure_usd"] == "1"
    assert len(reads) == 3
    # No second attempt, including after an output-path change.
    with pytest.raises(FileExistsError):
        launch.consume(
            authority,
            output=output,
            cap=Decimal("10"),
            mock=True,
            cells=launch.GROK_CELLS,
            series=series,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("cells", ["grok45"]),
        ("cells", ["grok46", "grok45"]),
        ("cells", list(launch.CELLS)),
        ("max_episode_decisions", 101),
        ("profile", launch.PROFILE),
        ("series_manifest_sha256", "b" * 64),
        ("source_sha", "0" * 40),
        ("cap_usd", "11"),
    ],
)
def test_authority_mismatch_before_secret_read(tmp_path, monkeypatch, field, value):
    argv, authority, output, _ = pair_files(tmp_path)
    approval = json.loads(authority.read_text())
    approval[field] = value
    authority.write_text(json.dumps(approval))
    original = launch.private_fd

    def refuse_secret(path):
        assert str(path) != approval["credential_file"]
        return original(path)

    monkeypatch.setattr(launch, "private_fd", refuse_secret)
    with pytest.raises(ValueError, match="exactly match"):
        launch.main(argv)
    assert not output.exists()
    assert not (tmp_path / "EPISODE100-CONSUMED").exists()


@pytest.mark.parametrize(
    "cells",
    ["grok45", "grok46", "grok46,grok45", "haiku45,grok46", "grok45,grok46,grok45"],
)
def test_cli_rejects_every_other_subset(tmp_path, cells):
    with pytest.raises(SystemExit):
        launch.main(
            [
                "--offline",
                "--cells",
                cells,
                "--cap-usd",
                "10",
                "--output",
                str(tmp_path / "out"),
            ]
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("committed_exposure_usd", "0"),
        ("new_cap_usd", "11"),
        ("authorized_total_usd", "10"),
        ("prior_evidence_sha256", []),
        ("committed_exposure_usd", "NaN"),
        ("new_cap_usd", 10),
    ],
)
def test_series_cannot_reset_or_drop_prior_link(tmp_path, field, value):
    path, series = series_file(tmp_path)
    series[field] = value
    path.write_text(json.dumps(series))
    with pytest.raises(ValueError):
        launch.load_series(path, Decimal("10"), launch.GROK_CELLS)


def test_pair_requires_series_and_default_refuses_it(tmp_path):
    path, _ = series_file(tmp_path)
    with pytest.raises(ValueError):
        launch.load_series(None, Decimal("10"), launch.GROK_CELLS)
    with pytest.raises(ValueError):
        launch.load_series(path, Decimal("10"), launch.CELLS)
