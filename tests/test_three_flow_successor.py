"""Next-series selection is explicit; historical evidence is not migrated."""

import json
from dataclasses import replace

import pytest

from tests.test_three_flow_runtime import scripted_handler, setup_transport
from tools.three_flow_runtime import (
    SPECS,
    Trial,
    provider_identity,
    replay_trial,
    run_trial,
)


@pytest.mark.parametrize("scenario", ["V1", "V2", "V3"])
def test_next_series_public_reference_and_replay(tmp_path, scenario):
    assert SPECS["maintenance"].endswith("maintenance_delivery_recovery_v0_7.yaml")
    calls = []
    trial = Trial(
        "next-series", "reference", "maintenance", scenario, "gpt-6-astra", 128000
    )
    budget, transport = setup_transport(
        tmp_path, trial, scripted_handler("maintenance", calls)
    )
    trial = replace(trial, provider_binding=provider_identity(transport))
    try:
        record = run_trial(trial, transport, output=tmp_path / "reference")
        assert record["evaluation"]["reliable"]
        assert (
            record["identity"]["operation_id"]
            == "lettings_maintenance_delivery_recovery_v2"
        )
        rows = [
            json.loads(line)
            for line in (tmp_path / "reference" / "partial.ndjson")
            .read_text()
            .splitlines()
        ]
        decisions = [row["decision"] for row in rows if row["kind"] == "decision"]
        assert decisions == record["decisions"]
        assert len(decisions) == len(calls)
        assert (
            replay_trial(trial, record, expected_record_digest=record["record_digest"])[
                "provider_calls"
            ]
            == 0
        )
        assert len(calls) == len(decisions)
    finally:
        transport.close()
        budget.close()
