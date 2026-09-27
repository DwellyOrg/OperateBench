"""Public profile packs execute through the shared development record lane."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from operatebench.cli import main
from operatebench.sdk.api import CheckRequest, ReplayRequest, RunRequest, ValidateRequest
from operatebench.sdk.builtins import BUILTIN_PACKS
from operatebench.sdk.development_record import _seal_record, read_record
from operatebench.sdk.development_runtime import DevelopmentRuntimeError, replay
from operatebench.sdk.profile_packs.pack import COMMERCE_COMMANDS, COMPLIANCE_COMMANDS

COMMANDS = (COMMERCE_COMMANDS, COMPLIANCE_COMMANDS)
PACKS = (
    (
        "commerce.return_refund.profiles.v1",
        "commerce.return_refund.profiles",
        "UK_NORMAL",
    ),
    (
        "lettings.property_compliance.profiles.v1",
        "lettings.property_compliance.profiles",
        "EN_normal",
    ),
)
CASES = [
    (commands, scenario)
    for commands in COMMANDS
    for scenario in commands.reference_scenarios
]


@pytest.mark.parametrize("pack_id,alias,scenario", PACKS)
def test_registered_profile_round_trip(tmp_path, pack_id, alias, scenario):
    pack = BUILTIN_PACKS.resolve(pack_id)
    assert BUILTIN_PACKS.resolve(alias) is pack
    assert not pack.metadata.evidence_eligible
    spec = Path(pack.metadata.default_spec)
    assert pack.validate(ValidateRequest(spec)).contract_passed
    output = tmp_path / "run.json"
    result = pack.run(RunRequest(spec, scenario, "reference", output))
    assert result.contract_passed
    assert result.payload["format"] == "operatebench.development-run.v1"
    result = pack.replay(ReplayRequest(spec, output))
    assert result.contract_passed
    assert result.payload["provider_calls"] == 0


@pytest.mark.parametrize("commands,scenario", CASES, ids=[s for _, s in CASES])
@pytest.mark.parametrize("agent", ["reference", "reference-mock"])
def test_all_cases_persist_read_replay(tmp_path, monkeypatch, commands, scenario, agent):
    pack = BUILTIN_PACKS.resolve(commands.factories.pack_id)
    spec = Path(pack.metadata.default_spec)
    output = tmp_path / "run.json"
    result = pack.run(RunRequest(spec, scenario, agent, output))
    assert result.contract_passed
    record = read_record(output)
    assert record == result.payload
    assert record["binding"]["scenario_id"] == scenario
    assert (
        record["binding"]["spec_digest"]
        == commands.factories.load_spec(spec).content_digest
    )
    assert record["binding"]["profile"] == commands.factories.profile_identity(
        commands.factories.load_spec(spec), scenario
    )
    assert record["episode"]["trajectory"]
    assert record["episode"]["final_state"]
    if agent == "reference-mock":
        assert record["agent"]["kind"] == "reference_driven_mock_model"
        assert record["agent"]["mock_calls"] == len(record["decisions"]) > 0
    # Poison the actual solver class, not a callback identity in the record.
    solver_type = type(commands.factories.build_agent("reference"))

    def forbidden(*args, **kwargs):
        raise AssertionError("replay invoked an original solver")

    monkeypatch.setattr(solver_type, "decide", forbidden)
    result = pack.replay(ReplayRequest(spec, output))
    assert result.contract_passed
    assert result.payload["provider_calls"] == 0
    assert result.payload["decisions_consumed"] == len(record["decisions"])


@pytest.mark.parametrize("commands", COMMANDS)
def test_check_authored_reference_and_negative_oracles(commands):
    pack = BUILTIN_PACKS.resolve(commands.factories.pack_id)
    result = pack.check(CheckRequest(Path(pack.metadata.default_spec)))
    assert result.contract_passed, result.payload
    rows = result.payload["rows"]
    assert len(rows) == len(commands.reference_agents) * len(
        commands.reference_scenarios
    ) + len(commands.negative_controls)
    assert all(row["passed"] for row in rows)


@pytest.mark.parametrize("commands", COMMANDS)
def test_check_does_not_derive_expectations_from_observed_result(monkeypatch, commands):
    import operatebench.sdk.development_pack as adapter

    pack = BUILTIN_PACKS.resolve(commands.factories.pack_id)
    oracle = commands.negative_controls[0]
    original = adapter.run

    def mutated(*args, **kwargs):
        record = original(*args, **kwargs)
        record["evaluation"] = {
            "reliable": True,
            "dimensions": dict.fromkeys(record["evaluation"]["dimensions"], True),
            "findings": [],
            "terminal_outcome": "pretended-success",
        }
        return record

    monkeypatch.setattr(adapter, "run", mutated)
    result = pack.check(
        CheckRequest(Path(pack.metadata.default_spec), (oracle["agent_id"],))
    )
    assert not result.contract_passed
    assert any(not row["passed"] for row in result.payload["rows"])


@pytest.mark.parametrize("pack_id,alias,scenario", PACKS)
def test_cli_commands_and_refusals(tmp_path, capsys, pack_id, alias, scenario):
    pack = BUILTIN_PACKS.resolve(pack_id)
    spec = pack.metadata.default_spec
    output = str(tmp_path / "run.json")
    assert main(["list-packs", "--json"]) == 0
    assert pack_id in capsys.readouterr().out
    assert main(["validate", spec, "--pack", alias, "--json"]) == 0
    capsys.readouterr()
    args = [
        "run",
        "--pack",
        pack_id,
        "--spec",
        spec,
        "--scenario",
        scenario,
        "--agent",
        "reference-mock",
        "--output",
        output,
        "--json",
    ]
    assert main(args) == 0
    capsys.readouterr()
    before = Path(output).read_bytes()
    assert main(args) == 1
    assert Path(output).read_bytes() == before
    capsys.readouterr()
    assert (
        main(["replay", "--pack", alias, "--spec", spec, "--run", output, "--json"]) == 0
    )
    capsys.readouterr()
    other = BUILTIN_PACKS.resolve(next(p for p, _, _ in PACKS if p != pack_id))
    assert main(["validate", spec, "--pack", other.metadata.pack_id, "--json"]) == 1
    capsys.readouterr()
    assert (
        main(
            [
                "replay",
                "--pack",
                other.metadata.pack_id,
                "--spec",
                other.metadata.default_spec,
                "--run",
                output,
                "--json",
            ]
        )
        == 1
    )
    capsys.readouterr()
    changed = tmp_path / "changed.yaml"
    body = yaml.safe_load(Path(spec).read_text())
    body["profiles"] = {}
    changed.write_text(json.dumps(body))
    assert (
        main(
            [
                "replay",
                "--pack",
                pack_id,
                "--spec",
                str(changed),
                "--run",
                output,
                "--json",
            ]
        )
        == 1
    )
    capsys.readouterr()
    args[args.index("--scenario") + 1] = "missing"
    assert main(args) == 1
    capsys.readouterr()


@pytest.mark.parametrize("commands", COMMANDS)
@pytest.mark.parametrize("field", ["profile", "spec", "state", "time", "case"])
def test_resealed_record_binding_or_episode_mutation_refused(tmp_path, commands, field):
    pack = BUILTIN_PACKS.resolve(commands.factories.pack_id)
    spec = Path(pack.metadata.default_spec)
    output = tmp_path / "run.json"
    record = deepcopy(
        pack.run(
            RunRequest(spec, commands.reference_scenarios[0], "reference", output)
        ).payload
    )
    if field == "profile":
        record["binding"]["profile"]["profile_digest"] = "0" * 64
    elif field == "spec":
        record["binding"]["spec_digest"] = "0" * 64
    elif field == "case":
        record["binding"]["scenario_id"] = commands.reference_scenarios[1]
    elif field == "state":
        record["episode"]["final_state"]["invented"] = True
    else:
        record["episode"]["trajectory"][0]["at"] = "2099-01-01T00:00:00Z"
    expected = {
        "profile": "trusted spec/profile/implementation/runtime binding mismatch",
        "spec": "trusted spec/profile/implementation/runtime binding mismatch",
        "case": "episode identity mismatch",
        "state": "recomputed full episode differs",
        "time": "recomputed full episode differs",
    }[field]
    record.pop("record_digest")
    # Seal outside the rejection assertion: all five mutations remain valid
    # codec inputs, and must reach the binding/recomputed-episode replay gate.
    sealed = _seal_record(record)
    with pytest.raises(DevelopmentRuntimeError, match=expected):
        replay(commands.factories, spec, sealed)


def test_runtime_callbacks_are_typed_and_metadata_agents_align():
    for commands in COMMANDS:
        pack = BUILTIN_PACKS.resolve(commands.factories.pack_id)
        assert set(pack.metadata.agent_ids) == {*commands.agent_ids, "reference-mock"}
        assert replace(commands).factories is commands.factories
