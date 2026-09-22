"""The four OperateBench commands, their exit codes and their failure messages.

Exit codes are part of the contract, because these commands are meant to be
wired into CI:

* ``0`` — the command ran and the answer is yes;
* ``1`` — the input was refused (a named domain error, never a traceback);
* ``2`` — the command ran and the answer is no (unreliable run, diverged replay,
  failed acceptance gate).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from operatebench.artifact import read_artifact
from operatebench.cli import EXIT_CONTRACT_FAILED, EXIT_ERROR, EXIT_OK, main

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "examples" / "operatebench" / "maintenance_v0_1.yaml"


def run(*argv: str) -> int:
    return main(list(argv))


class TestValidate:
    def test_validate_accepts_the_shipped_fixture(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert run("validate", str(FIXTURE)) == EXIT_OK
        out = capsys.readouterr().out
        assert "lettings_maintenance_synthetic_v1" in out
        assert "SYNTHETIC_ONLY" in out

    def test_validate_emits_json_identity(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert run("validate", str(FIXTURE), "--json") == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["spec_digest_sha256"]) == 64
        assert payload["scenario_ids"] == ["V1", "V2", "V3"]

    def test_validate_refuses_a_missing_spec_without_a_traceback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert run("validate", str(tmp_path / "absent.yaml")) == EXIT_ERROR
        assert "Traceback" not in capsys.readouterr().err

    def test_validate_refuses_a_malformed_spec_by_name(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        broken = tmp_path / "broken.yaml"
        broken.write_text(
            FIXTURE.read_text(encoding="utf-8").replace(
                "operation_id:", "operation_identifier:", 1
            ),
            encoding="utf-8",
        )
        assert run("validate", str(broken)) == EXIT_ERROR
        assert "operation_identifier" in capsys.readouterr().err


class TestRunMaintenance:
    def test_a_reference_run_writes_a_replayable_artefact(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        output = tmp_path / "run.json"
        code = run(
            "run-maintenance",
            "--spec",
            str(FIXTURE),
            "--scenario",
            "V1",
            "--agent",
            "reference",
            "--output",
            str(output),
        )
        assert code == EXIT_OK
        assert "completed_successfully" in capsys.readouterr().out
        payload = read_artifact(output)
        assert payload["evaluation"]["reliable"] is True

    def test_a_negative_run_exits_with_the_contract_failed_code(
        self, tmp_path: Path
    ) -> None:
        code = run(
            "run-maintenance",
            "--spec",
            str(FIXTURE),
            "--scenario",
            "V1",
            "--agent",
            "always_wait",
            "--output",
            str(tmp_path / "neg.json"),
        )
        assert code == EXIT_CONTRACT_FAILED

    def test_an_unknown_agent_is_refused_by_name(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = run(
            "run-maintenance",
            "--spec",
            str(FIXTURE),
            "--scenario",
            "V1",
            "--agent",
            "some_model",
            "--output",
            str(tmp_path / "x.json"),
        )
        assert code == EXIT_ERROR
        assert "some_model" in capsys.readouterr().err

    def test_an_unknown_scenario_is_refused_by_name(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = run(
            "run-maintenance",
            "--spec",
            str(FIXTURE),
            "--scenario",
            "V9",
            "--agent",
            "reference",
            "--output",
            str(tmp_path / "x.json"),
        )
        assert code == EXIT_ERROR
        assert "V9" in capsys.readouterr().err

    def test_it_refuses_to_overwrite_an_existing_artefact(self, tmp_path: Path) -> None:
        output = tmp_path / "run.json"
        argv = (
            "run-maintenance",
            "--spec",
            str(FIXTURE),
            "--scenario",
            "V2",
            "--agent",
            "reference",
            "--output",
            str(output),
        )
        # V2 writes truthful unreliable evidence; overwrite protection is unchanged.
        assert run(*argv) == EXIT_CONTRACT_FAILED
        assert run(*argv) == EXIT_ERROR

    def test_json_output_carries_the_result_vector(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = run(
            "run-maintenance",
            "--spec",
            str(FIXTURE),
            "--scenario",
            "V3",
            "--agent",
            "reference",
            "--output",
            str(tmp_path / "v3.json"),
            "--json",
        )
        assert code == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["evaluation"]["reliable"] is True
        assert payload["artifact_path"].endswith("v3.json")


class TestReplay:
    def test_replaying_a_real_artefact_succeeds(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        output = tmp_path / "run.json"
        run(
            "run-maintenance",
            "--spec",
            str(FIXTURE),
            "--scenario",
            "V1",
            "--agent",
            "reference",
            "--output",
            str(output),
        )
        capsys.readouterr()
        assert run("replay", "--spec", str(FIXTURE), "--run", str(output)) == EXIT_OK
        assert "replay OK" in capsys.readouterr().out

    def test_a_tampered_artefact_is_refused_before_it_is_replayed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        output = tmp_path / "run.json"
        run(
            "run-maintenance",
            "--spec",
            str(FIXTURE),
            "--scenario",
            "V1",
            "--agent",
            "reference",
            "--output",
            str(output),
        )
        payload = read_artifact(output)
        payload["final_state"]["phase"] = "NEW"
        output.write_text(json.dumps(payload), encoding="utf-8")
        # A record whose final state no longer matches the digest written beside
        # it is not this run's record, so it is a named refusal rather than a
        # divergence report about a comparison that should never have run.
        code = run("replay", "--spec", str(FIXTURE), "--run", str(output))
        assert code == EXIT_ERROR
        err = capsys.readouterr().err
        assert "Traceback" not in err
        assert "final_state_digest_sha256" in err

    def test_a_malformed_artefact_is_a_named_domain_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        broken = tmp_path / "run.json"
        broken.write_text("{not json", encoding="utf-8")
        assert run("replay", "--spec", str(FIXTURE), "--run", str(broken)) == EXIT_ERROR
        err = capsys.readouterr().err
        assert "Traceback" not in err
        assert "JSON" in err


class TestCheckMaintenance:
    def test_the_gate_passes_on_the_shipped_fixture(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert run("check-maintenance", "--spec", str(FIXTURE)) == EXIT_OK
        out = capsys.readouterr().out
        assert "reference" in out
        assert "trust_actor_claim" in out
        assert "causal acceptance: OK" in out
        assert "maintenance.reference.V2.permanent-notification-fault.v1" in out
        assert "reliable=False" in out

    def test_the_gate_emits_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert run("check-maintenance", "--spec", str(FIXTURE), "--json") == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is True
        assert len(payload["checks"]) >= 9
        v2 = next(c for c in payload["checks"] if c["scenario_id"] == "V2")
        assert v2["ok"] is True and v2["reliable"] is False
        assert v2["expected_targets"] == ["recovery", "obligations"]

    def test_the_gate_can_be_restricted_to_named_agents(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = run(
            "check-maintenance", "--spec", str(FIXTURE), "--agent", "reference", "--json"
        )
        assert code == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert {check["agent_id"] for check in payload["checks"]} == {"reference"}

    def test_a_spec_whose_reference_cannot_pass_fails_the_gate(
        self, tmp_path: Path
    ) -> None:
        # Break the world rather than the agent: the approver never answers and
        # no deadline can save it, so the reference cannot legitimately finish.
        text = FIXTURE.read_text(encoding="utf-8").replace(
            "          decision: APPROVED\n", "          decision: REJECTED\n", 1
        )
        broken = tmp_path / "broken.yaml"
        broken.write_text(text, encoding="utf-8")
        code = run("check-maintenance", "--spec", str(broken), "--agent", "reference")
        assert code == EXIT_CONTRACT_FAILED


class TestUsage:
    def test_no_command_is_a_usage_error(self) -> None:
        from operatebench.cli import EXIT_USAGE

        assert main([]) == EXIT_USAGE

    def test_help_lists_all_four_commands(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            main(["--help"])
        out = capsys.readouterr().out
        for command in ("validate", "run-maintenance", "replay", "check-maintenance"):
            assert command in out
