"""Unregistered Finsi pack with executable offline SDK commands."""

from __future__ import annotations

import json

from operatebench._write_once import _write_once_bytes, _WriteOnceFailure
from operatebench.sdk import (
    CheckRequest,
    CommandResult,
    OperationPackError,
    OperationPackMetadata,
    ReplayRequest,
    RunRequest,
    ValidateRequest,
)

from .agents import AGENT_IDS
from .controls import NEGATIVE_CONTROLS
from .runtime import read_record, replay, run
from .spec import OPERATION_TYPE, PACK_ID, REFERENCE_SCENARIOS, load_spec


def write_error(failure: _WriteOnceFailure) -> Exception:
    return OperationPackError(f"run write refused: {failure.reason}")


class FinsiPack:
    metadata = OperationPackMetadata(
        pack_id="partner.finsi.governed_campaign_launch.synthetic",
        operation_type="finsi.governed_campaign_launch",
        operation_id="finsi_governed_campaign_launch_synthetic_v1",
        pack_version="0.1.0",
        display_name="Finsi governed campaign launch",
        owner_id="finsi",
        owner_display_name="Finsi",
        contribution_kind="partner",
        status="incubator",
        privacy_status="SYNTHETIC_ONLY",
        default_spec="examples/contributions/finsi/governed_campaign_launch/operation.yaml",
        agent_ids=(
            "alternate",
            "chat_approval",
            "counts_only",
            "legacy_bypass",
            "new_key_retry",
            "premature_close",
            "reference",
            "unnecessary_approval",
            "unverified",
        ),
        aliases=(),
        evidence_eligible=False,
    )

    def validate(self, request: ValidateRequest) -> CommandResult:
        spec = load_spec(request.spec_path)
        return CommandResult(
            True,
            {
                "pack_id": PACK_ID,
                "operation_type": OPERATION_TYPE,
                "spec_digest": spec.content_digest,
                "scenarios": list(spec.scenarios),
                "registered": False,
            },
        )

    def run(self, request: RunRequest) -> CommandResult:
        record = run(request.spec_path, request.scenario_id, request.agent_id)
        _write_once_bytes(
            (json.dumps(record, sort_keys=True, indent=2) + "\n").encode(),
            request.output_path,
            error_adapter=write_error,
        )
        return CommandResult(record["evaluation"]["reliable"], record)

    def replay(self, request: ReplayRequest) -> CommandResult:
        result = replay(request.spec_path, read_record(request.run_path))
        return CommandResult(True, result)

    def check(self, request: CheckRequest) -> CommandResult:
        selected = request.agent_ids if request.agent_ids is not None else AGENT_IDS
        if not selected or set(selected) - set(AGENT_IDS):
            raise OperationPackError("unknown check agent")
        rows = []
        for agent in ("reference", "alternate"):
            if agent in selected:
                for case in REFERENCE_SCENARIOS:
                    record = run(request.spec_path, case, agent)
                    rows.append(
                        {
                            "agent_id": agent,
                            "scenario_id": case,
                            "passed": record["evaluation"]["reliable"],
                            "decisions": len(record["decisions"]),
                        }
                    )
        for c in NEGATIVE_CONTROLS:
            if c["agent_id"] in selected:
                record = run(request.spec_path, c["scenario_id"], c["agent_id"])
                grade = record["evaluation"]
                passed = {k for k, v in grade["dimensions"].items() if not v} == set(
                    c["failed_dimensions"]
                ) and {f["code"] for f in grade["findings"]} == set(c["finding_codes"])
                rows.append(
                    {
                        "agent_id": c["agent_id"],
                        "scenario_id": c["scenario_id"],
                        "passed": passed,
                        "decisions": len(record["decisions"]),
                    }
                )
        return CommandResult(
            bool(rows) and all(r["passed"] for r in rows),
            {"registered": False, "evidence_eligible": False, "rows": rows},
        )


PACK = FinsiPack()
