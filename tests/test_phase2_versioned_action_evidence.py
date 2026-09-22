# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Phase-2 identities move without redefining historical contracts."""

from __future__ import annotations

import ast
import json
import re
from copy import deepcopy
from pathlib import Path

import pytest

from operatebench.agents.model import (
    MODEL_PROTOCOL_VERSION,
    MODEL_PROTOCOL_VERSION_V3,
)
from operatebench.agents.openai_responses import (
    LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION,
    LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION_V6,
)
from operatebench.artifact import (
    ARTIFACT_VERSION,
    ARTIFACT_VERSION_V7,
    REPRODUCIBLE_ARTIFACT_VERSIONS,
    SUPPORTED_ARTIFACT_VERSIONS,
    build_artifact,
    validate_artifact,
)
from operatebench.core.errors import ArtifactError
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.runner import run_episode
from operatebench.version import OPERATEBENCH_VERSION

FIXTURE = Path("examples/operatebench/maintenance_v0_1.yaml")


def test_phase2_identity_matrix_moves_only_semantic_surfaces() -> None:
    spec = load_spec(FIXTURE)
    assert OPERATEBENCH_VERSION == "0.12.0"
    assert spec.operation_version == "0.6.0"
    assert MODEL_PROTOCOL_VERSION == "operatebench.model.v4"
    assert LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION == (
        "lifecycle_openai_responses_model_request_v9"
    )
    assert ARTIFACT_VERSION == 8
    assert SUPPORTED_ARTIFACT_VERSIONS == (1, 2, 3, 4, 5, 6, 7, 8)
    assert REPRODUCIBLE_ARTIFACT_VERSIONS == (8,)


def test_historical_protocol_and_mapping_identities_are_literal_and_distinct() -> None:
    assert MODEL_PROTOCOL_VERSION_V3 == "operatebench.model.v3"
    assert MODEL_PROTOCOL_VERSION_V3 != MODEL_PROTOCOL_VERSION
    assert LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION_V6 == (
        "lifecycle_openai_responses_model_request_v6"
    )
    assert (
        LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION_V6
        != LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION
    )
    for path, names in (
        (
            Path("src/operatebench/agents/model.py"),
            {"MODEL_PROTOCOL_VERSION", "MODEL_PROTOCOL_VERSION_V3"},
        ),
        (
            Path("src/operatebench/agents/openai_responses.py"),
            {
                "LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION",
                "LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION_V6",
            },
        ),
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assignments = {
            target.id: node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id in names
        }
        assert assignments.keys() == names
        assert all(isinstance(value, ast.Constant) for value in assignments.values())


def test_artifact_seven_is_a_literal_historical_identity_not_the_live_alias() -> None:
    assert ARTIFACT_VERSION_V7 == 7
    assert ARTIFACT_VERSION_V7 != ARTIFACT_VERSION
    tree = ast.parse(Path("src/operatebench/artifact.py").read_text(encoding="utf-8"))
    assignments = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id in {"ARTIFACT_VERSION", "ARTIFACT_VERSION_V7"}
    }
    assert all(isinstance(value, ast.Constant) for value in assignments.values())


@pytest.mark.parametrize(
    "v8_owned_code",
    [
        "REQUIRED_EVIDENCE_REF_NOT_CITED",
        "REQUIRED_EVIDENCE_SELECTOR_UNRESOLVED",
    ],
)
def test_v7_reader_rejects_v8_owned_evidence_semantics(v8_owned_code: str) -> None:
    current = build_artifact(run_episode(load_spec(FIXTURE), "V1", "reference"))
    historical = deepcopy(current)
    historical["artifact_version"] = 7
    finding = historical["evaluation"]["dimensions"][-1]["findings"]
    finding.append(
        {
            "code": v8_owned_code,
            "detail": "v8-owned semantic finding",
            "at": None,
        }
    )
    historical["evaluation"]["finding_codes"].append(v8_owned_code)
    historical["evaluation"]["dimensions"][-1]["ok"] = False
    historical["evaluation"]["failed_dimensions"].append("retrieval_discipline")
    historical["evaluation"]["reliable"] = False
    with pytest.raises(ArtifactError, match=rf"contract 7|{v8_owned_code}"):
        validate_artifact(historical)


def test_publication_truth_records_private_diagnostic_without_capability_claim() -> None:
    manifest = json.loads(Path("PUBLICATION_MANIFEST.json").read_text(encoding="utf-8"))
    declarations = manifest["declarations"]
    assert "no_live_provider_calls" not in declarations
    assert declarations["no_published_live_provider_evidence"]["value"] is True
    assert declarations["no_published_benchmark_model_results"]["value"] is True
    assert "private_lifecycle_diagnostics" not in declarations
    assert manifest["published_artifacts"] == []
    assert "approvals" not in manifest
    for field in (
        "assigned_episodes",
        "integrity_valid_scored_episodes",
        "preregistered_provider_budget_exclusions",
        "successful_completions",
    ):
        assert field not in json.dumps(manifest)


def test_public_docs_retire_absolute_no_provider_and_no_result_claims() -> None:
    current_docs = [
        Path("README.md"),
        Path("docs/METHODOLOGY.md"),
        Path("docs/RELATED_WORK.md"),
        Path("docs/VERSIONING.md"),
        Path("CITATION.cff"),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in current_docs)
    assert "no published benchmark model results" in text.lower()
    assert not re.search(r"\bno model results\b", text, flags=re.IGNORECASE)
    assert not re.search(r"\bno live provider evidence\b", text, flags=re.IGNORECASE)
    assert "no live provider call has been made" not in text.lower()
    related = " ".join(
        Path("docs/RELATED_WORK.md").read_text(encoding="utf-8").lower().split()
    )
    assert "no run makes a provider call" not in related
    assert "provider cost is not in it and is not measured anywhere" not in related
    assert "no public benchmark artifact reports" in related
    assert "not benchmark capability or public evidence" in related


def test_execution_ledger_comment_states_the_literal_shipped_artifact_boundary() -> None:
    source = Path("src/operatebench/execution_ledger.py").read_text(encoding="utf-8")
    assert "binding itself is not shipped yet" not in source
    assert "binding has shipped since artifact 7" in source
    assert "schema is independently versioned" in source
    assert "intended artifact 8" in source
