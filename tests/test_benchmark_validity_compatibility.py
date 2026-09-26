# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Compatibility controls for the reporting-customer correction."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from operatebench import artifact
from operatebench.agents import (
    anthropic_messages,
    mistral_chat,
    openai_responses,
    xai_chat_completions,
    xai_responses,
)
from operatebench.core.errors import ArtifactError
from operatebench.domains.lettings.maintenance.state import validate_canonical_state
from operatebench.version import OPERATEBENCH_VERSION

ROOT = Path(__file__).resolve().parents[1]


def test_current_and_historical_request_mapping_identities_are_independent() -> None:
    assert OPERATEBENCH_VERSION == "0.13.0"
    assert openai_responses.LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION == (
        "lifecycle_openai_responses_model_request_v9"
    )
    assert openai_responses.LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION_V7 == (
        "lifecycle_openai_responses_model_request_v7"
    )
    assert anthropic_messages.LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION == (
        "lifecycle_anthropic_messages_model_request_v4"
    )
    assert anthropic_messages.LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION_V2 == (
        "lifecycle_anthropic_messages_model_request_v2"
    )
    assert mistral_chat.LIFECYCLE_MISTRAL_REQUEST_MAPPING_VERSION == (
        "lifecycle_mistral_chat_completions_model_request_v3"
    )
    assert xai_chat_completions.LIFECYCLE_XAI_REQUEST_MAPPING_VERSION == (
        "lifecycle_xai_openai_compat_chat_completions_model_request_v3"
    )
    assert xai_responses.LIFECYCLE_XAI_REQUEST_MAPPING_VERSION == (
        "lifecycle_xai_responses_model_request_v6"
    )
    assert xai_responses.LIFECYCLE_XAI_GROK46_REQUEST_MAPPING_VERSION == (
        "lifecycle_xai_responses_model_request_v7"
    )
    assert xai_responses.LIFECYCLE_XAI_REQUEST_MAPPING_VERSION_V1 == (
        "lifecycle_xai_responses_model_request_v1"
    )
    assert xai_responses.LIFECYCLE_XAI_GROK46_REQUEST_MAPPING_VERSION_V2 == (
        "lifecycle_xai_responses_model_request_v2"
    )


def test_historical_state_reads_without_inventing_a_reporting_actor(monkeypatch) -> None:
    path = ROOT / "tests/fixtures/b3/artifact8-deterministic-reference-v1.json"
    before = path.read_bytes()
    record = artifact.read_artifact(path)
    assert record["engine_version"] == "0.9.0"
    assert "issue_reporting_actor_id" not in record["final_state"]
    old_state = copy.deepcopy(record["final_state"])
    validate_canonical_state(old_state, "historical", reporting_actor=False)
    with pytest.raises(ArtifactError, match="issue_reporting_actor_id"):
        validate_canonical_state(old_state, "current")
    old_state["issue_reporting_actor_id"] = "customer_1"
    with pytest.raises(ArtifactError, match="issue_reporting_actor_id"):
        validate_canonical_state(old_state, "historical", reporting_actor=False)
    validate_canonical_state(old_state, "current")

    def forbidden(*args, **kwargs):
        pytest.fail("historical replay reached the current runtime")

    monkeypatch.setattr(artifact, "_reproduce", forbidden)
    from operatebench.domains.lettings.maintenance.spec import load_spec

    spec = load_spec(ROOT / "examples/operatebench/maintenance_v0_1.yaml")
    with pytest.raises(ArtifactError, match="original engine contract"):
        artifact.replay_artifact(spec, record)
    assert path.read_bytes() == before


def test_profile_successor_preserves_every_unmodified_semantic_class() -> None:
    directory = ROOT / "tests/fixtures/maintenance-card-v2"
    histories = {
        "profile-v1.json": (
            "033dec4e19c977c52582760dda96a680009efb503bcf922d61aabe02583d1a09"
        ),
        "profile-v1-semantic-oracle.json": (
            "1b98fde9db10055fe1a62f938159420d7a142302c7e9d97d514e5e329a76bbfb"
        ),
        "source-owner-projection-v1.json": (
            "04976c1fbbe5fbd04a42ff497db240dbfd6b515fa32d660670e2d1d9b1cfe0be"
        ),
    }
    for name, digest in histories.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    for suffix in (".json", "-semantic-oracle.json"):
        old = json.loads((directory / f"profile-v1{suffix}").read_bytes())
        new = json.loads((directory / f"profile-v2{suffix}").read_bytes())
        if suffix != ".json":
            old, new = old["semantic_classes"], new["semantic_classes"]
        assert new["profile_id"] == "maintenance.card_identity_profile@2.0.0"
        assert new["state_policy"]["schema_id"] == "maintenance.state.canonical@2.0.0"
        assert (
            "/issue_reporting_actor_id"
            in new["state_policy"]["canonical_state_projection"]
        )
        new["profile_id"] = old["profile_id"]
        new["state_policy"]["schema_id"] = old["state_policy"]["schema_id"]
        new["state_policy"]["canonical_state_projection"].remove(
            "/issue_reporting_actor_id"
        )
        assert new == old
