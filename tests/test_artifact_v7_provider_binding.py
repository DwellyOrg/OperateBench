"""Artefact 7: an episode bound to the execution ledger that witnessed it.

Contract 7 adds exactly one top-level field, ``provider_execution``. It is
``null`` for every run that reached no provider, and it is **required, non-null
and exact** for a run whose decisions came off one. What it carries is a
*binding* — identities, digests, totals and one call index per decision — and
never the ledger itself: the sidecar stays a separate file, and a reader that
wants the ledger's content verified has to open it (see
:mod:`tests.test_execution_bundle_audit`).

Three claims this module holds the build to.

**A model run without a complete scored ledger writes no episode artefact.** Not
a null binding, not an empty one: a refusal. A binding a run could mint for
itself would be a claim about a provider with nothing behind it.

**The validator checks coherence, and says only that.** Identity, cardinality,
the exact contiguous decision-to-call mapping and the request digests against
``agent_execution.attempts`` are all internal to the document. Whether the
ledger says what the binding says it says is not knowable from the artefact
alone, and nothing here claims otherwise.

**Replay opens nothing.** ``replay_artifact`` carries the binding forward beside
the execution record, reproduces the episode from the tape against a forbidden
transport, and makes zero provider calls.

Every provider interaction below runs through the real OpenAI SDK over an
in-process ``httpx.MockTransport``. No credential is read and no socket opened.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from operatebench.agents.model import MODEL_PROTOCOL_VERSION, ModelAgent
from operatebench.agents.openai_responses import (
    LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION,
)
from operatebench.artifact import (
    ARTIFACT_VERSION,
    PROVIDER_EXECUTION_FIELDS,
    REPRODUCIBLE_ARTIFACT_VERSIONS,
    SUPPORTED_ARTIFACT_VERSIONS,
    build_artifact,
    project_top_level,
    read_artifact,
    replay_artifact,
    validate_artifact,
    write_artifact,
)
from operatebench.core.errors import ArtifactError
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.execution_ledger import (
    EXECUTION_LEDGER_VERSION,
    read_execution_ledger,
)
from operatebench.providers.openai_responses import GPT_5_6_LUNA_MODEL
from operatebench.runner import run_episode
from operatebench.version import OPERATEBENCH_VERSION
from tests.model_transport import TEST_MODEL, ReferenceDrivenTransport
from tests.provider_evidence_runs import (
    EXPECTED_PROVIDER_CALLS,
    INPUT_TOKENS,
    MODEL_AGENT_ID,
    OUTPUT_TOKENS,
    model_run_with_ledger,
)

FIXTURE = Path("examples/operatebench/maintenance_v0_1.yaml")
FROZEN_V1 = Path(__file__).resolve().parent / "fixtures/artifact_v1_reference_V1.json"


@pytest.fixture()
def spec() -> Any:
    return load_spec(FIXTURE)


@pytest.fixture()
def ledger_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "evidence"
    directory.mkdir(mode=0o700)
    return directory


@pytest.fixture()
def bound(ledger_dir: Path, spec: Any) -> Any:
    """One scored V1 model run and the journal that witnessed it."""
    return model_run_with_ledger(ledger_dir, spec)


def tampered(payload: dict[str, Any], **changes: Any) -> dict[str, Any]:
    body = json.loads(json.dumps(payload))
    body.update(changes)
    return body


# -- what moved, and what did not --------------------------------------------


class TestTheContractBump:
    def test_artifact_and_lifecycle_request_versions_are_pinned(self) -> None:
        assert ARTIFACT_VERSION == 8
        assert OPERATEBENCH_VERSION == "0.12.0"
        assert MODEL_PROTOCOL_VERSION == "operatebench.model.v4"
        assert LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION == (
            "lifecycle_openai_responses_model_request_v9"
        )

    def test_every_earlier_contract_is_still_read_and_only_seven_reproduced(
        self,
    ) -> None:
        assert SUPPORTED_ARTIFACT_VERSIONS == (1, 2, 3, 4, 5, 6, 7, 8)
        assert REPRODUCIBLE_ARTIFACT_VERSIONS == (8,)

    def test_the_frozen_v1_record_still_reads_under_its_own_contract(self) -> None:
        body = read_artifact(FROZEN_V1)
        assert body["artifact_version"] == 1
        assert "provider_execution" not in body


# -- deterministic runs carry a null binding ---------------------------------


class TestADeterministicRun:
    def test_writes_version_seven_with_a_null_binding(self, spec: Any) -> None:
        artifact = build_artifact(run_episode(spec, "V1", "reference"))
        assert artifact["artifact_version"] == 8
        assert artifact["provider_execution"] is None
        assert validate_artifact(artifact)["provider_execution"] is None

    def test_replays_clean_and_compares_the_binding_it_re_derived(
        self, spec: Any
    ) -> None:
        artifact = build_artifact(run_episode(spec, "V1", "reference"))
        report = replay_artifact(spec, artifact)
        assert report.ok is True
        assert report.reproduction == "rerun"
        assert report.carried_forward == ()
        assert report.sections["provider_execution"]["provider_execution"] is True

    def test_may_not_carry_a_binding_it_could_not_have_produced(
        self, spec: Any, bound: Any
    ) -> None:
        deterministic = build_artifact(run_episode(spec, "V1", "reference"))
        model = build_artifact(bound.run)
        body = tampered(deterministic, provider_execution=model["provider_execution"])
        with pytest.raises(ArtifactError, match="provider_execution"):
            validate_artifact(body)


# -- a model run is bound, exactly -------------------------------------------


class TestAModelRunsBinding:
    def test_states_exactly_the_binding_fields_and_nothing_else(self, bound: Any) -> None:
        artifact = build_artifact(bound.run)
        binding = artifact["provider_execution"]
        assert tuple(sorted(binding)) == tuple(sorted(PROVIDER_EXECUTION_FIELDS))

    def test_every_value_is_the_ledgers_own(self, bound: Any) -> None:
        artifact = build_artifact(bound.run)
        binding = artifact["provider_execution"]
        audit = read_execution_ledger(bound.ledger_path, require_complete=True)
        assert binding["execution_run_id"] == audit.header.execution_run_id
        assert binding["execution_ledger_version"] == EXECUTION_LEDGER_VERSION
        assert binding["execution_ledger_digest_sha256"] == audit.ledger_digest_sha256
        assert binding["provider"] == "openai"
        assert binding["api"] == "responses"
        assert binding["model"] == GPT_5_6_LUNA_MODEL
        assert binding["settings_digest_sha256"] == (
            audit.header.provider.settings_digest_sha256
        )
        assert binding["pricing_digest_sha256"] == (
            audit.header.controls.pricing_digest_sha256
        )
        assert binding["provider_calls"] == EXPECTED_PROVIDER_CALLS
        assert binding["attempts"] == EXPECTED_PROVIDER_CALLS
        assert binding["measured_cost_usd"] == audit.totals.measured_cost_usd
        assert binding["forfeited_reservation_usd"] == "0"
        assert binding["input_tokens"] == EXPECTED_PROVIDER_CALLS * INPUT_TOKENS
        assert binding["output_tokens"] == EXPECTED_PROVIDER_CALLS * OUTPUT_TOKENS

    def test_names_one_call_index_per_recorded_decision_contiguously(
        self, bound: Any
    ) -> None:
        artifact = build_artifact(bound.run)
        binding = artifact["provider_execution"]
        assert binding["decision_call_index"] == list(range(EXPECTED_PROVIDER_CALLS))
        assert len(binding["decision_call_index"]) == len(artifact["decisions"])

    def test_carries_no_credential_no_prose_and_no_ledger_path(
        self, bound: Any, ledger_dir: Path
    ) -> None:
        target = ledger_dir / "artifact.json"
        write_artifact(bound.run, target)
        text = target.read_text(encoding="utf-8")
        assert bound.ledger_path.name not in text
        assert "sk-" not in text
        assert "uthoriz" not in text

    def test_reads_back_and_validates(self, bound: Any, ledger_dir: Path) -> None:
        target = ledger_dir / "artifact.json"
        write_artifact(bound.run, target)
        assert read_artifact(target) == build_artifact(bound.run)


# -- no ledger, no artefact ---------------------------------------------------


class TestAModelRunWithNoLedger:
    def test_is_refused_rather_than_written_with_a_null_binding(self, spec: Any) -> None:
        run = run_episode(
            spec,
            "V1",
            MODEL_AGENT_ID,
            agent_factory=lambda: ModelAgent(
                ReferenceDrivenTransport(),
                model=TEST_MODEL,
                agent_id=MODEL_AGENT_ID,
            ),
            agent_kind="model",
        )
        with pytest.raises(ArtifactError, match="provider_execution"):
            build_artifact(run)

    def test_is_refused_at_the_reader_too(self, bound: Any) -> None:
        body = tampered(build_artifact(bound.run), provider_execution=None)
        with pytest.raises(ArtifactError, match="provider_execution"):
            validate_artifact(body)


# -- the validator's own domain ----------------------------------------------


class TestTheValidatorBindsWhatTheDocumentStates:
    def test_refuses_a_decision_index_of_the_wrong_length(self, bound: Any) -> None:
        artifact = build_artifact(bound.run)
        binding = dict(artifact["provider_execution"])
        binding["decision_call_index"] = binding["decision_call_index"][:-1]
        with pytest.raises(ArtifactError, match="decision_call_index"):
            validate_artifact(tampered(artifact, provider_execution=binding))

    def test_refuses_a_decision_index_that_is_not_contiguous(self, bound: Any) -> None:
        artifact = build_artifact(bound.run)
        binding = dict(artifact["provider_execution"])
        indexes = list(binding["decision_call_index"])
        indexes[3] = indexes[2]
        binding["decision_call_index"] = indexes
        with pytest.raises(ArtifactError, match="decision_call_index"):
            validate_artifact(tampered(artifact, provider_execution=binding))

    def test_refuses_a_call_count_the_attempts_do_not_support(self, bound: Any) -> None:
        artifact = build_artifact(bound.run)
        binding = dict(artifact["provider_execution"])
        binding["provider_calls"] = binding["provider_calls"] + 1
        with pytest.raises(ArtifactError, match="provider call"):
            validate_artifact(tampered(artifact, provider_execution=binding))

    def test_refuses_an_attempt_count_that_is_not_the_recorded_attempts(
        self, bound: Any
    ) -> None:
        artifact = build_artifact(bound.run)
        binding = dict(artifact["provider_execution"])
        binding["attempts"] = binding["attempts"] - 1
        with pytest.raises(ArtifactError, match="attempts"):
            validate_artifact(tampered(artifact, provider_execution=binding))

    def test_refuses_a_model_that_is_not_the_one_the_execution_record_names(
        self, bound: Any
    ) -> None:
        artifact = build_artifact(bound.run)
        binding = dict(artifact["provider_execution"])
        binding["model"] = "gpt-not-the-pinned-model"
        with pytest.raises(ArtifactError, match="model"):
            validate_artifact(tampered(artifact, provider_execution=binding))

    def test_refuses_a_provider_or_api_this_build_does_not_speak(
        self, bound: Any
    ) -> None:
        artifact = build_artifact(bound.run)
        for field, value in (("provider", "not-openai"), ("api", "completions")):
            binding = dict(artifact["provider_execution"])
            binding[field] = value
            with pytest.raises(ArtifactError, match=field):
                validate_artifact(tampered(artifact, provider_execution=binding))

    def test_refuses_a_digest_that_is_not_a_digest(self, bound: Any) -> None:
        artifact = build_artifact(bound.run)
        for field in (
            "execution_ledger_digest_sha256",
            "settings_digest_sha256",
            "pricing_digest_sha256",
        ):
            binding = dict(artifact["provider_execution"])
            binding[field] = "not-a-digest"
            with pytest.raises(ArtifactError, match=field):
                validate_artifact(tampered(artifact, provider_execution=binding))

    def test_refuses_an_execution_run_id_that_is_not_one_this_build_mints(
        self, bound: Any
    ) -> None:
        artifact = build_artifact(bound.run)
        binding = dict(artifact["provider_execution"])
        binding["execution_run_id"] = "a" * 64
        with pytest.raises(ArtifactError, match="execution_run_id"):
            validate_artifact(tampered(artifact, provider_execution=binding))

    def test_refuses_an_amount_that_is_not_an_exact_decimal_string(
        self, bound: Any
    ) -> None:
        artifact = build_artifact(bound.run)
        binding = dict(artifact["provider_execution"])
        binding["measured_cost_usd"] = 0.2964125
        with pytest.raises(ArtifactError, match="measured_cost_usd"):
            validate_artifact(tampered(artifact, provider_execution=binding))

    def test_refuses_a_ledger_version_this_build_does_not_write(self, bound: Any) -> None:
        artifact = build_artifact(bound.run)
        binding = dict(artifact["provider_execution"])
        binding["execution_ledger_version"] = EXECUTION_LEDGER_VERSION + 1
        with pytest.raises(ArtifactError, match="execution_ledger_version"):
            validate_artifact(tampered(artifact, provider_execution=binding))

    def test_refuses_an_unknown_or_missing_binding_field(self, bound: Any) -> None:
        artifact = build_artifact(bound.run)
        extra = dict(artifact["provider_execution"], raw_ledger=[{"row": 1}])
        with pytest.raises(ArtifactError, match="raw_ledger"):
            validate_artifact(tampered(artifact, provider_execution=extra))
        short = dict(artifact["provider_execution"])
        short.pop("pricing_digest_sha256")
        with pytest.raises(ArtifactError, match="pricing_digest_sha256"):
            validate_artifact(tampered(artifact, provider_execution=short))


# -- projection to contract 6 -------------------------------------------------


class TestProjectingOntoContractSix:
    def test_removes_the_binding_and_nothing_else(self, bound: Any) -> None:
        artifact = build_artifact(bound.run)
        projected = project_top_level(artifact, 6)
        assert "provider_execution" not in projected
        assert projected == {
            name: value
            for name, value in artifact.items()
            if name != "provider_execution"
        }

    def test_the_projection_is_a_readable_contract_six_document(self, spec: Any) -> None:
        artifact = build_artifact(run_episode(spec, "V1", "reference"))
        projected = project_top_level(artifact, 6)
        projected["artifact_version"] = 6
        assert validate_artifact(projected)["artifact_version"] == 6

    def test_a_contract_six_document_carrying_a_binding_is_refused(
        self, spec: Any, bound: Any
    ) -> None:
        artifact = build_artifact(run_episode(spec, "V1", "reference"))
        projected = project_top_level(artifact, 6)
        projected["artifact_version"] = 6
        projected["provider_execution"] = build_artifact(bound.run)["provider_execution"]
        with pytest.raises(ArtifactError, match="provider_execution"):
            validate_artifact(projected)

    def test_a_contract_seven_document_missing_the_field_is_refused(
        self, spec: Any
    ) -> None:
        artifact = build_artifact(run_episode(spec, "V1", "reference"))
        artifact.pop("provider_execution")
        with pytest.raises(ArtifactError, match="provider_execution"):
            validate_artifact(artifact)


# -- replay opens nothing -----------------------------------------------------


class TestReplayCarriesTheBindingAndReachesNothing:
    def test_a_model_artefact_replays_at_zero_provider_calls(
        self, bound: Any, spec: Any
    ) -> None:
        artifact = build_artifact(bound.run)
        before = len(bound.wire.bodies)
        report = replay_artifact(spec, artifact)
        assert report.ok is True
        assert report.reproduction == "playback"
        assert "provider_execution" in report.carried_forward
        assert len(bound.wire.bodies) == before == EXPECTED_PROVIDER_CALLS

    def test_the_replay_never_opens_the_sidecar(self, bound: Any, spec: Any) -> None:
        artifact = build_artifact(bound.run)
        bound.ledger_path.unlink()
        report = replay_artifact(spec, artifact)
        assert report.ok is True

    def test_a_contract_six_record_is_read_but_not_replayed(self, spec: Any) -> None:
        artifact = build_artifact(run_episode(spec, "V1", "reference"))
        projected = project_top_level(artifact, 6)
        projected["artifact_version"] = 6
        assert validate_artifact(projected)["artifact_version"] == 6
        with pytest.raises(ArtifactError, match="contract"):
            replay_artifact(spec, projected)
