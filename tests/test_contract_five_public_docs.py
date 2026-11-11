"""What the published documents have to say about the current artefact contract.

Three documents make version-scoped claims a reader acts on: the manifest note
that says which contract this build writes, the versioning document that says
what moved and what is reproducible, and the dated verification record that says
what has and has not been established since. A contract that moved while those
three still described the one before it is the failure mode these assert
against — not prose style, but a number a reader would have looked for and not
found.

The migration disclosure is held to the same standard. Adding retrieval to the
record does not remove the duplicated facts from what the model is shown, and a
document that let a reader infer otherwise would be claiming a construct result
this build has not measured.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from operatebench.artifact import (
    ARTIFACT_VERSION,
    REPRODUCIBLE_ARTIFACT_VERSIONS,
    SUPPORTED_ARTIFACT_VERSIONS,
)

ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _flowed(relative: str) -> str:
    return " ".join(_read(relative).split())


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(_read("PUBLICATION_MANIFEST.json"))


class TestTheManifestNoteDescribesTheContractItRecords:
    def test_the_recorded_contract_is_the_one_this_build_writes(self, manifest) -> None:
        assert manifest["release"]["artifact_version"] == ARTIFACT_VERSION == 8

    def test_the_note_describes_the_current_contract_and_not_an_older_one(
        self, manifest
    ) -> None:
        note = " ".join(manifest["release"]["artifact_version_note"].split())
        assert "Artefact contract 8" in note
        assert not note.startswith("Artefact contract 4")
        assert not note.startswith("Artefact contract 6")

    def test_the_note_names_the_rows_contract_five_added(self, manifest) -> None:
        note = " ".join(manifest["release"]["artifact_version_note"].split())
        for row in (
            "retrieval_served",
            "retrieval_refused",
            "derived_context_invalidated",
        ):
            assert row in note, f"the manifest note does not name {row}"
        assert "RETRIEVE" in note

    def test_the_note_states_that_only_the_current_contract_reproduces(
        self, manifest
    ) -> None:
        note = " ".join(manifest["release"]["artifact_version_note"].split())
        assert "read" in note
        assert "1 to 7" in note or "1, 2, 3, 4, 5, 6 and 7" in note
        assert REPRODUCIBLE_ARTIFACT_VERSIONS == (ARTIFACT_VERSION,)
        assert SUPPORTED_ARTIFACT_VERSIONS == (1, 2, 3, 4, 5, 6, 7, 8)

    def test_the_note_says_it_is_a_draft_of_the_current_contract(self, manifest) -> None:
        note = " ".join(manifest["release"]["artifact_version_note"].split())
        assert "draft" in note.lower()

    def test_the_note_states_no_provider_telemetry_and_no_live_call(
        self, manifest
    ) -> None:
        note = " ".join(manifest["release"]["artifact_version_note"].split())
        lowered = note.lower()
        assert "telemetry" in lowered
        assert "no live" in lowered or "makes no live" in lowered


class TestTheVersioningDocumentCarriesACurrentContractSection:
    def test_it_has_a_section_for_what_moved_the_current_contract(self) -> None:
        assert "### The artefact contract, and what moved it to 8" in _read(
            "docs/VERSIONING.md"
        )

    def test_the_contract_four_section_is_kept_as_history(self) -> None:
        assert "### What moved the contract to 4" in _read("docs/VERSIONING.md")

    def test_it_states_exactly_what_is_read_and_what_is_reproduced(self) -> None:
        flowed = _flowed("docs/VERSIONING.md")
        assert "contracts 1 to 7 are read and not reproduced" in flowed
        assert "reproduces contract 8 only" in flowed

    def test_it_names_the_three_row_types_and_the_retrieve_decision(self) -> None:
        text = _read("docs/VERSIONING.md")
        for row in (
            "`retrieval_served`",
            "`retrieval_refused`",
            "`derived_context_invalidated`",
        ):
            assert row in text, f"docs/VERSIONING.md does not name {row}"
        assert "`RETRIEVE`" in text

    def test_the_preview_table_history_counts_every_contract_move(self) -> None:
        flowed = _flowed("docs/VERSIONING.md")
        assert "`2`, then to `3`, `4` and `5`" in flowed

    def test_it_states_that_no_provider_telemetry_is_recorded(self) -> None:
        flowed = _flowed("docs/VERSIONING.md").lower()
        assert "provider telemetry" in flowed


class TestTheVerificationRecordDescribesCurrentEvidence:
    def test_it_claims_deterministic_construct_evidence_and_nothing_more(self) -> None:
        flowed = _flowed("PUBLIC_CANDIDATE_VERIFICATION.md")
        assert "deterministic construct evidence" in flowed
        assert "not model evidence" in flowed

    def test_it_scopes_provider_offline_evidence_to_the_exercised_gate(self) -> None:
        flowed = _flowed("PUBLIC_CANDIDATE_VERIFICATION.md").lower()
        assert "this maintenance check made no model or provider call" in flowed
        assert "private provider-backed diagnostics historically occurred" in flowed
        assert "changed-candidate hosted ci: pending" in flowed
        assert (
            "this exact verification matrix made no model or provider call" not in flowed
        )
        assert "no provider was contacted" not in flowed

    def test_it_records_the_current_contract_identity(self) -> None:
        flowed = _flowed("PUBLIC_CANDIDATE_VERIFICATION.md")
        assert "`operatebench.model.v4`" in flowed
        assert "`lifecycle_openai_responses_model_request_v9`" in flowed
        assert "Artefact contract | 8" in flowed
        assert "Execution ledger | 3" in flowed


class TestNoPublishedDocumentContradictsTheCurrentContract:
    @pytest.mark.parametrize(
        "relative",
        [
            "README.md",
            "docs/VERSIONING.md",
            "docs/METHODOLOGY.md",
            "docs/ARCHITECTURE_RFC.md",
            "PUBLIC_CANDIDATE_VERIFICATION.md",
            "PUBLIC_RELEASE_CHECKLIST.md",
        ],
    )
    def test_no_document_calls_an_earlier_contract_the_current_one(
        self, relative: str
    ) -> None:
        flowed = _flowed(relative)
        for stale in (
            "artifact contract is 4",
            "artefact contract is 4",
            "current artefact contract 4",
            "this build writes contract 4",
            "artifact_version` | the artefact contract itself | `4`",
        ):
            assert stale not in flowed, f"{relative} still calls contract 4 current"

    @pytest.mark.parametrize(
        "relative",
        [
            "README.md",
            "docs/VERSIONING.md",
            "docs/METHODOLOGY.md",
            "docs/ARCHITECTURE_RFC.md",
            "PUBLIC_CANDIDATE_VERIFICATION.md",
            "PUBLIC_RELEASE_CHECKLIST.md",
            "PUBLICATION_MANIFEST.json",
        ],
    )
    def test_no_document_claims_the_retrieval_migration_is_finished(
        self, relative: str
    ) -> None:
        lowered = _flowed(relative).lower()
        for overclaim in (
            "construct blocker is closed",
            "construct blocker closed",
            "no longer duplicates",
            "removes the duplicated facts",
            "state duplication is resolved",
        ):
            assert overclaim not in lowered, f"{relative} overclaims the migration"


class TestMistralDocumentationKeepsImplementationBoundaries:
    def test_each_lane_describes_its_own_schema_and_validation(self) -> None:
        from boundarybench.providers import mistral_chat as historical
        from operatebench.providers import mistral_chat as lifecycle

        text = _read("docs/PROVIDER_MATRIX.md")
        historical_text, lifecycle_text = text.split(
            "### Lifecycle/operatebench selected-six Mistral metadata", 1
        )
        historical_text = historical_text.split(
            "### Historical BoundaryBench Mistral: one object on the usage block", 1
        )[1]
        lifecycle_text = lifecycle_text.split("## Pricing provenance", 1)[0]
        historical_text = " ".join(historical_text.split())
        lifecycle_text = " ".join(lifecycle_text.split())

        assert dict(historical.RESPONSE_EXTENSION_SCHEMA) == {
            "prompt_tokens_details": {"cached_tokens": historical.EXTENSION_COUNT}
        }
        assert dict(lifecycle.RESPONSE_EXTENSION_SCHEMA) == {
            "prompt_tokens_details": {"cached_tokens": lifecycle.EXTENSION_COUNT},
            "completion_tokens_details": {"reasoning_tokens": lifecycle.EXTENSION_COUNT},
        }
        for lane, prose in ((historical, historical_text), (lifecycle, lifecycle_text)):
            assert lane.MISTRAL_RESPONSE_EXTENSIONS in prose
            assert lane.__name__.replace(".", "/") + ".py" in prose
            contract = lane.RESPONSE_EXTENSION_CONTRACT
            assert contract["object_members_required"] is True
            assert contract["top_level_members_required"] is False
            assert contract["extension_values_read"] is False
            for block, members in lane.RESPONSE_EXTENSION_SCHEMA.items():
                assert f"`{block}`" in prose
                for member in members:
                    assert f"`{member}`" in prose
            assert "not an invoice" in prose
            assert "additional or unreported fees remain unknown" in prose

        assert "subsets" not in historical.RESPONSE_EXTENSION_CONTRACT
        assert not hasattr(historical, "check_usage_subsets")
        assert "does not check `cached_tokens <= prompt_tokens`" in historical_text
        assert "rejects `completion_tokens_details` entirely" in historical_text
        assert "reasoning_tokens" not in historical_text
        assert "two bounded schema-only diagnostics" in historical_text
        assert "raw bodies were not kept" in historical_text
        assert "actual retained native" not in historical_text
        assert "Read for nothing, and priced into nothing" in historical_text

        subsets = lifecycle.RESPONSE_EXTENSION_CONTRACT["subsets"]
        assert subsets == {
            "prompt_tokens_details.cached_tokens": "prompt_tokens",
            "completion_tokens_details.reasoning_tokens": "completion_tokens",
        }
        for field, total in subsets.items():
            assert f"`{field.split('.')[-1]} <= {total}`" in lifecycle_text
        for claim in (
            "independently optional",
            "exact non-negative bounded integer",
            "exactly its required member",
            "check_response_extensions_agree",
            "no subset double charge and no cache discount",
            "actual retained native responses and offline pinned-SDK replay",
            "not public traces",
            "not an official vendor declaration",
        ):
            assert claim in lifecycle_text
