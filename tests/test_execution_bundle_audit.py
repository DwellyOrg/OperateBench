"""The bundle audit: an artefact, its sidecar, and whether they are one run.

The artefact's own validator can say a binding is *coherent* — the right shape,
the right cardinality, the right internal mapping. It cannot say the ledger
holds what the binding says it holds, because the ledger is not in the document.
:func:`~operatebench.execution_bundle.audit_execution_bundle` is the only thing
in this build that answers that question, and this module is what holds it to
answering it exactly.

Every refusal here is a *named* one. A bundle that disagrees in the run identity
is a different failure from one that disagrees in the totals, and an operator
reading the exception has to be able to tell which of the two happened without
re-deriving it. The adversarial half below edits one field at a time, re-chains
the journal where the chain would otherwise catch the edit first, and requires
the audit to name what it caught.

Nothing here reaches a provider.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from operatebench.artifact import build_artifact, write_artifact
from operatebench.execution_bundle import (
    BundleCallBindingError,
    BundleDigestMismatchError,
    BundleIdentityMismatchError,
    BundleLedgerStateError,
    BundleLedgerVersionMismatchError,
    BundleTotalsMismatchError,
    ExecutionBundleAudit,
    ExecutionBundleError,
    audit_execution_bundle,
    audit_execution_bundle_files,
)
from operatebench.execution_ledger import read_execution_ledger
from operatebench.providers.openai_responses import GPT_5_6_LUNA_MODEL
from operatebench.runner import run_episode
from tests.execution_ledger_fixtures import rechain_ledger_file
from tests.provider_evidence_runs import (
    EXPECTED_PROVIDER_CALLS,
    model_run_with_ledger,
)

FIXTURE = Path("examples/operatebench/maintenance_v0_1.yaml")


@pytest.fixture()
def spec() -> Any:
    from operatebench.domains.lettings.maintenance.spec import load_spec

    return load_spec(FIXTURE)


@pytest.fixture()
def ledger_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "evidence"
    directory.mkdir(mode=0o700)
    return directory


@pytest.fixture()
def bound(ledger_dir: Path, spec: Any) -> Any:
    return model_run_with_ledger(ledger_dir, spec)


def edited(path: Path, mutate: Any) -> None:
    """Apply one edit to the journal and re-chain every row after it.

    Re-chaining is the point. Editing a row and leaving the chain broken is
    refused by the ledger reader before the audit ever sees it, which would
    prove nothing about the audit; re-chaining leaves a journal that reads
    perfectly and disagrees with the artefact, which is the attack.
    """
    rechain_ledger_file(path, mutate)


# -- the passing bundle -------------------------------------------------------


class TestAMatchedBundle:
    def test_passes_and_reports_what_it_verified(self, bound: Any) -> None:
        artifact = build_artifact(bound.run)
        audit = audit_execution_bundle(artifact, bound.ledger_path)
        assert isinstance(audit, ExecutionBundleAudit)
        assert audit.ok is True
        assert audit.provider == "openai"
        assert audit.model == GPT_5_6_LUNA_MODEL
        assert audit.provider_calls == EXPECTED_PROVIDER_CALLS
        assert audit.attempts == EXPECTED_PROVIDER_CALLS
        assert audit.decision_call_index == tuple(range(EXPECTED_PROVIDER_CALLS))

    def test_binds_the_run_identity_and_the_ledger_digest(self, bound: Any) -> None:
        artifact = build_artifact(bound.run)
        ledger = read_execution_ledger(bound.ledger_path, require_complete=True)
        audit = audit_execution_bundle(artifact, bound.ledger_path)
        assert ledger.ledger_version == 3
        assert audit.execution_ledger_version == ledger.ledger_version
        assert audit.execution_run_id == ledger.header.execution_run_id
        assert audit.execution_ledger_digest_sha256 == ledger.ledger_digest_sha256

    def test_states_plainly_that_a_digest_is_not_an_attestation(self, bound: Any) -> None:
        artifact = build_artifact(bound.run)
        audit = audit_execution_bundle(artifact, bound.ledger_path)
        summary = audit.summary()
        assert summary["ok"] is True
        assert summary["provider_attested"] is False
        assert summary["execution_ledger_version"] == 3

    def test_reads_both_documents_off_disk(self, bound: Any, ledger_dir: Path) -> None:
        target = ledger_dir / "artifact.json"
        write_artifact(bound.run, target)
        audit = audit_execution_bundle_files(target, bound.ledger_path)
        assert audit.ok is True


# -- the ledger has to be a terminal, scored one ------------------------------


class TestTheLedgerMustBeComplete:
    def test_an_incomplete_journal_is_refused_by_name(self, bound: Any) -> None:
        artifact = build_artifact(bound.run)
        lines = bound.ledger_path.read_text(encoding="utf-8").splitlines()
        bound.ledger_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        with pytest.raises(BundleLedgerStateError):
            audit_execution_bundle(artifact, bound.ledger_path)

    def test_an_excluded_journal_is_refused_by_name(
        self, ledger_dir: Path, spec: Any
    ) -> None:
        import httpx

        scored = model_run_with_ledger(ledger_dir, spec, ledger_name="scored.ndjson")
        faulted = model_run_with_ledger(
            ledger_dir,
            spec,
            ledger_name="excluded.ndjson",
            faults={4: httpx.Response(500, json={"error": {"message": "mock"}})},
        )
        assert faulted.failure is not None
        with pytest.raises(BundleLedgerStateError):
            audit_execution_bundle(build_artifact(scored.run), faulted.ledger_path)


# -- an artefact with no binding has no bundle --------------------------------


class TestAnArtefactWithNoBinding:
    def test_a_deterministic_record_is_refused_by_name(
        self, spec: Any, bound: Any
    ) -> None:
        artifact = build_artifact(run_episode(spec, "V1", "reference"))
        with pytest.raises(ExecutionBundleError):
            audit_execution_bundle(artifact, bound.ledger_path)


# -- one edit at a time -------------------------------------------------------


class TestEveryEditIsNamed:
    def test_a_v2_binding_is_refused_for_a_genuine_v3_ledger(self, bound: Any) -> None:
        artifact = build_artifact(bound.run)
        binding = dict(artifact["provider_execution"])
        binding["execution_ledger_version"] = 2
        artifact["provider_execution"] = binding

        with pytest.raises(
            BundleLedgerVersionMismatchError,
            match=(
                "execution ledger contract does not match the verified sidecar contract"
            ),
        ) as refusal:
            audit_execution_bundle(artifact, bound.ledger_path)
        assert refusal.type is BundleLedgerVersionMismatchError

    def test_a_v3_binding_is_refused_for_a_genuine_v2_ledger_via_files(
        self, tmp_path: Path
    ) -> None:
        fixture_dir = Path("tests/fixtures/b3")
        artifact = json.loads(
            (fixture_dir / "artifact8-fake-openai-reference-v1.json").read_text(
                encoding="utf-8"
            )
        )
        binding = dict(artifact["provider_execution"])
        binding["execution_ledger_version"] = 3
        artifact["provider_execution"] = binding
        artifact_path = tmp_path / "coherent-artifact8-with-v3-binding.json"
        artifact_path.write_text(
            json.dumps(
                artifact,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

        with pytest.raises(
            BundleLedgerVersionMismatchError,
            match=(
                "execution ledger contract does not match the verified sidecar contract"
            ),
        ) as refusal:
            audit_execution_bundle_files(
                artifact_path,
                fixture_dir
                / "artifact8-fake-openai-reference-v1.execution-ledger-v2.ndjson",
            )
        assert refusal.type is BundleLedgerVersionMismatchError

    def test_a_swapped_binding_digest_is_refused(self, bound: Any) -> None:
        artifact = build_artifact(bound.run)
        binding = dict(artifact["provider_execution"])
        binding["execution_ledger_digest_sha256"] = "f" * 64
        artifact["provider_execution"] = binding
        with pytest.raises(BundleDigestMismatchError):
            audit_execution_bundle(artifact, bound.ledger_path)

    def test_an_edited_ledger_row_breaks_the_digest_binding(self, bound: Any) -> None:
        artifact = build_artifact(bound.run)

        def mutate(rows: list[dict[str, Any]]) -> None:
            rows[1]["call"]["input_token_upper_bound"] += 1

        edited(bound.ledger_path, mutate)
        with pytest.raises(ExecutionBundleError):
            audit_execution_bundle(artifact, bound.ledger_path)

    def test_a_different_run_identity_is_refused(self, bound: Any) -> None:
        artifact = build_artifact(bound.run)
        binding = dict(artifact["provider_execution"])
        binding["execution_run_id"] = "exec_" + "0" * 32
        artifact["provider_execution"] = binding
        with pytest.raises(BundleIdentityMismatchError):
            audit_execution_bundle(artifact, bound.ledger_path)

    def test_a_settings_digest_that_is_not_the_ledgers_is_refused(
        self, bound: Any
    ) -> None:
        artifact = build_artifact(bound.run)
        binding = dict(artifact["provider_execution"])
        binding["settings_digest_sha256"] = "a" * 64
        artifact["provider_execution"] = binding
        with pytest.raises(BundleIdentityMismatchError):
            audit_execution_bundle(artifact, bound.ledger_path)

    def test_a_pricing_digest_that_is_not_the_ledgers_is_refused(
        self, bound: Any
    ) -> None:
        artifact = build_artifact(bound.run)
        binding = dict(artifact["provider_execution"])
        binding["pricing_digest_sha256"] = "b" * 64
        artifact["provider_execution"] = binding
        with pytest.raises(BundleIdentityMismatchError):
            audit_execution_bundle(artifact, bound.ledger_path)

    def test_an_artifact_ceiling_that_is_not_the_ledgers_is_refused(
        self, bound: Any
    ) -> None:
        artifact = build_artifact(bound.run)
        execution = dict(artifact["agent_execution"])
        execution["max_output_tokens"] = execution["max_output_tokens"] // 2
        artifact["agent_execution"] = execution

        with pytest.raises(BundleIdentityMismatchError, match="max_output_tokens"):
            audit_execution_bundle(artifact, bound.ledger_path)

    def test_totals_that_are_not_the_ledgers_are_refused(self, bound: Any) -> None:
        for name, value in (
            ("input_tokens", 1),
            ("output_tokens", 1),
            ("measured_cost_usd", "0.01"),
            ("forfeited_reservation_usd", "0.01"),
        ):
            artifact = build_artifact(bound.run)
            binding = dict(artifact["provider_execution"])
            binding[name] = value
            artifact["provider_execution"] = binding
            with pytest.raises(BundleTotalsMismatchError):
                audit_execution_bundle(artifact, bound.ledger_path)

    def test_a_call_index_that_names_another_call_is_refused(self, bound: Any) -> None:
        artifact = build_artifact(bound.run)
        binding = dict(artifact["provider_execution"])
        indexes = list(binding["decision_call_index"])
        indexes[5], indexes[6] = indexes[6], indexes[5]
        binding["decision_call_index"] = indexes
        artifact["provider_execution"] = binding
        with pytest.raises((BundleCallBindingError, ExecutionBundleError)):
            audit_execution_bundle(artifact, bound.ledger_path)

    def test_an_edited_request_digest_on_a_ledger_row_is_refused(
        self, bound: Any
    ) -> None:
        artifact = build_artifact(bound.run)

        def mutate(rows: list[dict[str, Any]]) -> None:
            rows[3]["call"]["request_digest_sha256"] = "c" * 64

        edited(bound.ledger_path, mutate)
        with pytest.raises(BundleCallBindingError):
            audit_execution_bundle(artifact, bound.ledger_path)

    def test_an_edited_decision_digest_on_a_ledger_row_is_refused(
        self, bound: Any
    ) -> None:
        artifact = build_artifact(bound.run)

        def mutate(rows: list[dict[str, Any]]) -> None:
            rows[3]["call"]["decision"]["decision_digest_sha256"] = "d" * 64

        edited(bound.ledger_path, mutate)
        with pytest.raises(BundleCallBindingError):
            audit_execution_bundle(artifact, bound.ledger_path)

    def test_a_decision_with_no_bound_answer_is_refused(self, bound: Any) -> None:
        """The binding a row states between its answer and its decision.

        Removing the normalized digest leaves a decision recorded against an
        answer nothing on the row ties it to. The audit never reaches its own
        per-call check here: the ledger reader refuses the row first, and the
        audit reports that as the bundle failure it is rather than letting an
        ``ExecutionLedgerError`` escape a bundle surface.
        """
        artifact = build_artifact(bound.run)

        def mutate(rows: list[dict[str, Any]]) -> None:
            attempts = rows[4]["call"]["attempts"]
            attempts[-1]["response_normalized_digest_sha256"] = None

        edited(bound.ledger_path, mutate)
        with pytest.raises(ExecutionBundleError):
            audit_execution_bundle(artifact, bound.ledger_path)

    def test_a_ledger_naming_another_model_is_refused(self, bound: Any) -> None:
        artifact = build_artifact(bound.run)

        def mutate(rows: list[dict[str, Any]]) -> None:
            rows[0]["header"]["provider"]["model"] = "gpt-not-the-pinned-model"

        edited(bound.ledger_path, mutate)
        with pytest.raises(BundleIdentityMismatchError):
            audit_execution_bundle(artifact, bound.ledger_path)


# -- a whole other run's ledger ----------------------------------------------


class TestAnotherRunsLedger:
    def test_a_second_valid_ledger_does_not_satisfy_the_first_artefact(
        self, ledger_dir: Path, spec: Any
    ) -> None:
        first = model_run_with_ledger(ledger_dir, spec, ledger_name="first.ndjson")
        second = model_run_with_ledger(ledger_dir, spec, ledger_name="second.ndjson")
        assert audit_execution_bundle(build_artifact(second.run), second.ledger_path).ok
        with pytest.raises(BundleIdentityMismatchError):
            audit_execution_bundle(build_artifact(first.run), second.ledger_path)

    def test_the_swap_is_caught_even_when_both_are_scored_and_complete(
        self, ledger_dir: Path, spec: Any
    ) -> None:
        first = model_run_with_ledger(ledger_dir, spec, ledger_name="a.ndjson")
        second = model_run_with_ledger(ledger_dir, spec, ledger_name="b.ndjson")
        for path in (first.ledger_path, second.ledger_path):
            assert read_execution_ledger(path, require_complete=True).scored is True
        with pytest.raises(ExecutionBundleError):
            audit_execution_bundle(build_artifact(second.run), first.ledger_path)


def test_the_audit_never_writes_anything(bound: Any, ledger_dir: Path) -> None:
    artifact = build_artifact(bound.run)
    before = sorted(p.name for p in ledger_dir.iterdir())
    audit_execution_bundle(artifact, bound.ledger_path)
    assert sorted(p.name for p in ledger_dir.iterdir()) == before


def test_the_ledger_is_still_exactly_the_bytes_it_was(bound: Any) -> None:
    artifact = build_artifact(bound.run)
    before = bound.ledger_path.read_bytes()
    audit_execution_bundle(artifact, bound.ledger_path)
    assert bound.ledger_path.read_bytes() == before
    assert json.loads(before.splitlines()[0])["row_kind"] == "header"
