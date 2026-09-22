"""The private five-cell Full-lifecycle diagnostic plan and offline-only CLI."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from operatebench.agents.model import MAX_OUTPUT_TOKENS
from operatebench.agents.pricing import LifecycleCostGuard, LifecyclePricingPolicy
from operatebench.artifact import read_artifact, replay_artifact
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.execution_bundle import (
    BundleIdentityMismatchError,
    audit_execution_bundle,
    audit_execution_bundle_files,
)
from operatebench.execution_ledger import derive_reservation_usd, read_execution_ledger
from tools import full_lifecycle_diagnostic as diagnostic

EXPECTED_CELLS = [
    ("anthropic", "messages", "claude-haiku-4-5-20251001", "implemented_offline"),
    ("anthropic", "messages", "claude-sonnet-5", "implemented_offline"),
    ("openai", "responses", "gpt-5.6-luna", "implemented_offline"),
    ("xai", "chat_completions", "grok-4.5", "implemented_offline"),
    ("mistral", "chat_completions", "mistral-small-2603", "implemented_offline"),
]


def test_plan_is_complete_ordered_unique_private_and_not_live_authorizable() -> None:
    plan = diagnostic.DIAGNOSTIC_PLAN
    cells = plan["cells"]

    assert [
        (c["provider"], c["api"], c["model"], c["adapter_status"]) for c in cells
    ] == EXPECTED_CELLS
    assert len({c["cell_id"] for c in cells}) == len(cells) == 5
    assert len({c["output_path"] for c in cells}) == 5
    assert all(c["episode"] == "maintenance-v1-synthetic" for c in cells)
    assert all(c["lifecycle"] == "full" for c in cells)
    assert plan["status"] == "private_engineering_diagnostic"
    assert plan["plan"] == "full_lifecycle_diagnostic_slice_2b"
    assert plan["live_authorizable"] is False
    assert plan["claim_boundary"] == {
        "integration_and_behavior_evidence_only": True,
        "ranking": False,
        "matched_arm_contrast": False,
        "public_claim": False,
        "model_capability_claim": False,
        "production_inference": False,
        "alpha_replacement": False,
    }
    assert plan["session_policy"] == "fresh_stateless_across_operation_invocations"
    assert plan["controls"]["attempts_per_cell"] == 1
    assert plan["controls"]["sdk_retries"] == 0
    assert plan["controls"]["rates_verified"] is False
    assert plan["controls"]["input_usd_per_mtok"] is None
    assert plan["controls"]["output_usd_per_mtok"] is None
    assert plan["controls"]["max_output_tokens_per_call"] == MAX_OUTPUT_TOKENS == 4096
    sonnet = next(c for c in cells if c["model"] == "claude-sonnet-5")
    assert sonnet["fixed_operator"] == {
        "status": "implemented_offline_verified_not_live_executed",
        "rates_verified_on": "2026-09-07",
        "rate_sources": (
            "https://platform.claude.com/docs/en/models/overview",
            "https://platform.claude.com/docs/en/about-claude/pricing",
        ),
        "input_usd_per_mtok": "2",
        "output_usd_per_mtok": "10",
        "expected_provider_calls": 46,
        "max_provider_calls": 50,
        "max_native_input_token_bound": 25_652,
        "max_output_tokens_per_call": 4_096,
        "expected_call_token_ceiling": 1_368_408,
        "hard_call_token_cap": 1_487_400,
        "exact_worst_case_usd": "4.6132",
        "minimal_cent_cap_usd": "4.62",
    }
    luna = next(c for c in cells if c["model"] == "gpt-5.6-luna")
    assert luna["fixed_operator"] == {
        "status": "implemented_offline_verified_not_live_executed",
        "repeat_ready": True,
        "rates_verified_on": "2026-09-08",
        "rate_sources": (
            "https://developers.openai.com/api/docs/models/gpt-5.6-luna.md",
            "https://developers.openai.com/api/docs/pricing",
        ),
        "model_identifier": "gpt-5.6-luna",
        "request_profile": "gpt56luna_reasoning_none_sampling_omitted_v2",
        "reasoning_effort": "none",
        "sampling_fields_omitted": ("temperature", "top_p"),
        "input_usd_per_mtok": "0.20",
        "output_usd_per_mtok": "1.20",
        "expected_provider_calls": 46,
        "max_provider_calls": 50,
        "max_native_input_token_bound": 27_009,
        "max_output_tokens_per_call": 4_096,
        "expected_call_token_ceiling": 1_430_830,
        "hard_call_token_cap": 1_555_250,
        "exact_worst_case_usd": "0.515850",
        "minimal_cent_cap_usd": "0.52",
    }
    mistral = next(c for c in cells if c["model"] == "mistral-small-2603")
    assert mistral["fixed_operator"] == {
        "status": "implemented_offline_verified_not_live_executed",
        "rates_verified_on": "2026-09-07",
        "rate_sources": (
            "https://docs.mistral.ai/models/mistral-small-4-0-26-03",
            "https://docs.mistral.ai/api",
            "https://docs.mistral.ai/openapi.yaml",
            "https://docs.mistral.ai/inference/pricing",
        ),
        "model_identifier": "mistral-small-2603",
        "moving_alias_not_used": "mistral-small-latest",
        "context_window_provider_text": "256k text",
        "benchmark_output_cap_not_official_ceiling": 4_096,
        "omitted_unresolved_reasoning_field": "reasoning_effort",
        "omitted_service_tier": "Global Standard",
        "input_usd_per_mtok": "0.15",
        "output_usd_per_mtok": "0.60",
        "expected_provider_calls": 46,
        "max_provider_calls": 50,
        "max_native_input_token_bound": 27_099,
        "max_output_tokens_per_call": 4_096,
        "expected_call_token_ceiling": 1_434_970,
        "hard_call_token_cap": 1_559_750,
        "exact_worst_case_usd": "0.32612250",
        "minimal_cent_cap_usd": "0.33",
    }
    assert plan["staged_order"] == (
        "offline_all_cells",
        "one_cell",
        "manual_audit",
        "next_cell",
    )
    assert plan["identities"]["source_base_archive_digest_sha256"] == (
        "26ee847e0304e1a1caba671bcdb1a76078b99f207d24e85719e38018d71d582d"
    )
    assert plan["identities"]["operation_version"] == "0.6.0"
    assert plan["identities"]["operation_spec_digest_sha256"] == (
        "9e99b409b153e03aeb240c6402e14ff21c8bf0df3b9a1df2016c8fa64138e44d"
    )
    assert plan["identities"]["protocol"] == "operatebench.model.v4"
    assert plan["identities"]["artifact"] == 8
    assert plan["identities"]["execution_ledger"] == 3

    with pytest.raises(TypeError):
        plan["controls"]["attempts_per_cell"] = 2
    with pytest.raises(AttributeError):
        cells.append({})


def test_cli_live_refuses_before_environment_client_network_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "must-not-exist"
    touched: list[str] = []

    def forbidden(name: str) -> Any:
        def tripwire(*_args: Any, **_kwargs: Any) -> Any:
            touched.append(name)
            raise AssertionError(f"live refusal reached {name}")

        return tripwire

    monkeypatch.setattr(diagnostic, "run_offline_matrix", forbidden("offline run"))
    monkeypatch.setattr(diagnostic.openai, "OpenAI", forbidden("OpenAI client"))
    monkeypatch.setattr(diagnostic.anthropic, "Anthropic", forbidden("Anthropic client"))
    monkeypatch.setattr(diagnostic, "build_mistral_client", forbidden("Mistral client"))
    monkeypatch.setattr(diagnostic.httpx, "Client", forbidden("HTTP client"))

    assert diagnostic.main(["--live", "--output-dir", str(target)]) == 2

    report = json.loads(capsys.readouterr().out)
    assert report["code"] == "live_execution_not_implemented"
    assert touched == []
    assert not target.exists()


def test_help_identifies_slice_2b_and_the_complete_offline_matrix() -> None:
    help_text = " ".join(diagnostic.build_parser().format_help().split())

    assert "Slice 2B" in help_text
    assert "five" in help_text
    assert "implemented-offline" in help_text
    assert "specified-only" not in help_text


def test_offline_matrix_executes_all_slice2a_cells_with_artifact_ledger_and_replay(
    tmp_path: Path,
) -> None:
    summary = diagnostic.run_offline_matrix(tmp_path)
    spec = load_spec(diagnostic.FIXTURE)

    assert summary["ok"] is True
    assert summary["executed_cells"] == 5
    assert summary["specified_only_cells"] == 0
    for cell in diagnostic.DIAGNOSTIC_PLAN["cells"]:
        directory = tmp_path / cell["output_path"]
        if cell["adapter_status"] == "specified_only":
            assert not directory.exists()
            continue
        artifact_path = directory / "episode_artifact.json"
        ledger_path = directory / "execution_ledger.ndjson"
        artifact = read_artifact(artifact_path)
        binding_schema = json.loads(
            Path("docs/schemas/provider-execution-binding-v1.schema.json").read_text()
        )
        ledger = read_execution_ledger(ledger_path, require_complete=True)
        assert artifact["artifact_version"] == 8
        Draft202012Validator(binding_schema).validate(
            {
                "schema": "operatebench.provider_execution_binding.v1",
                "schema_version": 1,
                **artifact["provider_execution"],
            }
        )
        binding_document = {
            "schema": "operatebench.provider_execution_binding.v1",
            "schema_version": 1,
            **artifact["provider_execution"],
        }
        if cell["provider"] == "mistral":
            assert binding_document["api"] == "chat_completions"
            for wrong_api in ("responses", "messages"):
                forged = {**binding_document, "api": wrong_api}
                with pytest.raises(ValidationError):
                    Draft202012Validator(binding_schema).validate(forged)
        elif cell["provider"] == "xai":
            assert binding_document["api"] == "chat_completions"
            responses_binding = {**binding_document, "api": "responses"}
            Draft202012Validator(binding_schema).validate(responses_binding)
            with pytest.raises(
                BundleIdentityMismatchError, match="binds api='responses'"
            ):
                audit_execution_bundle(
                    {
                        **artifact,
                        "provider_execution": {
                            **artifact["provider_execution"],
                            "api": "responses",
                        },
                    },
                    ledger_path,
                )
            messages_binding = {**binding_document, "api": "messages"}
            with pytest.raises(ValidationError):
                Draft202012Validator(binding_schema).validate(messages_binding)
        assert ledger.status == "scored"
        assert ledger.totals.provider_calls == 46
        assert ledger.totals.attempts == 46
        assert ledger.header.provider.provider == cell["provider"]
        assert ledger.header.provider.model == cell["model"]
        assert type(ledger.header.controls.max_output_tokens) is int
        assert type(ledger.header.provider.settings["max_output_tokens"]) is int
        assert (
            ledger.header.provider.settings["max_output_tokens"]
            == ledger.header.controls.max_output_tokens
            == MAX_OUTPUT_TOKENS
            == 4096
        )
        expected_implementation = {
            "anthropic": "AnthropicMessagesTransport",
            "openai": "OpenAIResponsesTransport",
            "xai": "xai_openai_compat_chat_completions",
            "mistral": "mistral_chat_completions",
        }[cell["provider"]]
        assert ledger.header.provider.implementation == expected_implementation
        assert (
            artifact["agent_execution"]["max_output_tokens"]
            == ledger.header.controls.max_output_tokens
            == MAX_OUTPUT_TOKENS
            == 4096
        )
        assert audit_execution_bundle_files(artifact_path, ledger_path).ok
        replay = replay_artifact(spec, artifact)
        assert replay.ok is True
        assert replay.reproduction == "playback"
        reported = next(
            item for item in summary["cells"] if item["cell_id"] == cell["cell_id"]
        )
        assert reported["execution_ledger_version"] == ledger.ledger_version == 3
        assert reported["replay_provider_calls"] == 0
        expected_sdk = {
            "anthropic": "Anthropic",
            "openai": "OpenAI",
            "xai": "OpenAI",
            "mistral": "Mistral",
        }[cell["provider"]]
        assert reported["transport"] == (
            f"httpx.MockTransport through the real {expected_sdk} SDK"
        )


def test_anthropic_successful_provider_prose_is_not_persisted(tmp_path: Path) -> None:
    cell = diagnostic.DIAGNOSTIC_PLAN["cells"][0]
    spec = load_spec(diagnostic.FIXTURE)
    directory = tmp_path / "anthropic"
    diagnostic._run_offline_cell(spec, cell, directory)

    assert (
        diagnostic._ANTHROPIC_PROSE_SENTINEL
        not in (directory / "execution_ledger.ndjson").read_text()
    )
    assert (
        diagnostic._ANTHROPIC_PROSE_SENTINEL
        not in (directory / "episode_artifact.json").read_text()
    )


def test_xai_successful_provider_prose_is_not_persisted(tmp_path: Path) -> None:
    cell = next(
        item for item in diagnostic.DIAGNOSTIC_PLAN["cells"] if item["provider"] == "xai"
    )
    spec = load_spec(diagnostic.FIXTURE)
    directory = tmp_path / "xai"
    diagnostic._run_offline_cell(spec, cell, directory)

    assert (
        diagnostic._XAI_PROSE_SENTINEL
        not in (directory / "execution_ledger.ndjson").read_text()
    )
    assert (
        diagnostic._XAI_PROSE_SENTINEL
        not in (directory / "episode_artifact.json").read_text()
    )


def test_offline_cell_passes_one_canonical_ceiling_to_guard_and_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, int | None] = {}

    def guard_spy(**kwargs: Any) -> LifecycleCostGuard:
        seen["guard"] = kwargs["max_output_tokens"]
        return LifecycleCostGuard(**kwargs)

    real_agent = diagnostic.EvidenceRecordingModelAgent

    def agent_spy(*args: Any, **kwargs: Any) -> Any:
        seen["agent"] = kwargs.get("max_output_tokens")
        return real_agent(*args, **kwargs)

    monkeypatch.setattr(diagnostic, "LifecycleCostGuard", guard_spy)
    monkeypatch.setattr(diagnostic, "EvidenceRecordingModelAgent", agent_spy)
    spec = load_spec(diagnostic.FIXTURE)
    cell = diagnostic.DIAGNOSTIC_PLAN["cells"][0]

    diagnostic._run_offline_cell(spec, cell, tmp_path / "cell")

    assert seen == {"guard": MAX_OUTPUT_TOKENS, "agent": MAX_OUTPUT_TOKENS}


def test_nonzero_reservation_rederives_from_the_exact_diagnostic_ceiling() -> None:
    policy = LifecyclePricingPolicy(
        policy_id="nonzero_ceiling_probe_v1",
        input_usd_per_mtok=Decimal("1"),
        output_usd_per_mtok=Decimal("2"),
    )
    controls = diagnostic._controls(policy)
    guard = LifecycleCostGuard(
        policy=policy,
        cap_usd=Decimal("1"),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    reservation = guard.authorize(input_tokens_upper_bound=137)

    assert controls.max_output_tokens == MAX_OUTPUT_TOKENS == 4096
    assert reservation > 0
    assert str(reservation) == derive_reservation_usd(
        input_token_upper_bound=137, controls=controls
    )


def test_plan_module_has_no_quarantined_or_boundary_semantic_imports() -> None:
    source = Path(diagnostic.__file__).read_text()
    assert "matched_arm_alpha" not in source
    for forbidden in (
        "boundarybench.environment",
        "boundarybench.evaluator",
        "boundarybench.scaffold",
        "boundarybench.adapter",
    ):
        assert forbidden not in source


def test_shared_lifecycle_authority_is_not_owned_by_provider_named_modules() -> None:
    agents = Path("src/operatebench/agents")
    neutral = (agents / "lifecycle_contract.py").read_text()
    assert "def check_model_request" in neutral
    assert "def outcome_tools" in neutral
    assert "def elide_null_optionals" in neutral
    for module in (
        "openai_responses.py",
        "anthropic_messages.py",
        "xai_chat_completions.py",
        "mistral_chat.py",
    ):
        source = (agents / module).read_text()
        assert "def check_model_request" not in source
        assert "def outcome_tools" not in source
        assert "def elide_null_optionals" not in source
        assert "boundarybench." not in source
        assert "matched_arm_alpha" not in source
