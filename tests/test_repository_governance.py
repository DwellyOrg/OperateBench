"""Repository governance and publication-claim boundaries."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = "@khanukov"
REQUIRED_OWNER_PATTERNS = {
    "/src/operatebench/",
    "/tools/",
    "/.github/",
    "/docs/METHODOLOGY.md",
    "/docs/REPOSITORY_GOVERNANCE.md",
    "/PUBLICATION_MANIFEST.json",
    "/pyproject.toml",
    "/uv.lock",
    "/REUSE.toml",
}
MANDATORY_OWNED_FILES = {
    ".github/CODEOWNERS",
    ".github/pull_request_template.md",
    ".github/workflows/ci.yml",
    ".github/workflows/dco.yml",
    "docs/METHODOLOGY.md",
    "docs/REPOSITORY_GOVERNANCE.md",
    "PUBLICATION_MANIFEST.json",
    "pyproject.toml",
    "uv.lock",
    "REUSE.toml",
    "src/operatebench/runner.py",
    "src/operatebench/execution_ledger.py",
    "src/operatebench/providers/executor.py",
    "src/operatebench/agents/openai_responses.py",
    "tools/check_public_release.py",
    "tools/check_dco.py",
}


def _flowed(path: str) -> str:
    return " ".join((ROOT / path).read_text(encoding="utf-8").split())


def _codeowner_rules() -> list[tuple[str, tuple[str, ...]]]:
    rules: list[tuple[str, tuple[str, ...]]] = []
    for raw in (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pattern, *owners = line.split()
        assert owners, f"CODEOWNERS rule has no owner: {raw!r}"
        assert not pattern.startswith("!"), "CODEOWNERS does not support negation"
        assert "[" not in pattern and "]" not in pattern
        rules.append((pattern, tuple(owners)))
    assert rules, "CODEOWNERS must contain at least one ownership rule"
    return rules


def _matches(pattern: str, repository_path: str) -> bool:
    """Match the exact-file/directory GitHub syntax this policy permits."""
    assert pattern.startswith("/"), (
        f"sensitive ownership must be root-anchored: {pattern}"
    )
    assert not any(character in pattern for character in "*?\\"), (
        f"policy CODEOWNERS rules must be exact files or directories: {pattern}"
    )
    anchored = pattern.removeprefix("/")
    if anchored.endswith("/"):
        return repository_path.startswith(anchored)
    return repository_path == anchored


def _owners_for(repository_path: str) -> tuple[str, ...]:
    owners: tuple[str, ...] = ()
    for pattern, candidate_owners in _codeowner_rules():
        if _matches(pattern, repository_path):
            owners = candidate_owners
    return owners


def test_codeowners_covers_current_and_future_sensitive_paths() -> None:
    current_source_and_tools = {
        path.relative_to(ROOT).as_posix()
        for root in (ROOT / "src" / "operatebench", ROOT / "tools")
        for path in root.rglob("*")
        if path.is_file()
    }
    current_github_files = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / ".github").rglob("*")
        if path.is_file()
    }
    sensitive = current_source_and_tools | current_github_files | MANDATORY_OWNED_FILES

    assert current_source_and_tools
    assert current_github_files
    assert all(OWNER in _owners_for(path) for path in sensitive)


def test_codeowners_uses_broad_root_anchored_rules() -> None:
    patterns = {pattern for pattern, owners in _codeowner_rules() if OWNER in owners}
    assert patterns >= REQUIRED_OWNER_PATTERNS


def test_readme_states_the_bounded_non_claim_and_execution_boundary() -> None:
    readme = _flowed("README.md")
    assert (
        "Not built: any release-grade matched-control model evaluation or official "
        "Lifecycle benchmark model result." in readme
    )
    assert "documented public/offline benchmark commands make no provider calls" in readme
    assert "Historical full44 observations in the results page" in readme
    assert "not release-grade benchmark evidence" in readme
    assert "a descriptive full44 aggregate comparison of 27 exact IDs" in readme
    assert "not a controlled ranking" in readme
    assert "Provider cost is not part of the benchmark result vector" in readme


ABSOLUTE_PROVIDER_CLAIMS = (
    "no run makes a provider call",
    "no run in this build makes a provider call",
    "nothing in this build calls a provider",
    "the preview runs no provider calls",
)


def test_public_docs_do_not_overstate_the_provider_call_boundary() -> None:
    for name in (
        "README.md",
        "docs/METHODOLOGY.md",
        "docs/ARCHITECTURE_RFC.md",
        "SECURITY.md",
    ):
        lowered = _flowed(name).lower()
        for stale in ABSOLUTE_PROVIDER_CLAIMS:
            assert stale not in lowered, f"{name} retains absolute claim: {stale}"

    methodology = _flowed("docs/METHODOLOGY.md")
    assert "documented public/offline benchmark commands" in methodology
    assert "make no provider calls" in methodology
    assert "Historical full44 observations in the results page" in methodology
    assert "rather than release-grade benchmark evidence" in methodology
    assert "Provider cost is not part of the benchmark result vector" in methodology
    assert "cannot establish current capability" in methodology

    security = _flowed("SECURITY.md")
    assert (
        "documented public/offline benchmark commands make no provider calls" in security
    )


def test_manifest_excludes_private_diagnostics_and_capability_authority() -> None:
    manifest = json.loads((ROOT / "PUBLICATION_MANIFEST.json").read_text())
    declarations = manifest["declarations"]
    assert declarations["no_published_benchmark_model_results"]["value"] is True
    assert declarations["no_published_benchmark_model_results"]["scope"] == (
        "official_benchmark_results"
    )
    assert declarations["no_published_live_provider_evidence"]["value"] is True
    assert "private_lifecycle_diagnostics" not in declarations
    assert "approvals" not in manifest
    assert manifest["published_artifacts"] == []
    for key in (
        "assigned_episodes",
        "integrity_valid_scored_episodes",
        "preregistered_provider_budget_exclusions",
        "successful_completions",
        "interface_commit",
        "repository_baseline_commit",
    ):
        assert key not in json.dumps(manifest)


def test_governance_names_stable_aggregate_checks_without_claiming_ruleset_state() -> (
    None
):
    governance = _flowed("docs/REPOSITORY_GOVERNANCE.md")
    governance_lower = governance.lower()
    required = {
        "test (py3.11)",
        "test (py3.14)",
        "build distribution",
        "dco",
        "independent secret scan",
        "dependency-review",
    }
    assert all(f"`{context}`" in governance for context in required)
    assert "Ruff" in governance
    assert "mypy" in governance
    assert "REUSE" in governance
    assert "causal" in governance
    assert "public scanner" in governance
    assert "Twine" in governance
    for phrase in (
        "transitively",
        "not separately require-able",
        "forge-side configuration",
        "does not prove that a ruleset is enabled",
        "one independent approval",
        "stale approvals",
        "resolved",
        "force-push",
        "named emergency role",
        "signed annotated tags",
        "exact-candidate verification",
        "separate authorization",
    ):
        assert phrase in governance_lower


def test_ruleset_activation_is_blocked_until_both_distinct_roles_are_verified() -> None:
    governance = _flowed("docs/REPOSITORY_GOVERNANCE.md")
    assert (
        "The approval-, CODEOWNERS-, and bypass-dependent ruleset MUST NOT be "
        "activated while only `@khanukov` exists and no distinct emergency role "
        "exists." in governance
    )
    assert (
        "Activation prerequisites are both: (a) at least one distinct reviewer or "
        "team with repository access and a verified approval path; and (b) a named, "
        "distinct emergency team or custom role with a verified bypass path."
        in governance
    )
    assert (
        "Until both prerequisites exist, the independent-approval rule is deferred; "
        "existing branch protections and required checks remain in force." in governance
    )


def test_descriptive_exception_does_not_authorize_official_or_human_pass() -> None:
    manifest = json.loads((ROOT / "PUBLICATION_MANIFEST.json").read_text())
    assert manifest["manifest_schema_version"] == "0.3.0"
    declarations = manifest["declarations"]
    report = declarations["descriptive_model_results"]
    assert report["value"] is True
    assert report["path"] == "docs/MODEL_RESULTS.md"
    assert report["scope"] == "owner_directed_retrospective_descriptive_only"
    assert report["human_semantic_validation"] == "pending"
    assert report["official_release_authorized"] is False
    assert declarations["no_leaderboard_or_ranking"]["value"] is True
    statement = report["statement"]
    for boundary in (
        "27 exact IDs",
        "including Gemini",
        "two original scored-case rates",
        "six additional models have one observation",
        "44 related synthetic cases",
        "technical exclusions",
        "scored denominators",
        "Raw traces remain private",
        "not a controlled ranking",
        "not independent approval",
        "No official Lifecycle Track leaderboard",
    ):
        assert boundary in statement
    gates = {gate["id"]: gate for gate in manifest["gates"]}
    for key in (
        "G8_claims_scan",
        "G11_independent_human_gate",
        "G12_explicit_publication_approval",
    ):
        assert gates[key]["status"] == "pending"
        assert gates[key]["blocking"] is True
    assert manifest["candidate_acceptance"]["status"] == "requires_exact_source_record"
    assert "original prepublication blanket results ban" in " ".join(manifest["notes"])
    excluded = {item["class"]: item for item in manifest["excluded_classes"]}
    assert (
        "explicit descriptive-table exception"
        in excluded["unvalidated_results"]["description"]
    )
    assert "Raw provider traces" in excluded["provider_run_artifacts"]["reason"]
    checklist = _flowed("PUBLIC_RELEASE_CHECKLIST.md")
    assert "The independently labelled, owner-directed retrospective" in checklist
    assert "This exception does not close G8, G11 or G12" in checklist
    assert "Apart from the specific descriptive-table exception in section 6" in checklist
    assert (
        "Any leaderboard, ranking, composite capability score, or comparison"
        not in checklist
    )


def test_observation_study_exception_has_exact_bounded_artifacts() -> None:
    manifest = json.loads((ROOT / "PUBLICATION_MANIFEST.json").read_text())
    report = manifest["declarations"]["descriptive_model_results"]
    statement = report["statement"]
    expected = {
        "docs/studies/2026-10-selected-six.json",
        "docs/studies/2026-10-model-averages.json",
        "docs/studies/2026-10-observations.md",
        "docs/studies/2026-10-aggregates.json",
        "docs/studies/2026-10-case-churn.svg",
        "docs/studies/2026-10-goal-strict.svg",
        "docs/studies/2026-10-models.csv",
        "docs/studies/2026-10-model-flows.csv",
        "docs/studies/2026-10-compliance-diagnostic.csv",
    }
    import re

    assert (
        set(re.findall(r"docs/studies/[\w-]+\.(?:md|csv|json|svg)", statement))
        == expected
    )
    assert all((ROOT / path).is_file() for path in expected)
    assert report["path"] == "docs/MODEL_RESULTS.md"
    assert report["human_semantic_validation"] == "pending"
    assert report["official_release_authorized"] is False
    assert (
        "does not authorize future results, raw traces or billing disclosure" in statement
    )


def test_actual_aggregate_matches_original_rates_and_common_list() -> None:
    import csv
    import re
    from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
    from fractions import Fraction
    from hashlib import sha256

    from tools.check_public_release import PUBLIC_AGGREGATE_SHA256

    packet_path = "docs/studies/2026-10-selected-six.json"
    means_path = "docs/studies/2026-10-model-averages.json"
    packet_bytes = (ROOT / packet_path).read_bytes()
    assert sha256(packet_bytes).hexdigest() == (
        "271070a082cbf3b868d61e0501203295d7cbaa364e0ac15f8fc98a268e97ca5e"
    )
    for path in (packet_path, means_path):
        assert (
            sha256((ROOT / path).read_bytes()).hexdigest()
            == PUBLIC_AGGREGATE_SHA256[path]
        )
    packet = json.loads(packet_bytes)
    means = json.loads((ROOT / means_path).read_bytes())
    provenance = means["selected_six_provenance"]
    assert provenance["source_revision"] == packet["source_revision"]
    assert provenance["evaluator"] == packet["evaluator"] == "Compliance-0.3.0"
    assert provenance["conditions"] == packet["conditions"]
    assert provenance["sha256"] == sha256(packet_bytes).hexdigest()
    expected = {
        "gpt-6.1-terra": (44, 44, 38, 0, 0, "none", "responses"),
        "claude-haiku-5-5": (13, 0, 0, 31, 0, "adaptive-low", "messages"),
        "mistral-large-4": (24, 0, 0, 20, 0, "none", "native-chat"),
        "grok-4.7": (15, 15, 5, 29, 0, "low", "responses"),
        "GLM-5.3": (28, 25, 5, 12, 4, "low", "http"),
        "moonshotai/Kimi-K3": (38, 16, 0, 6, 0, "none", "http"),
    }
    with (ROOT / "docs/studies/2026-10-models.csv").open() as stream:
        original = list(csv.DictReader(stream))
    assert len(original) == 21
    rates = {}
    coverage = {}
    for row in original:
        model = row["model"]
        rates[model] = tuple(
            (
                Fraction(int(row[f"old_{key}"]), int(row["old_scored"]))
                + Fraction(int(row[f"new_{key}"]), int(row["new_scored"]))
            )
            / 2
            for key in ("strict", "goals")
        )
        coverage[model] = (int(row["old_scored"]) + int(row["new_scored"]), 88)
    assert set(expected).isdisjoint(rates)
    assert {m["model"] for m in packet["models"]} == set(expected)
    for m in packet["models"]:
        assert (
            tuple(
                m[k]
                for k in (
                    "scored",
                    "goal_completed",
                    "reliable",
                    "excluded",
                    "aborted",
                    "reasoning",
                    "endpoint",
                )
            )
            == expected[m["model"]]
        )
        assert m["assigned"] == 44
        assert m["observation_count"] == 1
        assert m["cost_usd"] is None
        rates[m["model"]] = (
            Fraction(m["reliable"], m["scored"]),
            Fraction(m["goal_completed"], m["scored"]),
        )
        coverage[m["model"]] = (m["scored"], 44)
    order = sorted(rates, key=lambda m: (-rates[m][0], -rates[m][1], m.casefold(), m))
    assert len(order) == 27
    assert [r["model"] for r in means["models"]] == order
    table = re.findall(
        r"^\| `([^`]+)` \| ([^|]+) \| ([^|]+) \| ([^|]+) \| Scored (\d+)/(\d+) \|$",
        (ROOT / "docs/MODEL_RESULTS.md").read_text(),
        re.MULTILINE,
    )
    assert [r[0] for r in table] == order
    for row, cells in zip(means["models"], table, strict=True):
        model = row["model"]
        assert row["observation_count"] == (1 if model in expected else 2)
        assert (
            tuple(Fraction(**row[key]) for key in ("quality_rate", "goal_rate"))
            == rates[model]
        )
        for key, value, text in zip(
            ("average_quality_percent", "average_goal_achieved_percent"),
            rates[model],
            cells[1:3],
            strict=True,
        ):
            percent = (
                str(
                    (Decimal(value.numerator) * 100 / value.denominator).quantize(
                        Decimal(".1"), rounding=ROUND_HALF_UP
                    )
                )
                + "%"
            )
            assert row[key] == text == percent
        assert tuple(map(int, cells[4:])) == coverage[model]
        if model in expected:
            assert row["average_full_run_estimate"] is None
            assert cells[3] == "Unknown"
        else:
            cost = row["average_full_run_estimate"]
            low = Decimal(cost["mean_cost_low_usd"])
            high = Decimal(cost["mean_cost_high_usd"])
            rounding = (
                ROUND_HALF_UP
                if low == high and not cost["incomplete_usage"]
                else ROUND_FLOOR
                if cost["incomplete_usage"]
                else ROUND_CEILING
            )
            amount = high.quantize(Decimal(".01"), rounding=rounding)
            assert cells[3] == f"${amount:.2f}"
    assert sum(m["scored"] for m in packet["models"]) == 162
    assert sum(m["excluded"] for m in packet["models"]) == 98
    assert sum(m["aborted"] for m in packet["models"]) == 4


def test_seven_historical_study_artifacts_remain_byte_exact() -> None:
    from hashlib import sha256

    expected = {
        "docs/studies/2026-10-observations.md": (
            "8401f192af67662053ade2b164d00058bb29ffaae35c2e5d27dc12899761c0a6"
        ),
        "docs/studies/2026-10-aggregates.json": (
            "1c2c919f08c476b915fed334ad18f8c79a6caaabba56be982a8f4718dbb66a73"
        ),
        "docs/studies/2026-10-case-churn.svg": (
            "593ce92b4a808cf6411e57b31bc55687f646909fc52bb51d1330e5bb712702c7"
        ),
        "docs/studies/2026-10-goal-strict.svg": (
            "baf86d0bb6e83586fa0dd5131f0afae7707cfc4873eb4087964c70e52a8dcfad"
        ),
        "docs/studies/2026-10-models.csv": (
            "abfe654b2f19d46954cde0c58b955d0532fb75d7b0fa896024c9fe64f9dc0114"
        ),
        "docs/studies/2026-10-model-flows.csv": (
            "c88ebb4bc108b2e055d349f87876d583232e43786967c3535b42528da9c1d189"
        ),
        "docs/studies/2026-10-compliance-diagnostic.csv": (
            "ec89f2307db3e16d7106b02bfa0657f404f69d0deab35185cb49b49fb8514138"
        ),
    }
    for path, digest in expected.items():
        assert sha256((ROOT / path).read_bytes()).hexdigest() == digest
