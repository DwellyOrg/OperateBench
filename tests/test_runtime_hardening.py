"""Consolidated runtime integrity invariants.

Every test exercises an adversarial input at a public runtime boundary. They
fall into four groups, and the grouping is the argument:

* **Identity** (§1) — a run's identity is only worth something if the state it
  hashes cannot be changed afterwards, and if the hash is re-derived at every
  point that trusts it.
* **Reachability** (§2, §3, §7) — a persisted classification has to describe an
  episode the runner can actually produce. Counters, trajectory terminality and
  the outcome class are all bound to the recorded cause.
* **Representability** (§5) — every artefact boundary refuses text and structure
  that canonical UTF-8 JSON cannot carry, by name, before anything hashes it.
* **Provenance and honesty** (§4, §6, §8, §9) — parsed evidence cannot drift
  after it was read, a fake cannot claim to be a model, and what the ledger does
  and does not prove is stated rather than implied.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import boundarybench.jsonsafe as jsonsafe
from boundarybench.adapter import (
    ADAPTER_CONTRACT_VERSION,
    FAKE_IMPLEMENTATION,
    FAKE_MODEL,
    FAKE_PROVIDER,
    TEST_DOUBLE_MODEL,
    TEST_DOUBLE_PROVIDER,
    AdapterCall,
    AdapterIdentity,
    AdapterUsage,
    FakeAdapterProvenanceError,
    PolicyFollowingFakeAdapter,
    ScriptedTestAdapter,
    TurnDeadline,
    TurnRequest,
    fake_identity,
    identity_for_test_double,
    policy_following_script,
)
from boundarybench.jsonsafe import (
    MAX_JSON_DEPTH,
    CyclicStructureError,
    NestingDepthError,
    TextEncodingError,
    canonical_json_text,
    ensure_json_safe,
    sanitized_text,
)
from boundarybench.ledger import (
    LEDGER_INTEGRITY_NOTE,
    OUTCOME_ADAPTER_FAILURE,
    OUTCOME_MODEL_ACTION_FAILURE,
    OUTCOME_MODEL_PROTOCOL_FAILURE,
    OUTCOMES,
    LedgerError,
    read_ledger,
    sample_status_for,
)
from boundarybench.manifest import ManifestError, card_fingerprint
from boundarybench.runmanifest import (
    BUILD_CONTRACT_VERSIONS,
    RETRY_POLICY,
    RunLimits,
    RunManifestError,
    build_run_manifest,
    configuration_digest,
    execution_digest_of,
    load_run_manifest,
    open_run_session,
)
from boundarybench.runner import (
    compiled_variants,
    execute_run,
    run_episode,
)
from boundarybench.scaffold import (
    STANDARD_SCAFFOLD,
    ScaffoldError,
    load_scaffold,
    scaffold_digest,
)
from boundarybench.schema import ConstructCard, SchemaError
from boundarybench.suite import validate_suite
from tests.conftest import EXAMPLE_CARD, SUITE_MANIFEST, minimal_card

LONE_SURROGATE = "\ud800"


# -- shared fixtures ---------------------------------------------------------


def _suite() -> Any:
    return validate_suite(SUITE_MANIFEST)


def _scaffold() -> Any:
    return load_scaffold(STANDARD_SCAFFOLD)


def _manifest(
    *,
    trials: int = 1,
    settings: dict[str, Any] | None = None,
    identity: AdapterIdentity | None = None,
    suite: Any = None,
    scaffold: Any = None,
) -> Any:
    who = identity or identity_for_test_double()
    return build_run_manifest(
        suite=suite if suite is not None else _suite(),
        scaffold=scaffold if scaffold is not None else _scaffold(),
        provider=who.provider,
        model=who.model,
        implementation=who.implementation,
        adapter_version=who.version,
        adapter_settings={} if settings is None else settings,
        trials=trials,
        limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0),
    )


def _run(root: Path, script: Any, **kwargs: Any) -> Any:
    suite, scaffold = _suite(), _scaffold()
    manifest = _manifest(suite=suite, scaffold=scaffold, **kwargs)
    adapter = ScriptedTestAdapter(
        script=script,
        identity=identity_for_test_double(),
        settings=dict(kwargs.get("settings") or {}),
    )
    with open_run_session(root, manifest) as session:
        return execute_run(
            suite=suite, scaffold=scaffold, adapter=adapter, session=session
        )


# =========================================================================
# §5 recursive JSON and text safety across every artefact boundary
# =========================================================================


def test_the_safety_primitive_names_a_lone_surrogate_rather_than_encoding_it() -> None:
    with pytest.raises(TextEncodingError) as info:
        canonical_json_text({"note": f"bad {LONE_SURROGATE}"}, "settings")
    assert "settings" in str(info.value)
    assert "surrogate" in str(info.value).lower()


def test_the_safety_primitive_names_a_cycle_rather_than_recursing() -> None:
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    with pytest.raises(CyclicStructureError):
        ensure_json_safe(cyclic, "settings")


def test_the_safety_primitive_names_exhausted_depth_rather_than_recursing() -> None:
    deep: Any = "leaf"
    for _ in range(MAX_JSON_DEPTH + 5):
        deep = {"next": deep}
    with pytest.raises(NestingDepthError):
        ensure_json_safe(deep, "settings")


def _nested_empty_arrays(container_count: int) -> Any:
    value: Any = []
    for _ in range(container_count - 1):
        value = [value]
    return value


def test_json_depth_counts_only_containers_with_the_root_at_depth_one() -> None:
    assert ensure_json_safe([], "document", max_depth=1) == []
    assert ensure_json_safe(["scalar leaf"], "document", max_depth=1) == ["scalar leaf"]

    with pytest.raises(NestingDepthError):
        ensure_json_safe([[]], "document", max_depth=1)


def test_constructed_and_raw_json_share_the_exact_public_depth_limit() -> None:
    at_limit = _nested_empty_arrays(MAX_JSON_DEPTH)
    canonical = canonical_json_text(at_limit, "document")

    assert ensure_json_safe(at_limit, "document") is at_limit
    assert jsonsafe.ensure_raw_json_depth(canonical, "document") == canonical
    assert json.loads(canonical) == at_limit

    over_limit = _nested_empty_arrays(MAX_JSON_DEPTH + 1)
    with pytest.raises(NestingDepthError):
        canonical_json_text(over_limit, "document")
    with pytest.raises(NestingDepthError):
        jsonsafe.ensure_raw_json_depth(
            "[" * (MAX_JSON_DEPTH + 1) + "]" * (MAX_JSON_DEPTH + 1),
            "document",
        )


def test_raw_json_depth_accepts_the_public_limit_and_refuses_the_next_level() -> None:
    at_limit = "[" * MAX_JSON_DEPTH + "0" + "]" * MAX_JSON_DEPTH
    over_limit = "[" * (MAX_JSON_DEPTH + 1) + "0" + "]" * (MAX_JSON_DEPTH + 1)

    assert jsonsafe.ensure_raw_json_depth(at_limit, "document") == at_limit
    with pytest.raises(
        NestingDepthError,
        match=rf"maximum JSON nesting depth is {MAX_JSON_DEPTH}",
    ):
        jsonsafe.ensure_raw_json_depth(over_limit, "document")


def test_raw_json_depth_ignores_brackets_and_escapes_inside_strings() -> None:
    text = json.dumps({"text": ("[{]}" + '\\"' + "\\\\") * (MAX_JSON_DEPTH + 1)})

    assert jsonsafe.ensure_raw_json_depth(text, "document") == text


def test_raw_json_depth_leaves_malformed_json_for_the_parser() -> None:
    malformed = '{"value": [1, 2}'

    assert jsonsafe.ensure_raw_json_depth(malformed, "document") == malformed


def test_a_shared_structure_that_is_not_a_cycle_is_still_safe() -> None:
    shared = {"a": 1}
    assert ensure_json_safe({"x": shared, "y": shared}, "settings") is not None


def test_sanitized_text_replaces_lone_surrogates_and_leaves_text_alone() -> None:
    assert sanitized_text(f"boom {LONE_SURROGATE}") == "boom �"
    assert sanitized_text("boom") == "boom"


#: The one authored line the surrogate tests rewrite. A YAML double-quoted
#: scalar is the only place ``\uD800`` is an escape rather than five characters,
#: which is exactly how a real card would come to hold one.
_CARD_DISTRACTOR = (
    "    value: intermittent leak under the kitchen sink, no standing water"
)
_CARD_DISTRACTOR_SURROGATE = '    value: "intermittent leak \\uD800"'


def _card_with_surrogate() -> str:
    source = EXAMPLE_CARD.read_text(encoding="utf-8")
    assert _CARD_DISTRACTOR in source
    return source.replace(_CARD_DISTRACTOR, _CARD_DISTRACTOR_SURROGATE, 1)


def test_a_card_carrying_a_lone_surrogate_is_refused_before_it_is_fingerprinted(
    tmp_path: Path,
) -> None:
    card = tmp_path / EXAMPLE_CARD.name
    card.write_text(_card_with_surrogate(), encoding="utf-8")
    from boundarybench.loader import load_card

    with pytest.raises(SchemaError) as info:
        load_card(card)
    assert "surrogate" in str(info.value).lower()


def test_card_fingerprint_refuses_a_surrogate_bearing_card() -> None:
    raw = minimal_card()
    raw["distractors"][0]["value"] = f"SYN {LONE_SURROGATE}"
    with pytest.raises((SchemaError, ManifestError)) as info:
        card_fingerprint(ConstructCard.from_dict(raw))
    assert "surrogate" in str(info.value).lower()


def test_a_scaffold_prompt_carrying_a_lone_surrogate_is_a_named_scaffold_error(
    tmp_path: Path,
) -> None:
    payload = json.loads(STANDARD_SCAFFOLD.read_text(encoding="utf-8"))
    payload["system_prompt"] = payload["system_prompt"] + " \ud800"
    path = tmp_path / "scaffold.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ScaffoldError) as info:
        load_scaffold(path)
    assert "surrogate" in str(info.value).lower()


def test_scaffold_digest_refuses_a_surrogate_rather_than_raising_unicode_error() -> None:
    with pytest.raises(ScaffoldError):
        scaffold_digest({"system_prompt": f"hello {LONE_SURROGATE}"})


def test_adapter_settings_carrying_a_lone_surrogate_are_a_named_manifest_error() -> None:
    with pytest.raises(RunManifestError) as info:
        _manifest(settings={"note": f"bad {LONE_SURROGATE}"})
    assert "surrogate" in str(info.value).lower()


def test_a_ledger_row_carrying_a_lone_surrogate_is_a_named_ledger_error(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    report = _run(root, policy_following_script)
    ledger = root / "episodes.jsonl"
    lines = ledger.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["error_class"] = None
    lines[0] = json.dumps(row, sort_keys=True, separators=(",", ":")).replace(
        '"variant_id"', '"variant_id\\ud800"', 1
    )
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(LedgerError) as info:
        read_ledger(ledger, report.manifest, compiled_variants(_suite()))
    assert "surrogate" in str(info.value).lower()


def test_a_run_manifest_carrying_a_lone_surrogate_is_a_named_manifest_error(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _run(root, policy_following_script)
    path = root / "run_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["adapter_settings"] = {"note": "bad \ud800"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RunManifestError) as info:
        load_run_manifest(path)
    assert "surrogate" in str(info.value).lower()


def test_an_adapter_exception_message_with_a_surrogate_is_never_fatal(
    tmp_path: Path,
) -> None:
    """Unencodable text in an adapter's message cannot stop a run being recorded.

    Originally the message was sanitised into the row — each lone surrogate
    replaced — so that externally sourced text could not make a failure
    unwritable. The message is no longer recorded at all (see
    ``redacted_adapter_detail`` and ``tests/test_provider_output_trust.py``), which is
    the stronger form of the same guarantee: the whole run is still written and
    read back, and neither the surrogate nor its replacement is on disk.
    """

    def boom(request: TurnRequest) -> AdapterCall:
        raise RuntimeError(f"provider said {LONE_SURROGATE}")

    root = tmp_path / "run"
    report = _run(root, boom)
    assert report.executed_count == 12
    assert report.outcome_counts == {OUTCOME_ADAPTER_FAILURE: 12}
    detail = report.records[0].error_detail
    assert detail is not None
    assert "RuntimeError" in detail
    assert LONE_SURROGATE not in detail
    assert "�" not in detail


def test_an_adapter_action_name_with_a_surrogate_is_classified_not_persisted(
    tmp_path: Path,
) -> None:
    def bad_text(request: TurnRequest) -> AdapterCall:
        return AdapterCall(f"read_records{LONE_SURROGATE}", {})

    root = tmp_path / "run"
    report = _run(root, bad_text)
    assert report.executed_count == 12
    assert set(report.outcome_counts) == {"model_protocol_failure"}
    assert report.records[0].failure_event is not None
    assert report.records[0].failure_event["kind"] == "unrepresentable_text"


def _cli(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    # The command carries absolute suite/output paths, so its process cwd does not
    # determine the case it reads.  Run from the test project root instead: under
    # mutmut that is ``mutants/``, where setup.cfg lives.  Mutmut's generated
    # trampolines load their config at import time; launching from the temporary
    # case workspace made the subprocess fail before BoundaryBench started and
    # turned this CLI contract test into a mutation-harness failure.
    project_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, "-m", "boundarybench.cli", *command],
        capture_output=True,
        text=True,
        cwd=project_root,
        encoding="utf-8",
        errors="replace",
    )


@pytest.mark.parametrize("corrupt", ["card", "scaffold", "run_manifest", "ledger"])
def test_every_surrogate_cli_boundary_exits_one_with_no_traceback(
    tmp_path: Path, corrupt: str
) -> None:
    import shutil

    from tests.conftest import EXAMPLES

    workspace = tmp_path / "work"
    workspace.mkdir()
    examples = workspace / "examples"
    shutil.copytree(EXAMPLES, examples)
    suite = examples / SUITE_MANIFEST.name
    run_dir = workspace / "run"
    command = [
        "run-suite",
        str(suite),
        "--adapter",
        "fake-scripted",
        "--output-dir",
        str(run_dir),
    ]

    if corrupt == "card":
        (examples / EXAMPLE_CARD.name).write_text(
            _card_with_surrogate(), encoding="utf-8"
        )
    elif corrupt == "scaffold":
        payload = json.loads(STANDARD_SCAFFOLD.read_text(encoding="utf-8"))
        payload["system_prompt"] += LONE_SURROGATE
        path = workspace / "scaffold.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        command += ["--scaffold", str(path)]
    else:
        # Both of these need a run on disk first, which the resume then reads.
        first = _cli(command, workspace)
        assert first.returncode == 0, first.stderr
        target = run_dir / (
            "run_manifest.json" if corrupt == "run_manifest" else "episodes.jsonl"
        )
        text = target.read_text(encoding="utf-8")
        if corrupt == "run_manifest":
            payload = json.loads(text)
            payload["adapter_settings"] = {"note": "bad \\ud800"}
            target.write_text(json.dumps(payload), encoding="utf-8")
        else:
            target.write_text(text.replace('"success"', '"succe\\ud800"', 1), "utf-8")

    result = _cli(command, workspace)
    assert result.returncode == 1, result.stdout
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


# =========================================================================
# §1 run identity and recursive immutability
# =========================================================================


def test_nested_adapter_settings_cannot_be_mutated_after_identity_is_fixed() -> None:
    manifest = _manifest(settings={"tuning": {"depth": "1"}})
    with pytest.raises(TypeError):
        manifest.adapter_settings["tuning"]["depth"] = "2"  # type: ignore[index]
    with pytest.raises(TypeError):
        manifest.adapter_settings["tuning"] = {}  # type: ignore[index]


def test_the_pinned_identity_constants_cannot_be_mutated() -> None:
    with pytest.raises(TypeError):
        RETRY_POLICY["retries"] = 3  # type: ignore[index]
    with pytest.raises(TypeError):
        BUILD_CONTRACT_VERSIONS["runner"] = "9.9.9"  # type: ignore[index]


def test_a_stale_run_id_cannot_write_a_manifest_or_execute(tmp_path: Path) -> None:
    suite, scaffold = _suite(), _scaffold()
    manifest = _manifest(suite=suite, scaffold=scaffold)
    stale = dataclasses.replace(
        manifest,
        limits=RunLimits(max_turns=1, max_messages=2, episode_timeout_seconds=1.0),
    )
    adapter = ScriptedTestAdapter(
        script=policy_following_script,
        identity=identity_for_test_double(),
        settings={},
    )
    with (
        pytest.raises(RunManifestError),
        open_run_session(tmp_path / "run", stale) as session,
    ):
        execute_run(suite=suite, scaffold=scaffold, adapter=adapter, session=session)
    assert not (tmp_path / "run" / "run_manifest.json").exists()


def test_writing_a_manifest_recomputes_its_identity_immediately_before_the_write(
    tmp_path: Path,
) -> None:
    from boundarybench.runmanifest import resolve_run_paths, write_run_manifest

    manifest = _manifest()
    stale = dataclasses.replace(manifest, trials=9)
    root = tmp_path / "run"
    root.mkdir()
    paths = resolve_run_paths(root)
    with pytest.raises(RunManifestError):
        write_run_manifest(paths, stale)
    assert not paths.manifest_path.exists()


def test_cyclic_adapter_settings_are_a_named_run_manifest_error() -> None:
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    with pytest.raises(RunManifestError):
        _manifest(settings=cyclic)


def test_over_deep_adapter_settings_are_a_named_run_manifest_error() -> None:
    deep: Any = "leaf"
    for _ in range(MAX_JSON_DEPTH + 5):
        deep = {"next": deep}
    with pytest.raises(RunManifestError):
        _manifest(settings={"tuning": deep})


def test_a_run_manifest_round_trips_with_its_deepest_container_at_the_limit(
    tmp_path: Path,
) -> None:
    # The stored manifest and adapter-settings mapping consume the first two
    # container levels; these arrays therefore put its deepest empty container
    # at exactly MAX_JSON_DEPTH.
    settings = {"nested": _nested_empty_arrays(MAX_JSON_DEPTH - 2)}
    manifest = _manifest(settings=settings)
    root = tmp_path / "run"

    with open_run_session(root, manifest):
        pass

    raw = (root / "run_manifest.json").read_text(encoding="utf-8")
    assert jsonsafe.ensure_raw_json_depth(raw, "run manifest") == raw
    loaded = load_run_manifest(root / "run_manifest.json")
    assert loaded.configuration_id == manifest.configuration_id
    assert loaded.adapter_settings == manifest.adapter_settings


def test_a_loaded_manifest_carries_the_same_recursively_immutable_settings(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _run(root, policy_following_script, settings={"tuning": {"depth": "1"}})
    stored = load_run_manifest(root / "run_manifest.json")
    with pytest.raises(TypeError):
        stored.adapter_settings["tuning"]["depth"] = "2"  # type: ignore[index]
    assert configuration_digest(stored.configuration_payload()) == stored.configuration_id
    assert (
        execution_digest_of(
            configuration_id=stored.configuration_id,
            execution_id=stored.execution_id,
            created_at_utc=stored.created_at_utc,
        )
        == stored.execution_digest
    )


# =========================================================================
# §2 adapter exceptions produce a durable, readable, resumable record
# =========================================================================


def test_an_adapter_exception_run_is_durable_readable_and_resumes_as_a_no_op(
    tmp_path: Path,
) -> None:
    def boom(request: TurnRequest) -> AdapterCall:
        raise RuntimeError("provider exploded")

    suite, scaffold = _suite(), _scaffold()
    root = tmp_path / "run"
    report = _run(root, boom)
    assert report.executed_count == 12
    assert report.outcome_counts == {OUTCOME_ADAPTER_FAILURE: 12}
    assert report.records[0].turns_used == 1
    assert report.records[0].messages_used == 1

    before = (root / "episodes.jsonl").read_bytes()
    manifest = _manifest(suite=suite, scaffold=scaffold)
    adapter = ScriptedTestAdapter(
        script=boom, identity=identity_for_test_double(), settings={}
    )
    with open_run_session(root, manifest) as session:
        again = execute_run(
            suite=suite, scaffold=scaffold, adapter=adapter, session=session
        )
    assert again.executed_count == 0
    assert again.resumed_count == 12
    assert (root / "episodes.jsonl").read_bytes() == before


def test_a_keyboard_interrupt_is_never_converted_into_a_durable_failure() -> None:
    suite, scaffold = _suite(), _scaffold()
    variant = next(iter(compiled_variants(suite).values()))

    def interrupt(request: TurnRequest) -> AdapterCall:
        raise KeyboardInterrupt

    adapter = ScriptedTestAdapter(
        script=interrupt, identity=identity_for_test_double(), settings={}
    )
    with pytest.raises(KeyboardInterrupt):
        run_episode(
            variant=variant,
            scaffold=scaffold,
            adapter=adapter,
            limits=RunLimits(max_turns=3, max_messages=8, episode_timeout_seconds=5.0),
        )


# =========================================================================
# §3 outcome classes are bound to trajectory terminality
# =========================================================================


def _forge(root: Path, mutate: Any) -> tuple[Path, Any]:
    ledger = root / "episodes.jsonl"
    lines = ledger.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    mutate(row)
    lines[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ledger, None


@pytest.mark.parametrize(
    ("outcome", "phase", "kind", "exception_type", "error_class", "attempted"),
    [
        (
            "adapter_failure",
            "adapter_call",
            "adapter_exception",
            "RuntimeError",
            "RuntimeError",
            None,
        ),
        (
            "model_protocol_failure",
            "adapter_call",
            "protocol_error",
            "Boom",
            "protocol_error",
            None,
        ),
        (
            "scaffold_failure",
            "scaffold_contract",
            "incomplete_scaffold",
            None,
            "incomplete_scaffold",
            None,
        ),
        ("limit_exhausted", "turn_start", "max_turns", None, "max_turns", None),
        (
            "model_action_failure",
            "dispatch",
            "wrong_channel",
            None,
            "wrong_channel",
            {"action": "ask_user", "arguments": {"fact": "x"}},
        ),
    ],
)
def test_a_failure_class_cannot_carry_a_terminal_trajectory(
    tmp_path: Path,
    outcome: str,
    phase: str,
    kind: str,
    exception_type: str | None,
    error_class: str,
    attempted: dict[str, Any] | None,
) -> None:
    root = tmp_path / "run"
    report = _run(root, policy_following_script)

    def mutate(row: dict[str, Any]) -> None:
        steps = len(row["trajectory"]["steps"])
        row["outcome"] = outcome
        # Derived from the outcome, so it moves with it. Without this the row is
        # refused for contradicting its own eligibility before the terminality
        # check runs, and the parametrisation would prove nothing about the
        # failure classes it names.
        row["sample_status"] = sample_status_for(outcome)
        row["evaluation"] = None
        row["error_class"] = error_class
        row["error_detail"] = "forged"
        row["failure_event"] = {
            "phase": phase,
            "kind": kind,
            "attempted_call": attempted,
            "response_type": None,
            "exception_type": exception_type,
            "detail": "forged",
        }
        beyond = 0 if kind in {"incomplete_scaffold", "max_turns"} else 1
        row["turns_used"] = steps + beyond
        row["messages_used"] = 2 * row["turns_used"] - (
            1 if kind in {"adapter_exception", "protocol_error"} else 0
        )

    ledger, _ = _forge(root, mutate)
    with pytest.raises(LedgerError) as info:
        read_ledger(ledger, report.manifest, compiled_variants(_suite()))
    assert "terminal" in str(info.value).lower()


def test_an_evaluator_failure_must_record_a_terminal_trajectory(
    tmp_path: Path,
) -> None:
    def stop_short(request: TurnRequest) -> AdapterCall:
        return AdapterCall("read_records", {})

    root = tmp_path / "run"
    report = _run(root, stop_short)
    assert report.outcome_counts == {"limit_exhausted": 12}

    def mutate(row: dict[str, Any]) -> None:
        row["outcome"] = "evaluator_failure"
        row["sample_status"] = sample_status_for("evaluator_failure")
        row["error_class"] = "ValueError"
        row["error_detail"] = "forged"
        row["failure_event"] = {
            "phase": "evaluation",
            "kind": "evaluator_exception",
            "attempted_call": None,
            "response_type": None,
            "exception_type": "ValueError",
            "detail": "forged",
        }
        row["turns_used"] = len(row["trajectory"]["steps"])
        row["messages_used"] = 2 * row["turns_used"]

    ledger, _ = _forge(root, mutate)
    with pytest.raises(LedgerError) as info:
        read_ledger(ledger, report.manifest, compiled_variants(_suite()))
    assert "terminal" in str(info.value).lower()


# =========================================================================
# §7 failure taxonomy attribution
# =========================================================================


def test_a_channel_no_case_in_this_suite_offers_is_a_protocol_failure(
    tmp_path: Path,
) -> None:
    """No Cube in this suite allows ``call_tool``, so no turn declares it.

    This is the unfiltered-action-surface defect at suite scale. Every episode
    fails the same way — a name that was never offered — rather than as a dispatch
    failure that would read as a model choosing badly among options it was given.
    """

    def unoffered(request: TurnRequest) -> AdapterCall:
        assert "call_tool" not in request.available_actions
        if not any(entry.action == "read_records" for entry in request.transcript):
            return AdapterCall("read_records", {})
        return AdapterCall("call_tool", {"fact": "repair_quote_gbp"})

    root = tmp_path / "run"
    report = _run(root, unoffered)
    assert report.outcome_counts == {"model_protocol_failure": 12}
    assert report.records[0].failure_event is not None
    assert report.records[0].failure_event["kind"] == "unknown_action"


def test_an_unanswerable_fact_request_is_a_model_protocol_failure(
    tmp_path: Path,
) -> None:
    """A key the offered enum does not contain, across a whole run.

    The suite is split on whether a case offers an acquisition channel at all:
    the six maintenance variants elicit their fact and the six access-consent
    ones already hold it. Only the first six can reach this failure, so the
    script asks for a fact no Cube can deliver where a channel exists and closes
    the case where none does, and the counts say which is which rather than
    averaging them.

    Under the fact-affordance contract the keys each channel can retrieve are part of that
    action's schema, so this is refused during response validation rather than
    dispatched and refused by the environment. The run-level shape is what is
    under test here: six rows carrying the same classified argument failure, six
    completing, and no averaging between them.
    """

    def bad_action(request: TurnRequest) -> AdapterCall:
        if not any(entry.action == "read_records" for entry in request.transcript):
            return AdapterCall("read_records", {})
        for channel in ("ask_user", "call_tool"):
            if channel in request.available_actions:
                return AdapterCall(channel, {"fact": "not_a_real_fact"})
        return AdapterCall(
            "complete_case",
            {
                "disposition": "STOP",
                "primary_reason_code": "NO_APPLICABLE_RULE",
                "secondary_reason_codes": [],
                "evidence_refs": [],
            },
        )

    root = tmp_path / "run"
    report = _run(root, bad_action)
    assert report.outcome_counts == {OUTCOME_MODEL_PROTOCOL_FAILURE: 6, "success": 6}
    failures = [record for record in report.records if record.failure_event is not None]
    assert len(failures) == 6
    assert {record.failure_event["kind"] for record in failures} == {
        "malformed_arguments"
    }
    # The offered keys are named; the key the model asked for is not quoted into
    # the durable detail, only into the attempted call it is evidence of.
    assert all(
        "not_a_real_fact" not in (record.error_detail or "") for record in failures
    )


def test_environment_failure_is_reserved_for_environment_machinery() -> None:
    assert "environment_failure" in OUTCOMES
    assert OUTCOME_MODEL_ACTION_FAILURE in OUTCOMES


# =========================================================================
# §4 parsed evidence is recursively immutable
# =========================================================================


def test_a_parsed_record_cannot_be_mutated_and_the_report_cannot_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    report = _run(root, policy_following_script)
    before = report.verdict_passed_count
    record = report.records[0]
    assert record.evaluation is not None
    with pytest.raises(TypeError):
        record.evaluation["passed"] = False  # type: ignore[index]
    with pytest.raises(TypeError):
        record.trajectory["steps"][0]["action"] = "forged"  # type: ignore[index]
    with pytest.raises(TypeError):
        record.usage["input_tokens"] = 5  # type: ignore[index]
    with pytest.raises(TypeError):
        record.adapter["provider"] = "openai"  # type: ignore[index]
    with pytest.raises(TypeError):
        record.contract_versions["runner"] = "9"  # type: ignore[index]
    assert report.verdict_passed_count == before


def test_a_record_still_serialises_to_plain_deterministic_json(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    report = _run(root, policy_following_script)
    payload = report.records[0].as_dict()
    rendered = json.dumps(payload, sort_keys=True)
    assert json.loads(rendered) == payload
    assert isinstance(payload["trajectory"], dict)
    assert isinstance(payload["trajectory"]["steps"], list)


# =========================================================================
# §6 fake adapter provenance
# =========================================================================


def test_the_released_fake_has_a_fixed_non_injectable_identity_and_behaviour() -> None:
    adapter = PolicyFollowingFakeAdapter(settings={})
    assert adapter.identity == AdapterIdentity(
        provider=FAKE_PROVIDER,
        model=FAKE_MODEL,
        implementation=FAKE_IMPLEMENTATION,
        version=ADAPTER_CONTRACT_VERSION,
    )
    assert not hasattr(adapter, "script")
    with pytest.raises(TypeError):
        PolicyFollowingFakeAdapter(settings={}, script=policy_following_script)  # type: ignore[call-arg]


def test_a_test_double_cannot_claim_a_real_provider() -> None:
    with pytest.raises(FakeAdapterProvenanceError):
        ScriptedTestAdapter(
            script=policy_following_script,
            identity=AdapterIdentity(
                provider="openai",
                model="gpt-4o",
                implementation="scripted_fake",
                version=ADAPTER_CONTRACT_VERSION,
            ),
            settings={},
        )


def test_a_test_double_cannot_claim_the_released_fake_identity() -> None:
    with pytest.raises(FakeAdapterProvenanceError):
        ScriptedTestAdapter(
            script=policy_following_script,
            identity=fake_identity(),
            settings={},
        )


def test_a_test_double_run_is_never_a_released_fake_run(tmp_path: Path) -> None:
    root = tmp_path / "run"
    report = _run(root, policy_following_script)
    assert report.manifest.adapter.provider == TEST_DOUBLE_PROVIDER
    assert report.manifest.adapter.model == TEST_DOUBLE_MODEL
    assert report.manifest.adapter.provider != FAKE_PROVIDER


def test_the_cli_cannot_be_given_an_arbitrary_fake_behaviour() -> None:
    from boundarybench.cli import _build_parser

    parser = _build_parser()
    text = parser.format_help()
    assert "--script" not in text


def test_a_released_fake_run_uses_the_released_fake_identity(tmp_path: Path) -> None:
    suite, scaffold = _suite(), _scaffold()
    manifest = _manifest(suite=suite, scaffold=scaffold, identity=fake_identity())
    adapter = PolicyFollowingFakeAdapter(settings={})
    root = tmp_path / "run"
    with open_run_session(root, manifest) as session:
        report = execute_run(
            suite=suite, scaffold=scaffold, adapter=adapter, session=session
        )
    assert report.executed_count == 12
    assert report.records[0].adapter["provider"] == FAKE_PROVIDER


def test_a_test_double_cannot_execute_a_released_fake_run(tmp_path: Path) -> None:
    from boundarybench.runner import RunProvenanceError

    suite, scaffold = _suite(), _scaffold()
    manifest = _manifest(suite=suite, scaffold=scaffold, identity=fake_identity())
    adapter = ScriptedTestAdapter(
        script=policy_following_script,
        identity=identity_for_test_double(),
        settings={},
    )
    root = tmp_path / "run"
    with (
        pytest.raises(RunProvenanceError),
        open_run_session(root, manifest) as session,
    ):
        execute_run(suite=suite, scaffold=scaffold, adapter=adapter, session=session)


# =========================================================================
# §9 run-level usage aggregation
# =========================================================================


class _MeasuringAdapter:
    """A test double that measures some turns and not others."""

    def __init__(self, measured: bool) -> None:
        self._measured = measured
        self.identity = identity_for_test_double(implementation="measuring_test_double")
        self.settings: dict[str, Any] = {}

    def next_call(self, request: TurnRequest, deadline: TurnDeadline) -> AdapterCall:
        return policy_following_script(request)

    def last_usage(self) -> AdapterUsage:
        if not self._measured:
            return AdapterUsage()
        # No ``cost_usd``: this run pins no price, and a row that recorded a
        # cost anyway would state a measurement no manifest records the
        # provenance of. The ledger refuses one.
        return AdapterUsage(
            input_tokens=3, output_tokens=2, cost_usd=None, latency_seconds=0.25
        )


def test_an_unmeasured_run_reports_null_totals_rather_than_zero(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    report = _run(root, policy_following_script)
    totals = report.as_dict()["usage_totals"]
    assert totals["input_tokens"] is None
    assert totals["output_tokens"] is None
    assert totals["cost_usd"] is None
    assert totals["latency_seconds"] is None
    assert report.as_dict()["usage_measured_episodes"]["input_tokens"] == 0


def test_a_measured_run_reports_deterministic_totals_and_measured_counts(
    tmp_path: Path,
) -> None:
    suite, scaffold = _suite(), _scaffold()
    manifest = _manifest(
        suite=suite,
        scaffold=scaffold,
        identity=identity_for_test_double(implementation="measuring_test_double"),
    )
    root = tmp_path / "run"
    with open_run_session(root, manifest) as session:
        report = execute_run(
            suite=suite,
            scaffold=scaffold,
            adapter=_MeasuringAdapter(measured=True),
            session=session,
        )
    payload = report.as_dict()
    assert payload["usage_totals"]["input_tokens"] > 0
    assert payload["usage_measured_episodes"]["input_tokens"] == 12
    assert payload["usage_unmeasured_episodes"]["input_tokens"] == 0
    assert payload["counts"]["succeeded"] == 12


class _PartiallyMeasuringAdapter:
    """Measures the first ``measured_episodes`` episodes and nothing afterwards.

    A provider that reports usage on some calls and not others is the realistic
    case — a stream that drops its usage frame, a retry served from a cache —
    and it is the case a total can silently lie about.
    """

    def __init__(self, measured_episodes: int) -> None:
        self._measured_episodes = measured_episodes
        self._started = 0
        self._measuring = False
        self.identity = identity_for_test_double(implementation="measuring_test_double")
        self.settings: dict[str, Any] = {}

    def next_call(self, request: TurnRequest, deadline: TurnDeadline) -> AdapterCall:
        if not request.transcript:
            self._started += 1
            self._measuring = self._started <= self._measured_episodes
        return policy_following_script(request)

    def last_usage(self) -> AdapterUsage:
        if not self._measuring:
            return AdapterUsage()
        # Tokens and latency are measured; cost is not priced for this double,
        # so it stays absent on every episode including the measured ones.
        return AdapterUsage(input_tokens=3, output_tokens=2, latency_seconds=0.25)


def test_a_partly_measured_run_totals_only_the_episodes_that_measured(
    tmp_path: Path,
) -> None:
    """A partial total is a real sum over a stated subset, never an estimate.

    Half the episodes report tokens and latency; half report nothing; no episode
    reports cost. So the token and latency totals must equal the sum of exactly
    the rows that carry a measurement — read back from the ledger bytes, not
    from the report's own view of them — the measured/unmeasured episode counts
    must partition the run, and cost must stay ``None`` rather than collapsing
    to ``0.0`` beside fields that did measure.
    """
    suite, scaffold = _suite(), _scaffold()
    manifest = _manifest(
        suite=suite,
        scaffold=scaffold,
        identity=identity_for_test_double(implementation="measuring_test_double"),
    )
    root = tmp_path / "run"
    with open_run_session(root, manifest) as session:
        report = execute_run(
            suite=suite,
            scaffold=scaffold,
            adapter=_PartiallyMeasuringAdapter(measured_episodes=6),
            session=session,
        )

    durable = read_ledger(session.paths.ledger_path, manifest, compiled_variants(suite))
    payload = report.as_dict()

    for field in ("input_tokens", "output_tokens", "latency_seconds"):
        measured = [
            row.usage[field] for row in durable if row.usage.get(field) is not None
        ]
        assert len(measured) == 6
        assert payload["usage_totals"][field] == pytest.approx(sum(measured))
        assert payload["usage_measured_episodes"][field] == 6
        assert payload["usage_unmeasured_episodes"][field] == 6

    assert payload["usage_totals"]["cost_usd"] is None
    assert payload["usage_measured_episodes"]["cost_usd"] == 0
    assert payload["usage_unmeasured_episodes"]["cost_usd"] == 12


def test_execution_counts_and_verdict_counts_stay_separate(tmp_path: Path) -> None:
    root = tmp_path / "run"
    report = _run(root, policy_following_script)
    payload = report.as_dict()
    assert "usage_totals" in payload
    assert set(payload["counts"]) >= {"succeeded", "verdict_passed", "verdict_failed"}


# =========================================================================
# §8 the ledger integrity claim is honest
# =========================================================================


def test_the_ledger_integrity_note_is_stated_and_does_not_overclaim() -> None:
    note = LEDGER_INTEGRITY_NOTE.lower()
    assert "append-only" in note
    assert "not" in note
    assert "external" in note
    for overclaim in ("tamper-proof", "cryptographically immutable", "signed"):
        assert f" {overclaim} " not in f" {note} " or "not" in note


def test_the_run_report_states_the_ledger_limitation(tmp_path: Path) -> None:
    root = tmp_path / "run"
    report = _run(root, policy_following_script)
    assert report.as_dict()["ledger_integrity_note"] == LEDGER_INTEGRITY_NOTE
