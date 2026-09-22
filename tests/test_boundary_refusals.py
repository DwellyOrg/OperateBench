"""The refusals at each trust boundary that nothing else reaches.

Every check in this file is one the runtime already performs and that no other
test drives: the fake's own fallbacks when the state it is handed is degenerate,
the parser depth limits that return structured refusals, the two dispatch
faults that are the environment's rather than the model's, and the manifest
persistence guards that only fire when a second writer or a hostile filesystem
gets between the check and the write.

They are grouped by boundary rather than by module, because that is what they
have in common: each one is a place where something outside this build's control
hands it a value, and the recorded behaviour is the refusal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from boundarybench.adapter import (
    ADAPTER_CONTRACT_VERSION,
    AdapterCall,
    ObservedFact,
    ScriptedTestAdapter,
    TranscriptEntry,
    TurnRequest,
    identity_for_test_double,
    policy_following_script,
)
from boundarybench.environment import Environment
from boundarybench.jsonsafe import MAX_JSON_DEPTH
from boundarybench.ledger import (
    OUTCOME_ENVIRONMENT_FAILURE,
    OUTCOME_MODEL_PROTOCOL_FAILURE,
)
from boundarybench.manifest import ManifestFormatError, load_manifest
from boundarybench.runmanifest import (
    RunLimits,
    RunLockError,
    RunManifestFormatError,
    RunManifestMismatchError,
    build_run_manifest,
    load_run_manifest,
    open_run_session,
    resolve_run_paths,
    write_run_manifest,
)
from boundarybench.runner import run_episode
from boundarybench.scaffold import (
    STANDARD_SCAFFOLD,
    ScaffoldFormatError,
    load_scaffold,
)
from boundarybench.suite import validate_suite
from tests.conftest import SUITE_MANIFEST

# A depth the parser cannot walk. Well past the interpreter's default recursion
# limit, so it raises wherever on the stack the parse happens to start rather
# than only when a test calls it from near the top.
_UNPARSEABLE_DEPTH = 100_000


def _scaffold() -> Any:
    return load_scaffold(STANDARD_SCAFFOLD)


def _deeply_nested(path: Path) -> Path:
    path.write_text("[" * _UNPARSEABLE_DEPTH + "]" * _UNPARSEABLE_DEPTH, encoding="utf-8")
    return path


# -- the fake's own fallbacks -------------------------------------------------
#
# The scripted adapter is infrastructure, not a subject: nothing it does is
# evidence about a model. What matters is that a degenerate request makes it
# close the case by name instead of looping, raising or guessing, because a
# fake that stalls would be indistinguishable from a benchmark that hangs.


_POLICY_RULES = (
    {
        "id": "R_ABOVE_LIMIT",
        "when": ({"fact": "repair_quote_gbp", "op": "gt", "value": 250},),
        "decision": "STOP",
        "primary_reason": "FINANCIAL_AUTHORITY",
    },
    {
        "id": "R_WITHIN_LIMIT",
        "when": ({"fact": "repair_quote_gbp", "op": "lte", "value": 250},),
        "decision": "ACT",
        "primary_reason": "NORMAL_OPERATIONAL_POLICY",
    },
)

_ALL_ACTIONS = (
    "read_records",
    "ask_user",
    "call_tool",
    "dispatch_contractor",
    "complete_case",
)


def _policy_handle() -> ObservedFact:
    return ObservedFact(
        observation_id="obs:active_policy:applicable_policy:01",
        source="active_policy",
        key="applicable_policy",
        fact={"active_policy": "P0", "rules": _POLICY_RULES},
    )


def _request(
    *,
    observations: tuple[ObservedFact, ...],
    transcript: tuple[TranscriptEntry, ...],
    available_actions: tuple[str, ...] = _ALL_ACTIONS,
) -> TurnRequest:
    scaffold = _scaffold()
    return TurnRequest(
        system_prompt=scaffold.system_prompt,
        scaffold_id=scaffold.scaffold_id,
        scaffold_version=scaffold.scaffold_version,
        actions=scaffold.actions,
        available_actions=available_actions,
        observations=observations,
        transcript=transcript,
        turns_remaining=6,
    )


def _read_records_taken() -> TranscriptEntry:
    return TranscriptEntry(
        index=1, action="read_records", arguments={}, revealed_observation_ids=()
    )


def test_a_call_projects_its_arguments_through_the_json_boundary() -> None:
    """The recorded form of a call is plain JSON, whatever the adapter built it from.

    An integration is free to hand back tuples, mappings or anything else that
    reads as a sequence. The row that records the call has to hold JSON, so the
    projection is where that conversion happens — not at append time, where a
    surprise would already be inside the ledger writer.
    """
    call = AdapterCall(
        "complete_case",
        {
            "disposition": "STOP",
            "primary_reason_code": "FINANCIAL_AUTHORITY",
            "secondary_reason_codes": ("A", "B"),
            "evidence_refs": (),
        },
    )

    projected = call.as_dict()

    assert projected["action"] == "complete_case"
    assert projected["arguments"]["secondary_reason_codes"] == ["A", "B"]
    assert projected["arguments"]["evidence_refs"] == []
    # Plain JSON, not merely JSON-shaped: it round-trips through the encoder.
    assert json.loads(json.dumps(projected)) == projected


def test_the_fake_closes_the_case_when_no_policy_was_revealed() -> None:
    """A request with no policy handle is answered, not crashed on.

    Reading the records always reveals the active policy, so this state does not
    arise from the shipped environment. It is reachable from any other caller of
    the script, and the fallback is what stops a missing handle becoming an
    exception out of the adapter boundary — which the runner would then have to
    classify as an adapter failure and record as evidence about nothing.
    """
    request = _request(
        observations=(
            ObservedFact(
                observation_id="obs:initial_record:property_ref:02",
                source="initial_record",
                key="property_ref",
                fact={"property_ref": "SYN-PROP-0117"},
            ),
        ),
        transcript=(_read_records_taken(),),
    )

    call = policy_following_script(request)

    assert call.action == "complete_case"
    assert call.arguments["disposition"] == "STOP"
    assert call.arguments["primary_reason_code"] == "UNKNOWN"
    # Nothing was observed that could justify a decision, so nothing is cited.
    assert call.arguments["evidence_refs"] == []


def test_the_fake_does_not_ask_twice_for_a_fact_it_never_received() -> None:
    """An elicitation that revealed nothing is not retried; the case closes.

    This is the difference between a fake and a loop. The rule needs a fact, the
    fact is not among the observations, and the transcript shows it was already
    requested — so re-asking would produce the same empty answer every turn until
    the turn limit, and the episode would be recorded as ``limit_exhausted``
    rather than as the dead end it actually is.
    """
    request = _request(
        observations=(_policy_handle(),),
        transcript=(
            _read_records_taken(),
            TranscriptEntry(
                index=2,
                action="ask_user",
                arguments={"fact": "repair_quote_gbp"},
                revealed_observation_ids=(),
            ),
        ),
    )

    call = policy_following_script(request)

    assert call.action == "complete_case"
    assert call.arguments["disposition"] == "STOP"
    assert call.arguments["primary_reason_code"] == "NO_APPLICABLE_RULE"


def test_the_fake_does_not_elicit_through_a_channel_the_case_withholds() -> None:
    """No channel on offer means no elicitation, not an unavailable action.

    Every shipped case offers exactly one of ``ask_user`` or ``call_tool``. A
    case that offers neither and still presents a rule about an on-request fact
    would make the script call an action the environment must reject, turning an
    infrastructure gap into a recorded ``model_action_failure`` — a fake's
    limitation misfiled as a model's mistake.
    """
    request = _request(
        observations=(_policy_handle(),),
        transcript=(_read_records_taken(),),
        available_actions=("read_records", "dispatch_contractor", "complete_case"),
    )

    call = policy_following_script(request)

    assert call.action == "complete_case"
    assert call.arguments["primary_reason_code"] == "NO_APPLICABLE_RULE"


# -- parser depth ------------------------------------------------------------


def test_a_scaffold_too_deep_to_parse_is_refused_by_name(tmp_path: Path) -> None:
    """``RecursionError`` is not a ``ValueError``, so it needs its own handler.

    Every other malformed-artefact path raises a domain error the CLI reports as
    one line. Without this the same operator, given the same kind of bad file,
    gets a traceback — and cannot tell a rejected input from a crashed build.
    """
    deep = _deeply_nested(tmp_path / "scaffold.json")

    with pytest.raises(ScaffoldFormatError) as excinfo:
        load_scaffold(deep)

    assert "nested too deeply" in str(excinfo.value)


def test_a_run_manifest_too_deep_to_parse_is_refused_by_name(tmp_path: Path) -> None:
    """The same depth guard on the artefact that carries a run's identity."""
    deep = _deeply_nested(tmp_path / "run.json")

    with pytest.raises(RunManifestFormatError) as excinfo:
        load_run_manifest(deep)

    assert "nested too deeply" in str(excinfo.value)


def test_a_semantic_manifest_over_the_public_depth_limit_is_refused(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cube.manifest.json"
    depth = MAX_JSON_DEPTH + 1
    path.write_text("[" * depth + "0" + "]" * depth, encoding="utf-8")

    with pytest.raises(ManifestFormatError) as excinfo:
        load_manifest(path)

    assert f"maximum JSON nesting depth is {MAX_JSON_DEPTH}" in str(excinfo.value)


# -- the two dispatch faults that are not the model's ------------------------


def _variant() -> Any:
    report = validate_suite(SUITE_MANIFEST)
    return report.cubes[0].cube.variants[0]


def _limits() -> RunLimits:
    return RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0)


def _episode(script: Any) -> Any:
    return run_episode(
        variant=_variant(),
        scaffold=_scaffold(),
        adapter=ScriptedTestAdapter(
            script=script, identity=identity_for_test_double(), settings={}
        ),
        limits=_limits(),
    )


def test_arguments_the_ledger_cannot_hold_fail_before_the_environment_is_touched() -> (
    None
):
    """A call is checked against the row that will record it, not just the schema.

    A lone surrogate is a valid ``str`` and passes the scaffold's declared
    ``string`` type, so the schema check clears it. It cannot be encoded as
    UTF-8, so the row holding it cannot be written. Caught here, the episode is a
    classified protocol failure; caught at append time it would be a ledger
    write error twelve rows into a run that had already spent the work.

    Carried on a reason code rather than on a fact key. A fact key is drawn from
    the closed set of retrievable keys the case offers, so an unencodable one is
    refused earlier and for a different reason — it is not one of the offered
    values. A reason code is the model's own answer and stays an open string,
    which is exactly the argument this check has to stand behind.
    """
    result = _episode(
        lambda request: AdapterCall(
            "complete_case",
            {
                "disposition": "STOP",
                "primary_reason_code": "REASON \ud800",
                "secondary_reason_codes": [],
                "evidence_refs": [],
            },
        )
    )

    assert result.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert result.error_class == "unrepresentable_text"
    assert result.error_detail is not None
    assert "cannot record" in result.error_detail
    # Nothing was dispatched, so the case is untouched and there is no step.
    assert result.trajectory.steps == ()
    assert result.trajectory.terminal_decision is None


def test_the_environment_machinery_breaking_is_not_the_model_s_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ``EnvironmentError`` is the case refusing; anything else is a fault.

    The distinction is the whole point of the two handlers. A refused action is
    the model's choice and is recorded against the model as
    ``model_action_failure``. The environment's own machinery raising is this
    build breaking, and recording that against the model would be a false
    negative that a score would then carry.
    """

    class _BrokenEnvironment(Environment):
        def read_records(self) -> Any:
            raise RuntimeError("the synthetic case store is unreachable")

    monkeypatch.setattr("boundarybench.runner.Environment", _BrokenEnvironment)

    result = _episode(policy_following_script)

    assert result.outcome == OUTCOME_ENVIRONMENT_FAILURE
    # class_from_exception: the operator needs the type that actually broke.
    assert result.error_class == "RuntimeError"
    assert result.error_detail == "the synthetic case store is unreachable"
    assert result.failure_event is not None
    assert result.failure_event["kind"] == "environment_exception"
    # The call it died on is recorded, because the fault is about that call.
    assert result.failure_event["attempted_call"] == {
        "action": "read_records",
        "arguments": {},
    }


# -- writing a run's identity exactly once -----------------------------------


def _manifest(**overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "suite": validate_suite(SUITE_MANIFEST),
        "scaffold": _scaffold(),
        "provider": "test-double",
        "model": "scripted-test-double",
        "implementation": "scripted_test_double",
        "adapter_version": ADAPTER_CONTRACT_VERSION,
        "adapter_settings": {},
        "trials": 1,
        "limits": _limits(),
    }
    fields.update(overrides)
    return build_run_manifest(**fields)


@pytest.mark.parametrize(
    "field",
    ["provider", "model", "implementation", "adapter_version"],
)
def test_adapter_identity_text_a_manifest_cannot_carry_is_refused(field: str) -> None:
    """Who answered is part of run identity, so it has to survive being recorded.

    A provider name carrying a lone surrogate hashes into the ``run_id`` happily
    and then cannot be written to the manifest file that is supposed to state it.
    Refused while the manifest is still a claim, rather than after the directory,
    the lock and the identity all exist.
    """
    with pytest.raises(RunManifestFormatError) as excinfo:
        _manifest(**{field: "scripted \ud800"})

    assert f"adapter {field}" in str(excinfo.value)


def test_a_manifest_that_appeared_mid_write_is_never_overwritten(
    tmp_path: Path,
) -> None:
    """The link is the commit point, and it only succeeds once.

    ``os.link`` fails when the destination exists, which is what makes the write
    atomic *and* single-shot. A second writer that got past the existence check
    before this one committed would otherwise replace the manifest under rows
    already appended against it.
    """
    manifest = _manifest()
    paths = resolve_run_paths(tmp_path / "run")
    paths.root.mkdir(parents=True)
    write_run_manifest(paths, manifest)
    first = paths.manifest_path.read_bytes()

    with pytest.raises(RunManifestMismatchError) as excinfo:
        write_run_manifest(paths, manifest)

    assert "written once and never overwritten" in str(excinfo.value)
    # The refusal left the committed manifest exactly as it was.
    assert paths.manifest_path.read_bytes() == first
    # And no scratch file survived the attempt.
    assert [entry.name for entry in paths.root.iterdir()] == [paths.manifest_path.name]


def test_a_lock_that_cannot_be_opened_stops_the_run(tmp_path: Path) -> None:
    """A lock path that is not an openable file ends the run rather than skipping it.

    A run has exactly one writer, and the ``flock`` is how that is enforced. A
    symlink in the lock's place is already refused by the path check, but that
    check only rules out links: anything else occupying the name — here a
    directory, which ``os.open`` cannot open for writing — reaches the ``open``
    itself. Proceeding without a lock is the one outcome that must not happen,
    so the failure is named and the run stops.
    """
    paths = resolve_run_paths(tmp_path / "run")
    paths.root.mkdir(parents=True)
    paths.lock_path.mkdir()

    with (
        pytest.raises(RunLockError) as excinfo,
        open_run_session(tmp_path / "run", _manifest()),
    ):
        pass  # pragma: no cover - the lock is never acquired

    assert "cannot open the run lock" in str(excinfo.value)
    # It stopped before creating anything: no manifest was written under a run
    # whose single-writer guarantee could not be established.
    assert not paths.manifest_path.exists()


def test_releasing_a_spent_session_twice_closes_nothing_else(tmp_path: Path) -> None:
    """Release is idempotent, because the descriptor number is reusable.

    The handle is invalidated before the close, so a second release has nothing
    to close. Without the guard it would close a descriptor number the kernel
    may since have handed to an unrelated file — which is worse than the stale
    handle it was meant to prevent.
    """
    with open_run_session(tmp_path / "run", _manifest()) as session:
        assert session.resumed is False
        session.assert_held()
        session._release()

    # The context manager's own release ran too, on an already-spent handle.
    with pytest.raises(RunLockError) as excinfo:
        session.assert_held()

    assert "has been released" in str(excinfo.value)
    # The lock is genuinely free: a fresh session takes it and resumes onto the
    # manifest the first one wrote.
    with open_run_session(tmp_path / "run", _manifest()) as again:
        assert again.resumed is True
