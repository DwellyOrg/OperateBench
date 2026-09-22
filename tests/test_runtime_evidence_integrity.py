"""Adversarial checks for runtime evidence integrity.

They share one integrity theme. Each is a fact the
run *asserts about its own failures* that nothing derives independently: a
counter that no accounting explains, a manifest identity that nothing rehashes,
and a failure classification with no persisted cause to check it against.

The limit is worth stating plainly. This is an unsigned local ledger. Nothing
here makes it cryptographically tamper-proof:
an editor who rewrites *every* mutually bound field consistently still produces
a file this build accepts. What these tests pin is deterministic internal
consistency and provenance — a row must derive from evidence it carries, and a
single edit anywhere in that chain must be refused.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from boundarybench.adapter import (
    AdapterCall,
    AdapterProtocolError,
    TurnRequest,
)
from boundarybench.ledger import (
    OUTCOME_ADAPTER_FAILURE,
    OUTCOME_ENVIRONMENT_FAILURE,
    OUTCOME_MODEL_PROTOCOL_FAILURE,
    LedgerError,
    append_episode,
    read_ledger,
    sample_status_for,
)
from boundarybench.runmanifest import (
    RunLimits,
    RunManifestMismatchError,
    open_run_session,
)
from boundarybench.runner import (
    build_episode_record,
    compiled_variants,
    execute_run,
    run_episode,
)
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.suite import validate_suite
from tests.conftest import SUITE_MANIFEST
from tests.test_runtime_execution_integrity import (
    _fake_adapter,
    _manifest,
    _rewrite_first_row,
)


def _one_episode(root: Path, script: Any) -> tuple[Any, dict[str, Any], Path]:
    """Run the plan's first episode for real and append the row it produced.

    A genuine row, not a hand-written one: every negative test below only means
    something if the row it starts from is one this build actually writes.
    """
    manifest = _manifest()
    suite = validate_suite(SUITE_MANIFEST)
    variants = compiled_variants(suite)
    entry = manifest.episode_plan[0]
    variant = variants[entry.variant_id]
    result = run_episode(
        variant=variant,
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        adapter=_fake_adapter(script=script),
        limits=manifest.limits,
    )
    with open_run_session(root, manifest) as session:
        ledger = session.paths.ledger_path
        append_episode(ledger, build_episode_record(manifest, entry, variant, result))
    return manifest, variants, ledger


# -- 1. an adapter exception is a readable row, not a self-corrupt one --------


def _exploding(message: str) -> Any:
    def script(request: TurnRequest) -> AdapterCall:
        raise RuntimeError(message)

    return script


def test_a_whole_run_of_adapter_exceptions_records_reads_back_and_resumes(
    tmp_path: Path,
) -> None:
    """Adapter exceptions produce readable rows whose counters agree.

    A failed outbound call spends one turn and one message. The classified
    failure remains readable, reportable and resumable across a whole run.
    """
    root = tmp_path / "run"
    manifest = _manifest()
    suite = validate_suite(SUITE_MANIFEST)
    scaffold = load_scaffold(STANDARD_SCAFFOLD)
    ledger = root / "episodes.jsonl"

    with open_run_session(root, manifest) as session:
        report = execute_run(
            suite=suite,
            scaffold=scaffold,
            adapter=_fake_adapter(script=_exploding("provider connection reset")),
            session=session,
        )

    assert report.executed_count == report.planned_count == 12
    assert report.outcome_counts == {OUTCOME_ADAPTER_FAILURE: 12}
    assert report.error_class_counts == {"RuntimeError": 12}
    for record in report.records:
        assert record.turns_used == 1
        assert record.messages_used == 1
        # The class, not the message: an adapter's exception text is external
        # and is not persisted. See tests/test_provider_output_trust.py.
        assert "provider connection reset" not in (record.error_detail or "")
        assert "RuntimeError" in (record.error_detail or "")

    # And the run is a byte-for-byte no-op to repeat.
    before = ledger.read_bytes()
    with open_run_session(root, manifest) as session:
        again = execute_run(
            suite=suite,
            scaffold=scaffold,
            adapter=_fake_adapter(script=_exploding("provider connection reset")),
            session=session,
        )
    assert again.executed_count == 0
    assert again.resumed_count == 12
    assert ledger.read_bytes() == before


def test_an_adapter_exception_with_no_message_still_writes_a_valid_row(
    tmp_path: Path,
) -> None:
    """``str(exc)`` is empty for ``RuntimeError()``, and a row needs a detail.

    A failure with a blank ``error_detail`` is refused by the ledger, so an
    exception that carried no message produced a row the run could not read
    back. The detail falls back to something stable and non-empty instead.
    """
    root = tmp_path / "run"
    manifest, variants, ledger = _one_episode(root, _exploding(""))

    (record,) = read_ledger(ledger, manifest, variants)

    assert record.outcome == OUTCOME_ADAPTER_FAILURE
    assert record.error_class == "RuntimeError"
    assert record.error_detail
    assert "RuntimeError" in (record.error_detail or "")


# -- 2. a supplied manifest is rehashed before anything is opened -------------


def test_a_supplied_manifest_whose_limits_no_longer_hash_to_its_id_is_refused(
    tmp_path: Path,
) -> None:
    """A supplied manifest must hash to its own recorded identity.

    The adversarial object keeps its recorded ``configuration_id`` while changing
    a limit. It is refused before execution even though the on-disk manifest is
    valid.
    """
    root = tmp_path / "run"
    manifest = _manifest()
    assert manifest.limits.max_turns == 12
    with open_run_session(root, manifest) as session:
        stored_path = session.paths.manifest_path
    stored_payload = json.loads(stored_path.read_text(encoding="utf-8"))
    assert stored_payload["limits"]["max_turns"] == 12

    forged = dataclasses.replace(
        manifest,
        limits=RunLimits(max_turns=1, max_messages=64, episode_timeout_seconds=30.0),
    )
    assert forged.configuration_id == manifest.configuration_id

    with (
        pytest.raises(RunManifestMismatchError) as excinfo,
        open_run_session(root, forged),
    ):
        pass  # pragma: no cover - the identity check must refuse this

    assert "configuration_id" in str(excinfo.value)
    assert not (root / "episodes.jsonl").exists()


def test_a_forged_manifest_is_refused_before_a_run_directory_is_created(
    tmp_path: Path,
) -> None:
    """Nothing is created or opened until identity has been re-derived."""
    root = tmp_path / "fresh"
    forged = dataclasses.replace(_manifest(), trials=7)

    with pytest.raises(RunManifestMismatchError), open_run_session(root, forged):
        pass  # pragma: no cover - the identity check must refuse this

    assert not root.exists()


def test_execute_run_rehashes_the_manifest_it_was_handed(tmp_path: Path) -> None:
    """The session is not the only way a manifest reaches the runner.

    ``open_run_session`` refuses a manifest that does not hash to its own id,
    but the execution gate is where every other artefact is re-derived, and it
    compared them all against a manifest it took on trust. A ``RunSession`` is
    an ordinary frozen dataclass, so swapping the manifest inside one is exactly
    as easy as rebuilding the manifest was.
    """
    root = tmp_path / "run"
    manifest = _manifest()
    suite = validate_suite(SUITE_MANIFEST)
    forged = dataclasses.replace(
        manifest,
        limits=RunLimits(max_turns=1, max_messages=64, episode_timeout_seconds=30.0),
    )

    with open_run_session(root, manifest) as session:
        swapped = dataclasses.replace(session, manifest=forged)
        with pytest.raises(RunManifestMismatchError) as excinfo:
            execute_run(
                suite=suite,
                scaffold=load_scaffold(STANDARD_SCAFFOLD),
                adapter=_fake_adapter(),
                session=swapped,
            )
        assert not session.paths.ledger_path.exists()

    assert "does not hash to its own configuration_id" in str(excinfo.value)


# -- 3. a failure classification is derived from persisted cause --------------


def _teleport(request: TurnRequest) -> AdapterCall:
    return AdapterCall("teleport", {})


def test_a_rewritten_failure_taxonomy_is_refused_across_categories(
    tmp_path: Path,
) -> None:
    """A cross-taxonomy forgery is contradicted by its persisted cause.

    The adversarial row relabels ``model_protocol_failure``/``unknown_action`` as
    ``environment_failure``/``action_rejected`` while leaving its cause in place.
    Outcome, class and detail must all derive from the typed failure event.
    """
    root = tmp_path / "run"
    manifest, variants, ledger = _one_episode(root, _teleport)

    (genuine,) = read_ledger(ledger, manifest, variants)
    assert genuine.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert genuine.error_class == "unknown_action"

    def mutate(row: dict[str, Any]) -> None:
        row["outcome"] = OUTCOME_ENVIRONMENT_FAILURE
        # Carried through with the outcome. The two taxonomies moved together in
        # this forgery — a behavioural attempt relabelled as an infrastructure
        # failure is also a relabel of what the row is a sample of — and leaving
        # it behind would stop the row on that derivation rather than on the
        # failure event the test is about.
        row["sample_status"] = sample_status_for(OUTCOME_ENVIRONMENT_FAILURE)
        row["error_class"] = "action_rejected"
        row["error_detail"] = "forged classification"

    _rewrite_first_row(ledger, mutate)

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(ledger, manifest, variants)

    assert "failure_event" in str(excinfo.value)


def test_a_failure_event_is_present_on_every_failure_and_absent_on_success(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    suite = validate_suite(SUITE_MANIFEST)
    variants = compiled_variants(suite)
    entry = manifest.episode_plan[0]
    variant = variants[entry.variant_id]
    scaffold = load_scaffold(STANDARD_SCAFFOLD)

    from boundarybench.adapter import policy_following_script

    good = run_episode(
        variant=variant,
        scaffold=scaffold,
        adapter=_fake_adapter(script=policy_following_script),
        limits=manifest.limits,
    )
    bad = run_episode(
        variant=variant,
        scaffold=scaffold,
        adapter=_fake_adapter(script=_teleport),
        limits=manifest.limits,
    )

    assert good.failure_event is None
    assert bad.failure_event is not None
    assert bad.failure_event["kind"] == "unknown_action"
    assert bad.failure_event["attempted_call"] == {"action": "teleport", "arguments": {}}


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda event: event.update(kind="action_rejected"), "failure_event"),
        (lambda event: event.update(phase="dispatch"), "failure_event"),
        (lambda event: event.update(detail="something else entirely"), "error_detail"),
        (lambda event: event.update(extra="x"), "unknown field(s)"),
        (lambda event: event.pop("attempted_call"), "missing required field(s)"),
        (lambda event: event.update(attempted_call=None), "attempted_call"),
        (lambda event: event.update(exception_type="RuntimeError"), "exception_type"),
        (lambda event: event.update(kind=7), "'kind' must be"),
        (lambda event: event.update(detail=""), "'detail' must be"),
        (lambda event: event.update(kind="teleportation"), "unknown"),
        (lambda event: event.update(phase=7), "'phase' must be"),
        (
            lambda event: event.update(attempted_call={"action": "teleport"}),
            "attempted_call",
        ),
        (lambda event: event.update(attempted_call="teleport"), "must be a JSON object"),
        (
            lambda event: event.update(attempted_call={"action": 7, "arguments": {}}),
            "'action' must be a string",
        ),
        (
            lambda event: event.update(
                attempted_call={"action": "teleport", "arguments": []}
            ),
            "'arguments' must be a JSON object or null",
        ),
    ],
)
def test_a_tampered_failure_event_is_refused(
    tmp_path: Path, mutate: Any, expected: str
) -> None:
    """The event is the evidence, so it is parsed at least as strictly as a row."""
    root = tmp_path / "run"
    manifest, variants, ledger = _one_episode(root, _teleport)

    _rewrite_first_row(ledger, lambda row: mutate(row["failure_event"]))

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(ledger, manifest, variants)

    assert expected in str(excinfo.value)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda row: row.update(failure_event=None), "must carry a 'failure_event'"),
        (lambda row: row.update(failure_event=[]), "must be a JSON object or null"),
        (lambda row: row.update(error_class="malformed_response"), "error_class"),
    ],
)
def test_a_failure_row_that_disagrees_with_its_own_event_is_refused(
    tmp_path: Path, mutate: Any, expected: str
) -> None:
    root = tmp_path / "run"
    manifest, variants, ledger = _one_episode(root, _teleport)

    _rewrite_first_row(ledger, mutate)

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(ledger, manifest, variants)

    assert expected in str(excinfo.value)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda event: event.update(exception_type=""), "exception_type"),
        (lambda event: event.update(detail=""), "'detail' must be"),
        (lambda event: event.update(detail="   "), "'detail' must be"),
        (
            lambda event: event.update(
                attempted_call={"action": "read_records", "arguments": {}}
            ),
            "refuses no call",
        ),
    ],
)
def test_an_adapter_exception_event_carries_exactly_its_own_evidence(
    tmp_path: Path, mutate: Any, expected: str
) -> None:
    """Each kind is defined by one shape of evidence, and only that shape.

    An exception has a type and refuses no call; a rejected call is the reverse.
    Letting either borrow the other's evidence would let a rewritten ``kind``
    keep a cause that never belonged to it.
    """
    root = tmp_path / "run"
    manifest, variants, ledger = _one_episode(root, _exploding("gone"))

    _rewrite_first_row(ledger, lambda row: mutate(row["failure_event"]))

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(ledger, manifest, variants)

    assert expected in str(excinfo.value)


def test_a_successful_row_may_not_carry_a_failure_event(tmp_path: Path) -> None:
    """A success reached a terminal decision, so it recorded no fault."""
    from boundarybench.adapter import policy_following_script

    root = tmp_path / "run"
    manifest, variants, ledger = _one_episode(root, policy_following_script)
    (record,) = read_ledger(ledger, manifest, variants)
    assert record.failure_event is None

    _rewrite_first_row(
        ledger,
        lambda row: row.update(
            failure_event={
                "phase": "dispatch",
                "kind": "action_rejected",
                "attempted_call": {"action": "read_records", "arguments": {}},
                "response_type": None,
                "exception_type": None,
                "detail": "invented",
            }
        ),
    )

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(ledger, manifest, variants)

    assert "must be null for a successful episode" in str(excinfo.value)


def test_a_response_that_was_not_a_call_records_the_shape_it_was(
    tmp_path: Path,
) -> None:
    """``malformed_response`` has no call to record, so it records the shape."""

    def script(request: TurnRequest) -> Any:
        return {"action": "read_records"}

    root = tmp_path / "run"
    manifest, variants, ledger = _one_episode(root, script)

    (record,) = read_ledger(ledger, manifest, variants)

    assert record.error_class == "malformed_response"
    assert record.failure_event is not None
    assert record.failure_event["response_type"] == "dict"
    assert record.failure_event["attempted_call"] is None


def test_arguments_that_json_cannot_carry_back_are_recorded_as_absent(
    tmp_path: Path,
) -> None:
    """The refused call is persisted, and only when a row can carry it verbatim.

    A float argument is JSON; an arbitrary object is not. An unrepresentable call
    cannot be carried verbatim and is therefore recorded as absent.
    """

    def numeric(request: TurnRequest) -> AdapterCall:
        return AdapterCall("ask_user", {"fact": 1.5})

    def unserialisable(request: TurnRequest) -> AdapterCall:
        return AdapterCall("ask_user", {"fact": object()})

    manifest, variants, ledger = _one_episode(tmp_path / "numeric", numeric)
    (record,) = read_ledger(ledger, manifest, variants)
    assert record.failure_event is not None
    assert record.failure_event["attempted_call"] == {
        "action": "ask_user",
        "arguments": {"fact": 1.5},
    }

    manifest, variants, ledger = _one_episode(tmp_path / "opaque", unserialisable)
    (record,) = read_ledger(ledger, manifest, variants)
    assert record.error_class == "malformed_arguments"
    assert record.failure_event is not None
    assert record.failure_event["attempted_call"] == {
        "action": "ask_user",
        "arguments": None,
    }


def test_an_adapter_exception_row_may_not_claim_an_even_message_count(
    tmp_path: Path,
) -> None:
    """The odd count is an exception to the rule, not a hole in it.

    A request that was never answered leaves one message unmatched. That is the
    only shape allowed to, and it is allowed only because the event says the
    adapter call raised — every completed call still costs exactly two.
    """
    root = tmp_path / "run"
    manifest, variants, ledger = _one_episode(root, _exploding("gone"))

    (record,) = read_ledger(ledger, manifest, variants)
    assert (record.turns_used, record.messages_used) == (1, 1)

    _rewrite_first_row(ledger, lambda row: row.update(messages_used=2))

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(ledger, manifest, variants)

    assert "messages_used" in str(excinfo.value)


def test_a_completed_call_may_not_borrow_the_unanswered_request_allowance(
    tmp_path: Path,
) -> None:
    """A row whose event says the call returned must account for both messages."""
    root = tmp_path / "run"
    manifest, variants, ledger = _one_episode(root, _teleport)

    (record,) = read_ledger(ledger, manifest, variants)
    assert (record.turns_used, record.messages_used) == (1, 2)

    _rewrite_first_row(ledger, lambda row: row.update(messages_used=1))

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(ledger, manifest, variants)

    assert "messages_used" in str(excinfo.value)


def test_a_protocol_error_raised_by_the_adapter_is_a_readable_row(
    tmp_path: Path,
) -> None:
    """``AdapterProtocolError`` leaves the request unanswered too.

    It is a model/protocol failure rather than an infrastructure one, but it is
    still raised *out of* the call, so no response ever arrived.
    """

    def script(request: TurnRequest) -> AdapterCall:
        raise AdapterProtocolError("bad provider payload")

    root = tmp_path / "run"
    manifest, variants, ledger = _one_episode(root, script)

    (record,) = read_ledger(ledger, manifest, variants)

    assert record.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert record.error_class == "protocol_error"
    assert (record.turns_used, record.messages_used) == (1, 1)
