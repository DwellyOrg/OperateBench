"""Regressions for false run scope and usage lost on failure.

1. **False run scope.** ``build_run_manifest`` stamped every run — including one
   whose adapter is the Anthropic Messages integration — with the single status
   ``SYNTHETIC_INFRASTRUCTURE_SMOKE`` and a scope asserting that a deterministic
   fake ran and that *no model was executed*. A credentialed provider run would
   therefore have persisted, hashed into its own identity and reprinted in its
   report a statement about itself that is false. Run identity now carries an
   explicit **track**, and the track, its status and its scope are derived from
   the adapter's provider, hashed into ``run_id``, strictly loaded, checked
   against the live adapter before execution, shown in the report, and refused
   on resume when any of them has drifted.

2. **Usage on protocol failure.** A turn whose response was valid reported 11/12
   tokens; the next turn's response was text-only — a protocol failure — and
   reported 91/92. Those 91/92 tokens were spent and billed, and the adapter's
   ``last_usage()`` said so, but ``run_episode`` raised on
   ``AdapterProtocolError`` before reading it, so the durable row and the report
   omitted the provider consumption entirely. Consumption measured on a response
   that arrived is now counted whether or not the response passed protocol
   validation, and the adapter clears its measurement at the top of every
   ``next_call`` so a failure that happens before any response cannot inherit
   the previous call's numbers.

Every provider interaction here runs the real Anthropic SDK over the in-process
``httpx.MockTransport`` in ``tests.anthropic_transport``. Nothing contacts a
provider, and nothing here is evidence about a model.
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
    AdapterUsage,
    TurnDeadline,
    TurnRequest,
    build_turn_request,
    identity_for_test_double,
)
from boundarybench.compiler import Variant
from boundarybench.environment import Environment
from boundarybench.jsonsafe import JsonSafetyError
from boundarybench.ledger import (
    OUTCOME_ADAPTER_FAILURE,
    OUTCOME_MODEL_PROTOCOL_FAILURE,
    OUTCOME_PROVIDER_FAILURE,
    append_episode,
    read_ledger,
)
from boundarybench.runmanifest import (
    MANIFEST_FILENAME,
    RUN_SCOPE_PROVIDER_EXECUTION,
    RUN_SCOPE_SYNTHETIC_FAKE,
    RUN_STATUS_PROVIDER_EXECUTION,
    RUN_STATUS_SYNTHETIC_FAKE,
    RUN_TRACK_PROVIDER_EXECUTION,
    RUN_TRACK_SYNTHETIC_FAKE,
    RunLimits,
    RunManifest,
    RunManifestError,
    build_run_manifest,
    configuration_digest,
    execution_digest_of,
    load_run_manifest,
    open_run_session,
    run_track_for_provider,
)
from boundarybench.runner import (
    RunProvenanceError,
    build_episode_record,
    compiled_variants,
    execute_run,
    run_episode,
    verify_execution_provenance,
)
from boundarybench.scaffold import STANDARD_SCAFFOLD, Scaffold, load_scaffold
from boundarybench.suite import validate_suite
from tests.anthropic_transport import (
    RecordingTransport,
    error_body,
    message_body,
    scripted_client,
    text_block,
    tool_use_block,
)
from tests.conftest import SUITE_MANIFEST

#: The model these runs pin, request and record. Not a real model identifier.
PINNED_MODEL = "claude-test-20990101"


def _scaffold() -> Scaffold:
    return load_scaffold(STANDARD_SCAFFOLD)


def _deadline(seconds: float = 30.0) -> TurnDeadline:
    return TurnDeadline(remaining_seconds=seconds, cancelled=lambda: False)


def _suite_variant() -> Variant:
    return next(iter(compiled_variants(validate_suite(SUITE_MANIFEST)).values()))


def _turn_request() -> TurnRequest:
    return build_turn_request(
        scaffold=_scaffold(),
        environment=Environment(_suite_variant()),
        turns_remaining=12,
    )


def _anthropic_adapter(*steps: Any, **kwargs: Any) -> Any:
    from boundarybench.providers.anthropic_messages import AnthropicMessagesAdapter

    _transport, client = scripted_client(*steps)
    return AnthropicMessagesAdapter(model=PINNED_MODEL, client=client, **kwargs)


def _anthropic_manifest(trials: int = 1) -> RunManifest:
    from boundarybench.providers.anthropic_messages import (
        ANTHROPIC_ADAPTER_VERSION,
        ANTHROPIC_IMPLEMENTATION,
        ANTHROPIC_PROVIDER,
        AnthropicRetryPolicy,
        anthropic_settings,
    )

    return build_run_manifest(
        suite=validate_suite(SUITE_MANIFEST),
        scaffold=_scaffold(),
        provider=ANTHROPIC_PROVIDER,
        model=PINNED_MODEL,
        implementation=ANTHROPIC_IMPLEMENTATION,
        adapter_version=ANTHROPIC_ADAPTER_VERSION,
        adapter_settings=anthropic_settings(AnthropicRetryPolicy(), model=PINNED_MODEL),
        trials=trials,
        limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0),
    )


def _fake_manifest() -> RunManifest:
    from boundarybench.adapter import fake_identity

    identity = fake_identity()
    return build_run_manifest(
        suite=validate_suite(SUITE_MANIFEST),
        scaffold=_scaffold(),
        provider=identity.provider,
        model=identity.model,
        implementation=identity.implementation,
        adapter_version=identity.version,
        adapter_settings={},
        trials=1,
        limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0),
    )


def _rewrite_manifest(path: Path, **changes: Any) -> None:
    """Edit a stored manifest and recompute *both* of its digests from the result.

    The point of recomputing is that neither hash check can be what catches the
    edit. Both digests are unkeyed and their inputs are all in the file, so the
    forgery this models is the complete one: the configuration digest is taken
    over the edited configuration, and the execution digest is re-bound to the
    new configuration id so the cross-check between the two blocks also agrees.

    What survives that is the only thing that can: the stored track, status and
    scope are checked against the provider the same file records.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    execution = payload.pop("execution")
    payload.pop("configuration_id")
    payload.update(changes)
    configuration_id = configuration_digest(payload)
    payload["configuration_id"] = configuration_id
    payload["execution"] = {
        "configuration_id": configuration_id,
        "execution_id": execution["execution_id"],
        "created_at_utc": execution["created_at_utc"],
        "execution_digest": execution_digest_of(
            configuration_id=configuration_id,
            execution_id=execution["execution_id"],
            created_at_utc=execution["created_at_utc"],
        ),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", "utf-8")


# -- 1. a run manifest states which track it is on ---------------------------


def test_a_provider_run_does_not_claim_the_fake_infrastructure_scope() -> None:
    """An Anthropic run must not carry deterministic-fake scope.

    Its durable status and scope identify external-provider execution without
    asserting a score, ranking, or benchmark result.
    """
    manifest = _anthropic_manifest()

    assert manifest.run_track == RUN_TRACK_PROVIDER_EXECUTION
    assert manifest.run_status == RUN_STATUS_PROVIDER_EXECUTION
    assert manifest.run_status != RUN_STATUS_SYNTHETIC_FAKE
    scope = manifest.run_scope
    assert "no model was executed" not in scope
    assert "deterministic fake adapter" not in scope
    # What it does claim: execution of the same synthetic n = 2 suite against an
    # external provider model, and no score, ranking or benchmark result.
    assert "external provider" in scope.lower()
    for refusal in ("no score", "ranking", "benchmark"):
        assert refusal in scope.lower()


def test_a_fake_run_keeps_its_honest_fake_and_no_model_scope() -> None:
    """The other track is unchanged: a fake run still says it executed no model."""
    manifest = _fake_manifest()

    assert manifest.run_track == RUN_TRACK_SYNTHETIC_FAKE
    assert manifest.run_status == RUN_STATUS_SYNTHETIC_FAKE
    assert manifest.run_scope == RUN_SCOPE_SYNTHETIC_FAKE
    assert "no model was executed" in manifest.run_scope


def test_the_test_double_provider_is_on_the_fake_track() -> None:
    """A scripted double is not an external provider, whatever it is scripted to do."""
    assert run_track_for_provider("test-double").track == RUN_TRACK_SYNTHETIC_FAKE
    assert run_track_for_provider("fake").track == RUN_TRACK_SYNTHETIC_FAKE
    assert run_track_for_provider("anthropic").track == RUN_TRACK_PROVIDER_EXECUTION


def test_track_status_and_scope_are_all_inside_configuration_identity() -> None:
    """Each of the three is hashed, so none of them can be restated for free."""
    manifest = _anthropic_manifest()
    payload = manifest.configuration_payload()

    assert payload["track"] == RUN_TRACK_PROVIDER_EXECUTION
    assert payload["status"] == RUN_STATUS_PROVIDER_EXECUTION
    assert payload["scope"] == RUN_SCOPE_PROVIDER_EXECUTION
    assert configuration_digest(payload) == manifest.configuration_id
    for field, value in (
        ("track", RUN_TRACK_SYNTHETIC_FAKE),
        ("status", RUN_STATUS_SYNTHETIC_FAKE),
        ("scope", RUN_SCOPE_SYNTHETIC_FAKE),
    ):
        assert (
            configuration_digest({**payload, field: value}) != manifest.configuration_id
        )


def test_two_tracks_over_one_provider_are_two_different_run_ids() -> None:
    """Identity separates the tracks even before anything is validated."""
    assert _anthropic_manifest().run_id != _fake_manifest().run_id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("track", RUN_TRACK_SYNTHETIC_FAKE, id="track"),
        pytest.param("status", RUN_STATUS_SYNTHETIC_FAKE, id="status"),
        pytest.param("scope", RUN_SCOPE_SYNTHETIC_FAKE, id="scope"),
    ],
)
def test_a_rehashed_manifest_that_downgrades_its_track_is_refused(
    tmp_path: Path, field: str, value: str
) -> None:
    """Editing one of the three and recomputing the digest does not launder it.

    The stored file is self-consistent — it hashes to its own ``run_id`` — so
    only checking the three against the provider the same file records can
    refuse it.
    """
    manifest = _anthropic_manifest()
    root = tmp_path / "run"
    with open_run_session(root, manifest):
        pass
    _rewrite_manifest(root / MANIFEST_FILENAME, **{field: value})

    with pytest.raises(RunManifestError) as raised:
        load_run_manifest(root / MANIFEST_FILENAME)

    assert field in str(raised.value)
    assert "anthropic" in str(raised.value)


def test_resume_refuses_a_run_whose_stored_track_drifted(tmp_path: Path) -> None:
    """The tampered file is refused where a resume would read it, too."""
    manifest = _anthropic_manifest()
    root = tmp_path / "run"
    with open_run_session(root, manifest):
        pass
    _rewrite_manifest(root / MANIFEST_FILENAME, track=RUN_TRACK_SYNTHETIC_FAKE)

    with pytest.raises(RunManifestError), open_run_session(root, manifest):
        pass  # pragma: no cover - the session must not open


def test_an_unknown_track_is_refused_rather_than_read(tmp_path: Path) -> None:
    manifest = _anthropic_manifest()
    root = tmp_path / "run"
    with open_run_session(root, manifest):
        pass
    _rewrite_manifest(root / MANIFEST_FILENAME, track="OFFICIAL_LEADERBOARD_RUN")

    with pytest.raises(RunManifestError) as raised:
        load_run_manifest(root / MANIFEST_FILENAME)

    assert "track" in str(raised.value)


def test_a_manifest_object_cannot_be_constructed_on_the_wrong_track() -> None:
    """A manifest handed in by a caller is a claim, and this one is refused.

    ``RunManifest`` is a public frozen dataclass, so a caller can assemble one
    field by field and hash it correctly. Coherence between the provider and the
    three identity fields is therefore enforced at construction, which is the
    one place every stored, loaded and supplied manifest passes through.
    """
    manifest = _anthropic_manifest()
    fields = {name: getattr(manifest, name) for name in manifest.__dataclass_fields__}

    with pytest.raises(RunManifestError) as raised:
        RunManifest(**{**fields, "run_track": RUN_TRACK_SYNTHETIC_FAKE})

    assert RUN_TRACK_PROVIDER_EXECUTION in str(raised.value)


def test_execution_refuses_an_adapter_from_the_other_track(tmp_path: Path) -> None:
    """The live adapter's provider decides the track, and the manifest must agree.

    Checked against the adapter actually in hand rather than against the
    manifest's own copy of the provider name, because the manifest is the thing
    under suspicion: every ledger row about to be written asserts its track.
    """
    from boundarybench.adapter import PolicyFollowingFakeAdapter

    manifest = _anthropic_manifest()
    suite = validate_suite(SUITE_MANIFEST)

    with pytest.raises(RunProvenanceError) as raised:
        verify_execution_provenance(
            suite=suite,
            scaffold=_scaffold(),
            adapter=PolicyFollowingFakeAdapter(settings={}),
            manifest=manifest,
        )

    message = str(raised.value)
    assert RUN_TRACK_SYNTHETIC_FAKE in message
    assert RUN_TRACK_PROVIDER_EXECUTION in message


def test_a_written_manifest_does_not_claim_a_provider_call_happened(
    tmp_path: Path,
) -> None:
    """Planning is not calling. Creating the manifest attests to nothing else."""
    manifest = _anthropic_manifest()
    root = tmp_path / "run"

    with open_run_session(root, manifest) as session:
        assert session.paths.manifest_path.exists()
        # Nothing has run: there is no ledger, so no episode and no call.
        assert not session.paths.ledger_path.exists()

    stored = load_run_manifest(root / MANIFEST_FILENAME)
    assert "does not assert that any provider call" in stored.run_scope


def test_the_run_report_shows_the_track_status_and_scope(tmp_path: Path) -> None:
    """A whole provider-track run, reported under its own identity.

    Every episode answers one terminal call through the real SDK over the
    in-process transport, so the report is built from a ledger of episodes that
    genuinely executed the loop rather than from an in-memory summary.
    """
    from boundarybench.providers.anthropic_messages import AnthropicMessagesAdapter

    manifest = _anthropic_manifest()
    suite = validate_suite(SUITE_MANIFEST)
    transport = RecordingTransport(
        [],
        default=message_body(
            [
                tool_use_block(
                    "complete_case",
                    {
                        "disposition": "STOP",
                        "primary_reason_code": "NO_APPLICABLE_RULE",
                        "secondary_reason_codes": [],
                        "evidence_refs": [],
                    },
                )
            ]
        ),
    )
    adapter = AnthropicMessagesAdapter(model=PINNED_MODEL, client=transport.client())

    with open_run_session(tmp_path / "run", manifest) as session:
        report = execute_run(
            suite=suite, scaffold=_scaffold(), adapter=adapter, session=session
        )
    payload = report.as_dict()

    assert payload["track"] == RUN_TRACK_PROVIDER_EXECUTION
    assert payload["status"] == RUN_STATUS_PROVIDER_EXECUTION
    assert payload["scope"] == RUN_SCOPE_PROVIDER_EXECUTION
    assert payload["counts"]["executed"] == len(manifest.episode_plan)
    # The measurement is the provider's own, summed over the episodes that
    # reported one — which is every episode here, because every one got an answer.
    assert payload["usage_totals"]["input_tokens"] == 137 * payload["counts"]["completed"]
    assert payload["plan_note"]


def test_the_fake_run_report_still_states_the_fake_scope(tmp_path: Path) -> None:
    from boundarybench.adapter import PolicyFollowingFakeAdapter

    manifest = _fake_manifest()
    suite = validate_suite(SUITE_MANIFEST)

    with open_run_session(tmp_path / "run", manifest) as session:
        report = execute_run(
            suite=suite,
            scaffold=_scaffold(),
            adapter=PolicyFollowingFakeAdapter(settings={}),
            session=session,
        )
    payload = report.as_dict()

    assert payload["track"] == RUN_TRACK_SYNTHETIC_FAKE
    assert payload["status"] == RUN_STATUS_SYNTHETIC_FAKE
    assert payload["scope"] == RUN_SCOPE_SYNTHETIC_FAKE
    assert payload["usage_totals"] == {
        "input_tokens": None,
        "output_tokens": None,
        "cost_usd": None,
        "latency_seconds": None,
    }


# -- 2. what a failed turn actually cost -------------------------------------


def _episode(*steps: Any, **kwargs: Any) -> Any:
    variant = _suite_variant()
    return run_episode(
        variant=variant,
        scaffold=_scaffold(),
        adapter=_anthropic_adapter(*steps, **kwargs),
        limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0),
    )


def _valid_turn(input_tokens: int, output_tokens: int) -> Any:
    return message_body(
        [tool_use_block("read_records", {})],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _protocol_failure_turn(input_tokens: int, output_tokens: int) -> Any:
    """A response the model really produced and this scaffold cannot accept.

    Text and no tool call: the provider billed for it, and it is not one action.
    """
    return message_body(
        [text_block("I think I would rather explain myself.")],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def test_tokens_spent_on_a_protocol_invalid_response_are_still_counted() -> None:
    """A protocol-invalid response still contributes its consumed tokens.

    Turn one answers validly and costs 11/12. Turn two answers with text alone —
    a protocol failure — and costs 91/92. Those tokens were consumed; the
    episode's durable total includes both turns even though the second raises an
    ``AdapterProtocolError``.
    """
    result = _episode(_valid_turn(11, 12), _protocol_failure_turn(91, 92))

    assert result.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert result.usage.input_tokens == 11 + 91
    assert result.usage.output_tokens == 12 + 92


def test_a_first_turn_protocol_failure_reports_its_own_consumption() -> None:
    """No earlier turn to inherit from, and still a real measurement."""
    result = _episode(_protocol_failure_turn(91, 92))

    assert result.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert result.usage.input_tokens == 91
    assert result.usage.output_tokens == 92
    assert result.usage.latency_seconds is not None
    # Still no invented price: nothing in this repository knows what it cost.
    assert result.usage.cost_usd is None


def test_a_zero_token_response_is_recorded_as_zero_and_not_as_unmeasured() -> None:
    """Null and zero stay different facts on a failure row too."""
    result = _episode(_protocol_failure_turn(0, 0))

    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0


def test_a_transport_failure_with_no_response_fabricates_no_usage() -> None:
    """Nothing came back, so nothing is measured — not zero, and not stale.

    The first turn really did cost 11/12 and that stands. The second turn never
    received a response, so it contributes no tokens at all rather than a
    fabricated zero, and it does not silently re-report the first turn's.
    """
    from boundarybench.providers.anthropic_messages import AnthropicRetryPolicy

    class _NoSleep:
        def __call__(self, seconds: float) -> None:
            return None

    result = _episode(
        _valid_turn(11, 12),
        (500, error_body()),
        (500, error_body()),
        (500, error_body()),
        retry=AnthropicRetryPolicy(max_attempts=3),
        sleep=_NoSleep(),
    )

    assert result.outcome == OUTCOME_PROVIDER_FAILURE
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 12


def test_an_episode_whose_only_turn_never_answered_measures_nothing() -> None:
    result = _episode((401, error_body()))

    assert result.outcome != OUTCOME_MODEL_PROTOCOL_FAILURE
    assert result.usage == AdapterUsage()


def test_usage_is_cleared_before_the_request_is_even_built() -> None:
    """A failure before the request cannot inherit the last call's numbers.

    Usage is cleared before request serialisation. A turn whose request cannot
    be encoded therefore leaves no prior turn tokens readable through
    ``last_usage()`` — and the runner, which asks after a failure, cannot
    have banked them a second time.
    """
    adapter = _anthropic_adapter(_valid_turn(11, 12))
    adapter.next_call(_turn_request(), _deadline())
    assert adapter.last_usage().input_tokens == 11

    unencodable = dataclasses.replace(_turn_request(), scaffold_id="lone-\ud800")
    with pytest.raises(JsonSafetyError):
        adapter.next_call(unencodable, _deadline())

    assert adapter.last_usage() == AdapterUsage()


def test_one_turn_reports_the_whole_turns_consumption_across_its_attempts() -> None:
    """Retries inside a turn are one measurement: the total the turn consumed.

    The 500 carried no body to measure, so it contributes no tokens rather than
    zero ones; the answer that did arrive contributes its own counts once, not
    once per attempt; and the latency spans both attempts and the backoff
    between them, because that is the wall-clock the turn actually spent.
    """
    from boundarybench.providers.anthropic_messages import AnthropicRetryPolicy

    class _Clock:
        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            self.now += 0.25
            return self.now

    waited: list[float] = []
    adapter = _anthropic_adapter(
        (500, error_body()),
        _valid_turn(11, 12),
        retry=AnthropicRetryPolicy(max_attempts=3),
        sleep=waited.append,
        clock=_Clock(),
    )

    adapter.next_call(_turn_request(), _deadline())
    usage = adapter.last_usage()

    assert waited == [0.5]
    assert usage.input_tokens == 11
    assert usage.output_tokens == 12
    assert usage.latency_seconds is not None
    # Strictly more than one attempt's worth of clock ticks.
    assert usage.latency_seconds > 0.5


# -- 2b. precedence when reading the usage itself fails ----------------------


class _FailingUsageAdapter:
    """Raises one classified failure from ``next_call`` and another from usage.

    The whole point is that the second must not displace the first: what ended
    the episode is the protocol failure, and a broken usage endpoint is a
    measurement that is missing rather than a different outcome.
    """

    def __init__(self, usage: Any) -> None:
        self._usage = usage

    @property
    def identity(self) -> Any:
        return identity_for_test_double()

    @property
    def settings(self) -> dict[str, Any]:
        return {}

    def next_call(self, request: TurnRequest, deadline: TurnDeadline) -> AdapterCall:
        raise AdapterProtocolError("the answer carries 0 tool call(s)")

    def last_usage(self) -> AdapterUsage:
        if isinstance(self._usage, BaseException):
            raise self._usage
        return self._usage  # type: ignore[no-any-return]


def _run_with(adapter: Any) -> Any:
    return run_episode(
        variant=_suite_variant(),
        scaffold=_scaffold(),
        adapter=adapter,
        limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0),
    )


@pytest.mark.parametrize(
    "usage",
    [
        pytest.param(RuntimeError("the usage endpoint fell over"), id="raises"),
        pytest.param("11 in, 12 out", id="not_a_measurement"),
        pytest.param(AdapterUsage(input_tokens=-4), id="impossible_count"),
    ],
)
def test_a_broken_usage_endpoint_never_replaces_the_original_failure(
    usage: Any,
) -> None:
    """Deterministic precedence: the failure that ended the turn is the outcome.

    Retrieving the measurement is a best-effort addendum on a path that has
    already failed. If it raises, returns something that is not an
    ``AdapterUsage``, or reports a count that cannot be one, the turn simply
    measured nothing — and the protocol failure that actually ended the episode
    is what the row records.
    """
    result = _run_with(_FailingUsageAdapter(usage))

    assert result.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert result.error_class == "protocol_error"
    assert result.usage == AdapterUsage()


def test_a_broken_usage_endpoint_is_still_the_failure_on_a_healthy_turn() -> None:
    """The precedence only applies to an already-failed turn.

    A turn that answered correctly and then could not say what it cost is an
    adapter failure, exactly as before: there is no other fault for it to defer
    to, and an unmeasurable successful turn is not a measurement of zero.
    """

    class _Adapter(_FailingUsageAdapter):
        def next_call(self, request: TurnRequest, deadline: TurnDeadline) -> AdapterCall:
            return AdapterCall("read_records", {})

    result = _run_with(_Adapter(RuntimeError("the usage endpoint fell over")))

    assert result.outcome == OUTCOME_ADAPTER_FAILURE
    assert result.error_class == "RuntimeError"


# -- 2c. the whole path, through the ledger and the report -------------------


def test_the_durable_row_and_report_carry_the_failed_turns_tokens(
    tmp_path: Path,
) -> None:
    """End to end: measured on a refused response, persisted, read back, reported.

    The row is validated on the way back in by the ledger's own reader, so what
    the assertions inspect survived being written rather than being an in-memory
    result that might never have been storable.
    """
    from boundarybench.runner import RunReport

    manifest = _anthropic_manifest()
    variants = compiled_variants(validate_suite(SUITE_MANIFEST))
    entry = manifest.episode_plan[0]
    result = run_episode(
        variant=variants[entry.variant_id],
        scaffold=_scaffold(),
        adapter=_anthropic_adapter(_valid_turn(11, 12), _protocol_failure_turn(91, 92)),
        limits=manifest.limits,
    )
    record = build_episode_record(manifest, entry, variants[entry.variant_id], result)

    with open_run_session(tmp_path / "run", manifest) as session:
        append_episode(session.paths.ledger_path, record)
        rows = read_ledger(session.paths.ledger_path, manifest, variants)
        report = RunReport(
            manifest=manifest,
            paths=session.paths,
            records=rows,
            resumed_count=0,
            executed_count=1,
        )

    (row,) = rows
    assert row.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert row.usage["input_tokens"] == 102
    assert row.usage["output_tokens"] == 104
    assert row.usage["latency_seconds"] is not None
    assert row.usage["cost_usd"] is None

    payload = report.as_dict()
    assert payload["usage_totals"]["input_tokens"] == 102
    assert payload["usage_totals"]["output_tokens"] == 104
    assert payload["usage_totals"]["cost_usd"] is None
    assert payload["usage_measured_episodes"]["input_tokens"] == 1
    assert payload["usage_unmeasured_episodes"]["cost_usd"] == 1


def test_a_refused_response_model_still_banks_nothing(tmp_path: Path) -> None:
    """The narrow carve-out, restated where it could regress.

    Counting consumption on a protocol-invalid answer is about answers the
    *pinned* model gave badly. A response that states a different model produced
    it is refused whole, and its counts are not banked: a row attributes its
    usage to the adapter identity it carries, so banking them would attribute
    spend to a model that did not answer.
    """
    result = _episode(
        message_body([tool_use_block("read_records", {})], model="claude-OTHER-20990101")
    )

    assert result.outcome == OUTCOME_PROVIDER_FAILURE
    assert result.usage == AdapterUsage()
