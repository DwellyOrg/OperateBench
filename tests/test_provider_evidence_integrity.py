"""What a stored row has to prove before this build will read it back.

Provider telemetry adds a second kind of durable evidence: not just
what the agent did, but what the *provider* did to produce it — the attempts, the
faults, the statuses and the locally measured latencies behind each turn — plus
the coverage summary that says how much of that was measured at all.

Every one of those fields is re-derived on read rather than trusted. That is the
whole point of writing them down: a summary that is merely stored alongside its
evidence is a second opinion. The tests here are the negative half of that
contract. Each one takes a *genuine generated row* — produced by the shipped
runner and real Anthropic adapter/SDK code over a deterministic in-process fake
transport — edits one field the way a corrupted
write or a well-meaning hand-edit would, and asserts the reader refuses it and
says which claim failed.

Nothing here opens a socket or needs a credential.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from boundarybench.adapter import (
    ADAPTER_CONTRACT_VERSION,
    PolicyFollowingFakeAdapter,
    fake_identity,
)
from boundarybench.ledger import (
    MEASUREMENT_FULL,
    MEASUREMENT_PARTIAL,
    MEASUREMENT_UNMEASURED,
    SAMPLE_BEHAVIORAL_ATTEMPT,
    SAMPLE_COMPLETED,
    LedgerError,
    measurement_status_for,
    read_ledger,
)
from boundarybench.runmanifest import (
    MANIFEST_FILENAME,
    RunLimits,
    RunManifestFormatError,
    RunManifestMismatchError,
    build_run_manifest,
    load_run_manifest,
    open_run_session,
)
from boundarybench.runner import compiled_variants, execute_run
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.suite import validate_suite
from tests.anthropic_transport import (
    RecordingTransport,
    error_body,
    message_body,
    tool_use_block,
)
from tests.conftest import SUITE_MANIFEST

PINNED_MODEL = "claude-test-20990101"

LEDGER_FILENAME = "episodes.jsonl"


# -- generated adapter evidence, whose rows are then edited -----------------


def build_manifest(**overrides: Any):
    kwargs: dict[str, Any] = {
        "suite": validate_suite(SUITE_MANIFEST),
        "scaffold": load_scaffold(STANDARD_SCAFFOLD),
        "provider": "test-double",
        "model": "scripted-test-double",
        "implementation": "scripted_test_double",
        "adapter_version": ADAPTER_CONTRACT_VERSION,
        "adapter_settings": {},
        "trials": 1,
        "limits": RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0),
    }
    kwargs.update(overrides)
    return build_run_manifest(**kwargs)


def instant_retry():
    """The shipped retry policy with its waits removed, so tests do not sleep."""
    from boundarybench.providers.anthropic_messages import AnthropicRetryPolicy

    return AnthropicRetryPolicy(max_attempts=3, initial_backoff_seconds=0.0)


def anthropic_manifest(retry=None, **overrides: Any):
    """A manifest pinning exactly the settings the adapter under test reports."""
    from boundarybench.providers.anthropic_messages import (
        ANTHROPIC_ADAPTER_VERSION,
        ANTHROPIC_IMPLEMENTATION,
        ANTHROPIC_PROVIDER,
        AnthropicRetryPolicy,
        anthropic_settings,
    )

    return build_manifest(
        provider=ANTHROPIC_PROVIDER,
        model=PINNED_MODEL,
        implementation=ANTHROPIC_IMPLEMENTATION,
        adapter_version=ANTHROPIC_ADAPTER_VERSION,
        adapter_settings=anthropic_settings(
            retry or AnthropicRetryPolicy(), model=PINNED_MODEL
        ),
        **overrides,
    )


def anthropic_adapter(client, **kwargs: Any):
    from boundarybench.providers.anthropic_messages import AnthropicMessagesAdapter

    kwargs.setdefault("sleep", lambda seconds: None)
    return AnthropicMessagesAdapter(model=PINNED_MODEL, client=client, **kwargs)


def ticking(step: float = 0.25):
    """A monotonic clock that advances a fixed step on every read."""
    now = [0.0]

    def read() -> float:
        value = now[0]
        now[0] += step
        return value

    return read


def execute(root: Path, manifest, adapter):
    with open_run_session(root, manifest) as session:
        return execute_run(
            suite=validate_suite(SUITE_MANIFEST),
            scaffold=load_scaffold(STANDARD_SCAFFOLD),
            adapter=adapter,
            session=session,
        )


def ledger_rows(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (root / LEDGER_FILENAME).read_text(encoding="utf-8").splitlines()
    ]


def variants() -> dict[str, Any]:
    return compiled_variants(validate_suite(SUITE_MANIFEST))


@pytest.fixture(scope="module")
def generated_provider_evidence(tmp_path_factory: pytest.TempPathFactory):
    """One generated row whose first episode was rate limited twice.

    Built once by the shipped runner and real adapter/SDK code over a deterministic
    fake transport, then only ever read: every test below copies the row before
    editing it. The scenario is chosen for the shape of the evidence it leaves —
    several turns, one of which cost three requests and two named faults — which
    is what makes the per-turn and per-episode tallies distinguishable from each
    other and both distinguishable from the attempts they summarise.
    """
    root = tmp_path_factory.mktemp("provider-run") / "run"
    manifest = anthropic_manifest(retry=instant_retry())
    answer = message_body([tool_use_block("read_records", {})], model=PINNED_MODEL)
    transport = RecordingTransport(
        [(429, error_body()), (429, error_body())], default=answer
    )
    execute(
        root,
        manifest,
        anthropic_adapter(transport.client(), retry=instant_retry(), clock=ticking()),
    )
    row = ledger_rows(root)[0]
    telemetry = row["provider_telemetry"]
    assert len(telemetry["turns"]) > 1
    assert telemetry["turns"][0]["attempt_count"] == 3
    assert telemetry["fault_counts"] == {"provider_rate_limited": 2}
    assert telemetry["measurement_status"] == MEASUREMENT_FULL
    assert [attempt["fault"] for attempt in telemetry["turns"][0]["attempts"]] == [
        "provider_rate_limited",
        "provider_rate_limited",
        None,
    ]
    assert row["measurement"]["status"] == MEASUREMENT_FULL
    assert row["elapsed_seconds"] is not None
    return manifest, row


def refuses(manifest, row: dict[str, Any], path: Path) -> LedgerError:
    """Write one row as the only ledger line and demand the reader refuse it."""
    path.write_text(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(LedgerError) as raised:
        read_ledger(path, manifest, variants())
    return raised.value


def telemetry(row: dict[str, Any]) -> dict[str, Any]:
    return row["provider_telemetry"]


def first_attempt(row: dict[str, Any]) -> dict[str, Any]:
    return telemetry(row)["turns"][0]["attempts"][0]


def clear_every_turn_latency(row: dict[str, Any]) -> None:
    for turn in telemetry(row)["turns"]:
        turn["turn_latency_seconds"] = None


def repeat_the_first_turn_index(row: dict[str, Any]) -> None:
    turns = telemetry(row)["turns"]
    turns[1]["turn_index"] = turns[0]["turn_index"]


# -- the recorded attempts --------------------------------------------------


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        pytest.param(
            lambda row: telemetry(row)["turns"][0]["attempts"].__setitem__(0, "attempt"),
            "must be a JSON object",
            id="attempt-is-not-an-object",
        ),
        pytest.param(
            lambda row: first_attempt(row).__setitem__("index", 7),
            "'index' must be 1",
            id="attempt-is-numbered-out-of-order",
        ),
        pytest.param(
            lambda row: first_attempt(row).__setitem__("outcome", "went-fine"),
            "unknown attempt outcome",
            id="attempt-outcome-is-not-in-the-closed-set",
        ),
        pytest.param(
            lambda row: first_attempt(row).__setitem__("fault", "provider_was_rude"),
            "unknown provider fault",
            id="fault-is-not-in-the-closed-set",
        ),
        pytest.param(
            lambda row: first_attempt(row).__setitem__("outcome", "response"),
            "returned a response names none",
            id="a-response-that-also-names-a-fault",
        ),
        pytest.param(
            lambda row: first_attempt(row).__setitem__("http_status", 99),
            "'http_status' must be an HTTP status code or null",
            id="status-is-not-an-http-status",
        ),
        pytest.param(
            lambda row: first_attempt(row).__setitem__("http_status", 0),
            "'http_status' must be an HTTP status code or null",
            id="zero-is-not-the-absence-of-a-status",
        ),
        pytest.param(
            lambda row: first_attempt(row).update(
                {"response_received": False, "usage_reported": True}
            ),
            "without receiving a response",
            id="usage-reported-by-an-attempt-that-got-nothing",
        ),
        pytest.param(
            lambda row: first_attempt(row).__setitem__("response_received", "yes"),
            "'response_received' must be a boolean",
            id="a-flag-that-is-not-a-boolean",
        ),
        pytest.param(
            lambda row: first_attempt(row).__setitem__("latency_seconds", "quick"),
            "'latency_seconds' must be a non-negative finite number or null",
            id="a-latency-that-is-not-a-number",
        ),
    ],
)
def test_a_tampered_provider_attempt_is_refused(
    generated_provider_evidence,
    tmp_path: Path,
    edit: Callable[[dict[str, Any]], None],
    expected: str,
) -> None:
    """Every attempt field is a closed-set value, a small integer or a duration.

    A reader that accepted anything else would be the hole the closed structure
    exists to close: provider text has nowhere to live in a well-formed attempt
    only for as long as a malformed one is refused rather than read.
    """
    manifest, row = generated_provider_evidence
    edited = json.loads(json.dumps(row))
    edit(edited)

    error = refuses(manifest, edited, tmp_path / LEDGER_FILENAME)

    assert expected in str(error)


# -- the per-turn and per-episode summaries ---------------------------------


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        pytest.param(
            lambda row: row.__setitem__("provider_telemetry", []),
            "'provider_telemetry' must be a JSON object or null",
            id="telemetry-is-not-an-object",
        ),
        pytest.param(
            lambda row: telemetry(row).__setitem__("turns", []),
            "'turns' must not be empty",
            id="no-turns-is-spelled-null-not-empty",
        ),
        pytest.param(
            lambda row: telemetry(row)["turns"].__setitem__(0, "turn one"),
            "must be a JSON object",
            id="turn-is-not-an-object",
        ),
        pytest.param(
            lambda row: telemetry(row)["turns"][0].__setitem__("turn_index", 999),
            "'turn_index' must be a turn this episode spent",
            id="turn-index-is-not-a-turn-that-happened",
        ),
        pytest.param(
            repeat_the_first_turn_index,
            "turns are recorded once each",
            id="the-same-turn-recorded-twice",
        ),
        pytest.param(
            lambda row: telemetry(row)["turns"][0].__setitem__("attempt_count", 99),
            "is not the number of attempts this turn records",
            id="turn-attempt-count-disagrees-with-its-attempts",
        ),
        pytest.param(
            lambda row: telemetry(row).__setitem__("attempt_count", 1),
            "is not the number of attempts this episode records",
            id="episode-attempt-count-disagrees-with-its-turns",
        ),
        pytest.param(
            lambda row: telemetry(row)["turns"][0].__setitem__("fault_counts", []),
            "'fault_counts' must be a JSON object",
            id="fault-counts-are-not-an-object",
        ),
        pytest.param(
            lambda row: telemetry(row)["turns"][0].__setitem__("fault_counts", {}),
            "is not the tally of the attempts",
            id="fault-counts-erase-the-faults-they-summarise",
        ),
        pytest.param(
            lambda row: telemetry(row).__setitem__(
                "fault_counts", {"provider_rate_limited": 99}
            ),
            "is not the tally of the attempts",
            id="episode-fault-counts-inflate-their-turns",
        ),
        pytest.param(
            lambda row: telemetry(row)["turns"][0].__setitem__(
                "response_received", False
            ),
            "does not follow from the attempts",
            id="a-turn-that-denies-the-response-it-holds",
        ),
        pytest.param(
            lambda row: telemetry(row)["turns"][0].__setitem__("usage_reported", False),
            "does not follow from the attempts",
            id="a-turn-that-denies-the-usage-it-holds",
        ),
        pytest.param(
            lambda row: telemetry(row)["turns"][0].__setitem__(
                "turn_latency_seconds", "fast"
            ),
            "'turn_latency_seconds' must be a non-negative finite number or null",
            id="a-turn-latency-that-is-not-a-number",
        ),
    ],
)
def test_a_tampered_turn_or_episode_tally_is_refused(
    generated_provider_evidence,
    tmp_path: Path,
    edit: Callable[[dict[str, Any]], None],
    expected: str,
) -> None:
    """A tally is checked against the attempts it claims to summarise.

    Both levels, because they fail differently: a turn's tally can disagree with
    its own attempts while the episode total still adds up, and an episode total
    can be edited while every turn beneath it stays honest.
    """
    manifest, row = generated_provider_evidence
    edited = json.loads(json.dumps(row))
    edit(edited)

    error = refuses(manifest, edited, tmp_path / LEDGER_FILENAME)

    assert expected in str(error)


# -- how much of the row was actually measured ------------------------------


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        pytest.param(
            lambda row: telemetry(row).__setitem__(
                "measurement_status", MEASUREMENT_PARTIAL
            ),
            "does not follow from the",
            id="a-status-that-contradicts-its-measured-turns",
        ),
        pytest.param(
            lambda row: telemetry(row)["turns"][0].__setitem__(
                "turn_latency_seconds", None
            ),
            "does not follow from the",
            id="one-unmeasured-turn-is-no-longer-full-coverage",
        ),
        pytest.param(
            clear_every_turn_latency,
            "does not follow from the",
            id="no-measured-turn-is-not-full-coverage",
        ),
        pytest.param(
            lambda row: row.__setitem__("measurement", []),
            "'measurement' must be a JSON object",
            id="measurement-is-not-an-object",
        ),
        pytest.param(
            lambda row: row["measurement"].__setitem__("turns_measured", 0),
            "is not what this row's own turns and attempts produce",
            id="coverage-that-does-not-follow-from-its-evidence",
        ),
        pytest.param(
            lambda row: row["measurement"].__setitem__("attempts_total", None),
            "is not what this row's own turns and attempts produce",
            id="unmeasured-attempts-on-a-row-that-counted-them",
        ),
        pytest.param(
            lambda row: row.__setitem__("elapsed_seconds", None),
            "no such thing as an episode whose duration went unmeasured",
            id="an-episode-with-no-measured-duration",
        ),
    ],
)
def test_a_tampered_measurement_claim_is_refused(
    generated_provider_evidence,
    tmp_path: Path,
    edit: Callable[[dict[str, Any]], None],
    expected: str,
) -> None:
    """Coverage is re-derived from the turns and attempts the row itself holds."""
    manifest, row = generated_provider_evidence
    edited = json.loads(json.dumps(row))
    edit(edited)

    error = refuses(manifest, edited, tmp_path / LEDGER_FILENAME)

    assert expected in str(error)


@pytest.mark.parametrize(
    ("turns_used", "turns_measured", "expected"),
    [
        (4, 0, MEASUREMENT_UNMEASURED),
        (4, 1, MEASUREMENT_PARTIAL),
        (4, 3, MEASUREMENT_PARTIAL),
        (4, 4, MEASUREMENT_FULL),
        (0, 0, MEASUREMENT_UNMEASURED),
    ],
)
def test_measurement_coverage_is_three_answers_and_never_a_flag(
    turns_used: int, turns_measured: int, expected: str
) -> None:
    """One measured turn of four is not "measured", and it is not "unmeasured".

    The single boolean this replaced is why the distinction is asserted directly
    rather than only through a row: an episode that measured one turn of four and
    one that measured all four were reported under the same name, and a count of
    measured episodes was therefore counting two different things.
    """
    assert measurement_status_for(turns_used, turns_measured) == expected


# -- what a row is a sample of ----------------------------------------------


def test_a_completed_sample_and_a_behavioural_attempt_answer_differently(
    tmp_path: Path, generated_provider_evidence
) -> None:
    """``is_completed_sample`` is the only predicate a denominator may use."""
    identity = fake_identity()
    manifest = build_manifest(
        provider=identity.provider,
        model=identity.model,
        implementation=identity.implementation,
        adapter_version=identity.version,
        adapter_settings={},
    )
    root = tmp_path / "run"
    execute(root, manifest, PolicyFollowingFakeAdapter(settings={}))

    records = read_ledger(root / LEDGER_FILENAME, manifest, variants())
    assert len(records) == 12
    assert all(record.sample_status == SAMPLE_COMPLETED for record in records)
    assert all(record.is_completed_sample for record in records)

    provider_manifest, row = generated_provider_evidence
    path = tmp_path / "provider.jsonl"
    path.write_text(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    (attempt,) = read_ledger(path, provider_manifest, variants())
    assert attempt.sample_status == SAMPLE_BEHAVIORAL_ATTEMPT
    assert attempt.is_completed_sample is False


def test_a_run_report_names_its_completed_samples_and_its_configuration(
    tmp_path: Path,
) -> None:
    """The two accessors a caller reads instead of re-deriving the counts.

    ``run_id`` is kept as the name earlier payloads used, and it resolves
    to the *configuration* id: a reader that followed the old name must not
    silently start reading execution identity, which is a different fact about a
    different thing.
    """
    identity = fake_identity()
    manifest = build_manifest(
        provider=identity.provider,
        model=identity.model,
        implementation=identity.implementation,
        adapter_version=identity.version,
        adapter_settings={},
    )
    report = execute(tmp_path / "run", manifest, PolicyFollowingFakeAdapter(settings={}))

    assert report.run_id == report.configuration_id == manifest.configuration_id
    assert report.run_id != report.execution_id
    assert report.completed_sample_count == report.sample_counts[SAMPLE_COMPLETED] == 12


# -- the manifest's own execution record ------------------------------------


def test_an_impossible_calendar_date_is_refused_rather_than_normalised(
    tmp_path: Path,
) -> None:
    """A timestamp of the right *shape* is not yet an instant that existed.

    ``2026-02-30`` passes the pattern and no arithmetic will make it a day, so
    it is refused where it is read rather than rounded into March.
    """
    root = tmp_path / "run"
    with open_run_session(root, anthropic_manifest()):
        pass

    payload = json.loads((root / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    payload["execution"]["created_at_utc"] = "2026-02-30T12:00:00.000000Z"
    (root / MANIFEST_FILENAME).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    with pytest.raises(RunManifestFormatError) as raised:
        load_run_manifest(root / MANIFEST_FILENAME)

    assert "is not a real instant" in str(raised.value)


def test_an_execution_id_that_is_not_a_version_4_uuid_is_refused(
    tmp_path: Path,
) -> None:
    """An execution id names one thing that happened once, in one canonical form."""
    root = tmp_path / "run"
    with open_run_session(root, anthropic_manifest()):
        pass

    payload = json.loads((root / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    payload["execution"]["execution_id"] = "run-2"
    (root / MANIFEST_FILENAME).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    with pytest.raises(RunManifestFormatError) as raised:
        load_run_manifest(root / MANIFEST_FILENAME)

    assert "version-4 UUID" in str(raised.value)


def test_a_supplied_manifest_whose_execution_digest_is_wrong_never_opens_a_run(
    tmp_path: Path,
) -> None:
    """The in-memory manifest is rehashed too, before a directory exists.

    A caller can build a ``RunManifest`` object directly, and ``dataclasses``
    will happily produce one whose execution binding no longer matches the
    execution it names. That is checked on the way in, not only on the way back
    off disk, and it is checked before anything is created — so a rejected
    manifest leaves no run behind for anyone to find.
    """
    root = tmp_path / "run"
    forged = replace(anthropic_manifest(), execution_digest="0" * 64)

    with (
        pytest.raises(RunManifestMismatchError) as raised,
        open_run_session(root, forged),
    ):
        pass

    assert "execution_digest" in str(raised.value)
    assert not root.exists()


def test_a_stored_trajectory_reconstructs_to_exactly_what_was_stored(
    generated_provider_evidence,
) -> None:
    """The reconstruction is lossless, which is why the row can be re-graded.

    ``_trajectory`` rebuilds the typed object and then demands it re-encode to
    the bytes it came from. That second half cannot be made to fail from this
    reader: every field is either copied verbatim or refused, the only
    transformation is tuple/list, and JSON has no tuples — so the guard is a
    reader-independent invariant rather than a check the JSONL path can trip.
    It is left uncovered and stated here rather than silenced with a pragma,
    because the next caller that hands this function a value from somewhere
    other than ``json.loads`` is exactly what it is there for.
    """
    from boundarybench.runmanifest import canonical_json

    _manifest, row = generated_provider_evidence
    stored = row["trajectory"]

    reconstructed = _rebuild_trajectory(stored, stored["variant_id"])

    assert canonical_json(reconstructed.as_dict()) == canonical_json(stored)


def _rebuild_trajectory(stored: dict[str, Any], variant_id: str):
    from boundarybench.ledger import _trajectory

    return _trajectory(stored, variant_id, "trajectory round trip")
