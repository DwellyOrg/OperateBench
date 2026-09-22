"""Provider output trust: what a provider says never becomes durable text.

Three adversarial provider-output classes are covered, one section each.

1. The adapter accepted an answer whose ``message.model`` was not the model the
   run pinned and requested, so a ledger could record token counts and an action
   as evidence about a model that never answered.
2. Provider-controlled strings — ``stop_reason``, a content block's stated type —
   were quoted into ``AdapterProtocolError`` messages, which become a durable
   ``error_detail`` and ``failure_event.detail``.
3. An unclassified exception out of the adapter boundary had its ``str()``
   persisted, so any text the SDK or a third-party adapter chose to raise with
   landed verbatim in the ledger.

Every test here drives the real Anthropic SDK over the in-process transport in
``tests.anthropic_transport``. Nothing contacts a provider, and nothing here is
evidence about a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from boundarybench.adapter import (
    PROVIDER_FAULT_RESPONSE_INVALID,
    AdapterProtocolError,
    AdapterProviderError,
    AdapterUsage,
    TurnDeadline,
    TurnRequest,
    build_turn_request,
)
from boundarybench.compiler import Variant
from boundarybench.environment import Environment
from boundarybench.ledger import (
    OUTCOME_ADAPTER_FAILURE,
    OUTCOME_MODEL_PROTOCOL_FAILURE,
    OUTCOME_PROVIDER_FAILURE,
    append_episode,
    read_ledger,
)
from boundarybench.runmanifest import RunLimits, build_run_manifest, open_run_session
from boundarybench.runner import build_episode_record, compiled_variants, run_episode
from boundarybench.scaffold import STANDARD_SCAFFOLD, Scaffold, load_scaffold
from boundarybench.suite import validate_suite
from tests.anthropic_transport import message_body, scripted_client, tool_use_block
from tests.conftest import SUITE_MANIFEST, minimal_card

#: The model the run pins, requests and records. Not a real model identifier.
PINNED_MODEL = "claude-test-20990101"

#: What the provider claims answered instead. The marker the tests hunt for in
#: durable evidence: it is provider-controlled text, so no row may quote it.
IMPOSTOR_MODEL = "claude-DIFFERENT-20990101"

#: A provider-controlled ``stop_reason``, chosen so a substring search for it
#: cannot match anything this build writes for its own reasons.
ATTACKER_STOP_REASON = "attacker-controlled-stop-reason-7f31"


def _scaffold() -> Scaffold:
    return load_scaffold(STANDARD_SCAFFOLD)


def _variant() -> Variant:
    from boundarybench.compiler import compile_cube
    from boundarybench.schema import ConstructCard

    return compile_cube(ConstructCard.from_dict(minimal_card())).variants[0]


def _turn_request() -> TurnRequest:
    return build_turn_request(
        scaffold=_scaffold(), environment=Environment(_variant()), turns_remaining=12
    )


def _deadline(seconds: float = 30.0) -> TurnDeadline:
    return TurnDeadline(remaining_seconds=seconds, cancelled=lambda: False)


def _adapter(*steps: Any) -> Any:
    from boundarybench.providers.anthropic_messages import AnthropicMessagesAdapter

    _transport, client = scripted_client(*steps)
    return AnthropicMessagesAdapter(model=PINNED_MODEL, client=client)


# -- 1. the answer has to come from the model the run pinned -----------------


def test_an_answer_from_another_model_is_refused_before_it_is_read() -> None:
    """A response body naming a different model is a provenance failure.

    The provider answers a well-formed ``read_records`` tool call while naming a
    different model in the same body.

    Refused as a provider fault rather than as a model protocol failure: nothing
    about the model's behaviour is wrong here, the response's provenance is.
    Neither model string reaches the durable detail — the one the provider chose
    is attacker-controlled text, and the one the run pinned is already named by
    the adapter identity on every row.
    """
    adapter = _adapter(
        message_body([tool_use_block("read_records", {})], model=IMPOSTOR_MODEL)
    )

    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_turn_request(), _deadline())

    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    detail = str(raised.value)
    assert IMPOSTOR_MODEL not in detail
    assert PINNED_MODEL not in detail
    # No action was accepted, and no measurement was banked from the answer that
    # carried it: a refused response contributes neither.
    assert adapter.last_usage() == AdapterUsage()


# -- 2. no provider-controlled string reaches a durable protocol error -------


@pytest.mark.parametrize(
    ("content", "kwargs"),
    [
        pytest.param(
            [tool_use_block("read_records", {})],
            {"stop_reason": ATTACKER_STOP_REASON},
            id="stop_reason",
        ),
        pytest.param(
            [
                {"type": ATTACKER_STOP_REASON, "note": "hmm"},
                tool_use_block("read_records", {}),
            ],
            {},
            id="unknown_block_type",
        ),
    ],
)
def test_a_refused_answer_never_quotes_the_string_the_provider_chose(
    content: list[Any], kwargs: dict[str, Any]
) -> None:
    """Both parse refusals name what this build requires, not what arrived.

    The adversarial inputs are a provider-chosen ``stop_reason`` and a content
    block whose stated type is neither text nor a tool call. Both are checked
    with the same marker.

    Both are refusals *of* provider input, so the input is the one thing that
    may not be in them. The fixed half of the sentence still says which contract
    was broken, which is what an operator acts on.
    """
    adapter = _adapter(message_body(content, **kwargs))

    with pytest.raises(AdapterProtocolError) as raised:
        adapter.next_call(_turn_request(), _deadline())

    assert ATTACKER_STOP_REASON not in str(raised.value)


# -- 3. the adapter boundary, end to end through the ledger ------------------


def _anthropic_manifest(trials: int = 1) -> Any:
    """A run manifest that records exactly the adapter the episode will run."""
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


def _recorded(tmp_path: Path, *steps: Any) -> Any:
    """Run one episode against the scripted provider and read its row back.

    Through :func:`run_episode`, :func:`build_episode_record`, the ledger's
    append *and* the ledger's own validation on the way back in — so what the
    assertions inspect is a row that survived being persisted, not an in-memory
    result that might never have been writable.
    """
    manifest = _anthropic_manifest()
    variants = compiled_variants(validate_suite(SUITE_MANIFEST))
    entry = manifest.episode_plan[0]
    result = run_episode(
        variant=variants[entry.variant_id],
        scaffold=_scaffold(),
        adapter=_adapter(*steps),
        limits=manifest.limits,
    )
    record = build_episode_record(manifest, entry, variants[entry.variant_id], result)
    with open_run_session(tmp_path / "run", manifest) as session:
        append_episode(session.paths.ledger_path, record)
        (row,) = read_ledger(session.paths.ledger_path, manifest, variants)
    return result, row


def test_a_mismatched_response_model_is_a_recorded_provider_failure(
    tmp_path: Path,
) -> None:
    """The whole path: refused, classified, persisted, and quoting nothing.

    A provider that answers a well-formed action while naming a different model
    ends the episode as a provider failure with no step taken, no usage banked
    and no evaluation. Neither model identifier appears anywhere in the durable
    evidence — not in ``error_detail``, not in ``failure_event.detail`` — and
    the row is one the ledger reads back rather than one that merely wrote.
    """
    result, row = _recorded(
        tmp_path,
        message_body([tool_use_block("read_records", {})], model=IMPOSTOR_MODEL),
    )

    assert result.outcome == OUTCOME_PROVIDER_FAILURE
    assert result.error_class == PROVIDER_FAULT_RESPONSE_INVALID
    assert row.failure_event is not None
    assert row.failure_event["kind"] == PROVIDER_FAULT_RESPONSE_INVALID
    # Not a model protocol failure: the model's answer was well formed, and the
    # service's account of who produced it is what failed.
    assert result.outcome != OUTCOME_MODEL_PROTOCOL_FAILURE

    durable = f"{row.error_detail} {row.failure_event['detail']}"
    assert IMPOSTOR_MODEL not in durable
    assert PINNED_MODEL not in durable
    # Nothing was accepted from the refused answer: no action reached the
    # environment, and the provider's token counts were never banked.
    assert result.trajectory.steps == ()
    assert row.usage["input_tokens"] is None
    assert row.usage["output_tokens"] is None
    assert row.evaluation is None
    # And the run still records which model it *pinned*, in the one place a row
    # states it: the adapter identity, which is not a redacted diagnostic.
    assert row.adapter["model"] == PINNED_MODEL


def test_an_attacker_chosen_stop_reason_never_reaches_the_ledger(
    tmp_path: Path,
) -> None:
    """A provider-chosen marker never enters durable protocol-failure details."""
    result, row = _recorded(
        tmp_path,
        message_body(
            [tool_use_block("read_records", {})], stop_reason=ATTACKER_STOP_REASON
        ),
    )

    assert result.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert row.failure_event is not None
    assert row.failure_event["kind"] == "protocol_error"
    assert ATTACKER_STOP_REASON not in f"{row.error_detail} {row.failure_event['detail']}"


def test_an_unclassified_adapter_exception_is_recorded_by_class_and_not_by_text(
    tmp_path: Path,
) -> None:
    """An SDK error this contract does not classify keeps its class, loses its text.

    ``anthropic.AnthropicError`` raised from the transport is deliberately *not*
    a provider fault — see the adapter's own taxonomy — so it reaches the
    runner's generic adapter-exception catch. The catch persists the exception
    class but not ``str(exc)`` because SDK exception messages can quote
    a response body, a header or a request back at whoever reads the ledger.

    The class survives, because that is what an operator debugs from and what
    the row is filed under; the message does not.
    """
    import anthropic

    result, row = _recorded(tmp_path, anthropic.AnthropicError(ATTACKER_STOP_REASON))

    assert result.outcome == OUTCOME_ADAPTER_FAILURE
    assert row.error_class == "AnthropicError"
    assert row.failure_event is not None
    assert row.failure_event["exception_type"] == "AnthropicError"
    durable = f"{row.error_detail} {row.failure_event['detail']}"
    assert ATTACKER_STOP_REASON not in durable
    # Named, and not a traceback: the class is stated in plain words.
    assert "AnthropicError" in durable
    assert "Traceback" not in durable
