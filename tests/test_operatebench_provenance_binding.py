"""What a v2 artefact claims about *who ran it*, and why each claim is checkable.

An artefact says four things about provenance: which agent ran, what kind of
agent it was, what produced its decisions, and what the provider session looked
like. Holding each of those to its own shape and never comparing them leaves the
cheapest forgery there is — take a deterministic run, relabel the execution
record as a model's, and the file validates and replays clean under a model's
name. This suite is the standing check that it does not.

Four properties are asserted.

**The execution record, the decision tape and the identity describe one run.**
Calls, attempts and decisions have one cardinality between them; attempt *i* and
decision *i* are the same invocation and turn; an attempt's classification and
the kind of the decision beside it agree; an in-process or replayed execution
names no model, no protocol, no ceiling, makes no calls and records no attempts.

**A recorded provider identity is load-bearing.** Playback rebuilds the request
each decision answered from the model, protocol version and output ceiling the
record names, and requires it to hash to the attempt recorded beside that
decision — so editing the model a run is attributed to moves every request
digest in it. It reaches no provider while doing so, and the count is read back
rather than assumed.

**An excluded execution has no episode artefact.** The contract chosen here is
the fail-closed one of the two available: this build refuses to read an
``excluded=True`` episode artefact rather than defining a strict excluded shape.
The reason is that the engine cannot produce one — a provider fault propagates
out of the runner before an artefact exists — so a permissive field would be a
state nothing writes and nothing can check, and the state it would admit is
exactly the contradiction this file exists to prevent: a scored, reliable,
completed evaluation sitting beside a provider fault. Recording an excluded
execution needs a ledger whose result-bearing fields are the attempts and the
fault rather than a terminal. That is a separate artefact, not a flag on this
one.

**Every recorded outcome has an exact body.** ACT, WAIT, ASK, ESCALATE, COMPLETE
and MALFORMED each have one shape, unknown keys are refused, and the two shape
tables are checked against the vocabularies they describe so drift cannot turn
into a ``KeyError`` raised out of a reader.

Two notes for the lanes this build is integrated with.

*v1 compatibility.* This build still **validates** the byte-frozen v1 artefact,
and :class:`TestV1CompatibilityIsStated` pins that. It does not replay it, and
that is the case this file anticipated: a v1 replay re-executes the shipped
agent against the current spec, engine and evaluator, so a semantic change in
any of them is a divergence rather than a silent pass — and the operation's
authored semantics have since moved. The answer taken is the named
incompatibility, asserted by digest, rather than a rewritten fixture or a
comparison narrowed until it agrees. What keeps the v1 *path* executable
evidence is a v1-shaped artefact derived from a run this build performs (see
:mod:`tests.artifact_shapes`), which replays here with nothing carried forward.

*Fields a later contract adds.* The event reader requires ``verdict_code`` to be
a non-empty string on every delivered event, and the evaluation reader accepts
any number of dimensions with any names, so a tenth dimension is schema
compatible with this reader as it stands. Both are asserted below so a change to
either is a test failure here rather than a surprise downstream.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from operatebench.agents.model import (
    CLASSIFICATIONS,
    MAX_OUTPUT_TOKENS,
    READABLE_CLASSIFICATIONS,
    UNCLASSIFIED_CODE,
    MalformedModelOutcome,
    ModelAgent,
)
from operatebench.agents.playback import (
    MALFORMED_KIND,
    RecordedModelAgent,
    RecordedOutcomeAgent,
    RequestIdentityMismatchError,
    decision_record,
    outcome_from_record,
    tape_from_records,
)
from operatebench.agents.transport import (
    FAULTS,
    RETRY_COUNT,
    SOURCE_DETERMINISTIC,
    SOURCE_MODEL,
    SOURCE_RECORDED,
    ForbiddenTransport,
    ProviderFailure,
)
from operatebench.artifact import (
    _OUTCOME_SHAPES,
    _ROW_SHAPES,
    DETERMINISTIC_AGENT_KINDS,
    KIND_MODEL,
    KIND_RECORDED,
    REPRODUCIBLE_MODEL_PROTOCOLS,
    SUPPORTED_MODEL_PROTOCOLS,
    _content_digest,
    build_artifact,
    derived_agent_kind,
    outcome_shape_coverage_problem,
    read_artifact,
    replay_artifact,
    row_shape_coverage_problem,
    validate_artifact,
)
from operatebench.cli import EXIT_ERROR, main
from operatebench.core.errors import ArtifactError
from operatebench.core.ledger import RECORD_TYPES
from operatebench.core.outcomes import OUTCOME_KINDS
from operatebench.core.retrieval import RETRIEVE_KIND
from operatebench.domains.lettings.maintenance.agents import AGENTS
from operatebench.domains.lettings.maintenance.spec import OperationSpec, load_spec
from operatebench.execution_ledger import EXECUTION_LEDGER_VERSION
from operatebench.runner import run_episode
from tests.artifact_shapes import as_v1_shape
from tests.model_transport import TEST_MODEL, ReferenceDrivenTransport
from tests.provider_evidence_runs import bound_model_artifact

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "examples/operatebench/maintenance_v0_1.yaml"
FROZEN_V1 = Path(__file__).resolve().parent / "fixtures/artifact_v1_reference_V1.json"

MODEL_AGENT_ID = "model_reference"


@pytest.fixture(scope="module")
def spec() -> OperationSpec:
    return load_spec(FIXTURE)


@pytest.fixture(scope="module")
def deterministic(spec: OperationSpec) -> dict[str, Any]:
    return build_artifact(run_episode(spec, "V1", "reference"))


@pytest.fixture(scope="module")
def model_artifact(spec: OperationSpec, tmp_path_factory: Any) -> dict[str, Any]:
    """A genuine model artefact, with the execution ledger that witnessed it.

    Contract 7 gives no way to assemble one without a journal behind it: a model
    run with no complete scored ledger is refused at ``build_artifact`` rather
    than written with a null binding. So this fixture runs the real thing —
    through the real OpenAI SDK over an in-process mock transport — and the
    adversarial edits below are made to a document that was actually produced.
    """
    return bound_model_artifact(tmp_path_factory, spec, agent_id=MODEL_AGENT_ID)


def copy_of(payload: Mapping[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(json.dumps(payload))
    return body


def with_decisions(payload: Mapping[str, Any], decisions: list[Any]) -> dict[str, Any]:
    """A copy carrying an edited tape, with the digest the reader checks redone."""
    body = copy_of(payload)
    body["decisions"] = decisions
    body["decisions_digest_sha256"] = _content_digest(decisions)
    return body


# ------------------------------------------------- execution, tape and identity


class TestTheExecutionRecordAndTheTapeDescribeOneRun:
    def test_a_genuine_model_artefact_is_accepted_and_replays(
        self, spec: OperationSpec, model_artifact: Mapping[str, Any]
    ) -> None:
        # The control. Everything below is one edit away from this record, so a
        # refusal here would make the rest of the suite vacuous.
        validate_artifact(model_artifact)
        assert replay_artifact(spec, model_artifact).ok

    @pytest.mark.parametrize(
        ("bogus", "refusal"),
        [
            (0, "more attempts than calls"),
            (3, "more attempts than calls"),
            (999, "calls nobody made"),
        ],
    )
    def test_a_call_count_that_is_not_the_number_of_attempts_is_refused(
        self, model_artifact: Mapping[str, Any], bogus: int, refusal: str
    ) -> None:
        """Both directions, and each says which way the count is wrong.

        Fewer calls than attempts is refused as a record holding attempts no
        call was made for; more is refused as calls nobody made. The recorded
        run makes more than three, so 0 and 3 land on the first and 999 on the
        second.
        """
        body = copy_of(model_artifact)
        body["agent_execution"]["transport_calls"] = bogus
        with pytest.raises(ArtifactError, match=refusal):
            validate_artifact(body)

    def test_a_removed_attempt_is_refused(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        body = copy_of(model_artifact)
        body["agent_execution"]["attempts"].pop()
        body["agent_execution"]["transport_calls"] -= 1
        with pytest.raises(ArtifactError, match="describe different runs"):
            validate_artifact(body)

    def test_an_extra_attempt_is_refused(self, model_artifact: Mapping[str, Any]) -> None:
        body = copy_of(model_artifact)
        attempts = body["agent_execution"]["attempts"]
        extra = copy_of(attempts[-1])
        extra["turn_index"] += 1
        extra["request_digest_sha256"] = "a" * 64
        attempts.append(extra)
        body["agent_execution"]["transport_calls"] += 1
        with pytest.raises(ArtifactError, match="describe different runs"):
            validate_artifact(body)

    def test_attempts_out_of_order_are_refused(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        body = copy_of(model_artifact)
        attempts = body["agent_execution"]["attempts"]
        attempts[0], attempts[1] = attempts[1], attempts[0]
        with pytest.raises(ArtifactError, match="reordered, removed or added"):
            validate_artifact(body)

    def test_an_attempt_on_an_invocation_its_decision_was_not_made_on_is_refused(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        body = copy_of(model_artifact)
        body["agent_execution"]["attempts"][0]["invocation_index"] += 5
        with pytest.raises(ArtifactError, match="reordered, removed or added"):
            validate_artifact(body)

    def test_two_attempts_with_one_request_identity_are_refused(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        body = copy_of(model_artifact)
        attempts = body["agent_execution"]["attempts"]
        attempts[1]["request_digest_sha256"] = attempts[0]["request_digest_sha256"]
        with pytest.raises(ArtifactError, match="one call written down twice"):
            validate_artifact(body)

    def test_a_classified_attempt_beside_a_real_decision_is_refused(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        body = copy_of(model_artifact)
        body["agent_execution"]["attempts"][0]["outcome"] = "classified"
        with pytest.raises(ArtifactError, match="beside a"):
            validate_artifact(body)

    def test_a_decided_attempt_beside_a_malformed_decision_is_refused(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        decisions = copy_of(model_artifact)["decisions"]
        decisions[0]["outcome"] = {
            "kind": MALFORMED_KIND,
            "code": "MODEL_NO_TOOL_CALL",
        }
        body = with_decisions(model_artifact, decisions)
        assert body["agent_execution"]["attempts"][0]["outcome"] == "decided"
        with pytest.raises(ArtifactError, match="beside a 'MALFORMED' decision"):
            validate_artifact(body)

    def test_a_failed_attempt_cannot_appear_in_an_episode_artefact(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        body = copy_of(model_artifact)
        body["agent_execution"]["attempts"][0]["outcome"] = "failed"
        body["agent_execution"]["attempts"][0]["fault"] = FAULTS[0]
        with pytest.raises(ArtifactError, match="produced no decision at all"):
            validate_artifact(body)

    def test_a_decision_on_an_invocation_the_episode_never_reached_is_refused(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        decisions = copy_of(model_artifact)["decisions"]
        decisions[-1]["invocation_index"] = int(model_artifact["agent_invocations"]) + 1
        body = with_decisions(model_artifact, decisions)
        body["agent_execution"]["attempts"][-1]["invocation_index"] = decisions[-1][
            "invocation_index"
        ]
        with pytest.raises(ArtifactError, match="the episode never reached"):
            validate_artifact(body)

    def test_a_model_run_must_name_its_model_and_a_protocol_this_build_speaks(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        body = copy_of(model_artifact)
        body["agent_execution"]["model"] = None
        with pytest.raises(ArtifactError, match="must name the model"):
            validate_artifact(body)

        body = copy_of(model_artifact)
        body["agent_execution"]["protocol_version"] = "operatebench.model.v9"
        with pytest.raises(ArtifactError, match="not a model protocol this build"):
            validate_artifact(body)
        # Three protocols are *read*, one is reproduced. A v1- or v2-protocol
        # record can only sit inside an artefact contract this build no longer
        # reproduces, so nothing rebuilds a request under a protocol whose
        # observation projection this build cannot produce.
        assert SUPPORTED_MODEL_PROTOCOLS == (
            "operatebench.model.v1",
            "operatebench.model.v2",
            "operatebench.model.v3",
            "operatebench.model.v4",
        )
        assert REPRODUCIBLE_MODEL_PROTOCOLS == ("operatebench.model.v4",)

    def test_a_model_run_must_state_the_ceiling_it_was_requested_under(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        assert model_artifact["agent_execution"]["max_output_tokens"] == (
            MAX_OUTPUT_TOKENS
        )
        body = copy_of(model_artifact)
        body["agent_execution"]["max_output_tokens"] = None
        with pytest.raises(ArtifactError, match="must state the output ceiling"):
            validate_artifact(body)

    def test_an_in_process_run_carries_no_provider_identity_at_all(
        self, deterministic: Mapping[str, Any]
    ) -> None:
        execution = deterministic["agent_execution"]
        assert execution["outcome_source"] == SOURCE_DETERMINISTIC
        assert (execution["model"], execution["protocol_version"]) == (None, None)
        assert execution["max_output_tokens"] is None
        assert (execution["transport_calls"], execution["attempts"]) == (0, [])
        assert execution["retry_count"] == RETRY_COUNT == 0

        for field, value in (
            ("model", TEST_MODEL),
            ("protocol_version", SUPPORTED_MODEL_PROTOCOLS[0]),
            ("max_output_tokens", MAX_OUTPUT_TOKENS),
        ):
            body = copy_of(deterministic)
            body["agent_execution"][field] = value
            with pytest.raises(ArtifactError, match="never called"):
                validate_artifact(body)

    def test_an_in_process_run_cannot_record_provider_attempts(
        self, deterministic: Mapping[str, Any], model_artifact: Mapping[str, Any]
    ) -> None:
        """Two rules say this, and neither needs a third to repeat them.

        An in-process execution is held to zero calls, and no record may hold
        more attempts than calls were made. An attempt on a deterministic run
        therefore has nowhere to sit.
        """
        body = copy_of(deterministic)
        body["agent_execution"]["attempts"] = copy_of(model_artifact)["agent_execution"][
            "attempts"
        ][:1]
        with pytest.raises(ArtifactError, match="more attempts than calls"):
            validate_artifact(body)

        body = copy_of(deterministic)
        body["agent_execution"]["transport_calls"] = 1
        with pytest.raises(ArtifactError, match="has a provider to call"):
            validate_artifact(body)


# ------------------------------------------------------------- agent identity


class TestAgentIdentityIsDerivedNotBelieved:
    def test_the_kind_of_a_registered_agent_comes_from_the_registry(self) -> None:
        for agent_id, entry in AGENTS.items():
            assert derived_agent_kind(agent_id, SOURCE_DETERMINISTIC) == entry.kind
        assert set(DETERMINISTIC_AGENT_KINDS) == {"reference", "negative"}

    def test_the_kind_of_an_unregistered_agent_comes_from_the_source(self) -> None:
        assert derived_agent_kind("anything_at_all", SOURCE_MODEL) == KIND_MODEL
        assert derived_agent_kind("anything_at_all", SOURCE_RECORDED) == KIND_RECORDED

    def test_an_unknown_deterministic_agent_is_refused(
        self, deterministic: Mapping[str, Any]
    ) -> None:
        body = copy_of(deterministic)
        body["agent_id"] = "no_such_agent_anywhere"
        with pytest.raises(ArtifactError, match="not an agent this build registers"):
            validate_artifact(body)
        with pytest.raises(ArtifactError):
            derived_agent_kind("no_such_agent_anywhere", SOURCE_DETERMINISTIC)

    def test_an_unknown_agent_in_a_v1_artefact_is_refused(self) -> None:
        body = read_artifact(FROZEN_V1)
        body["agent_id"] = "no_such_agent_anywhere"
        with pytest.raises(ArtifactError, match="not an agent this build registers"):
            validate_artifact(body)

    def test_a_deterministic_run_relabelled_as_a_model_run_is_refused(
        self, deterministic: Mapping[str, Any], model_artifact: Mapping[str, Any]
    ) -> None:
        """The cheapest forgery: one agent's work published under another's name."""
        body = copy_of(deterministic)
        body["agent_kind"] = KIND_MODEL
        body["agent_execution"] = copy_of(model_artifact)["agent_execution"]
        with pytest.raises(ArtifactError, match="relabels one run as another"):
            validate_artifact(body)

    def test_a_registered_agent_id_cannot_be_carried_by_a_model_record(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        body = copy_of(model_artifact)
        body["agent_id"] = "reference"
        with pytest.raises(ArtifactError, match="relabels one run as another"):
            validate_artifact(body)

    def test_a_model_record_must_carry_the_model_kind(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        for kind in ("reference", "negative", KIND_RECORDED, "something_else"):
            body = copy_of(model_artifact)
            body["agent_kind"] = kind
            with pytest.raises(ArtifactError, match="for a 'model' execution"):
                validate_artifact(body)

    def test_a_deterministic_record_must_carry_a_registered_kind(
        self, deterministic: Mapping[str, Any]
    ) -> None:
        body = copy_of(deterministic)
        body["agent_kind"] = KIND_MODEL
        with pytest.raises(ArtifactError, match="agent registry declares"):
            validate_artifact(body)

    def test_a_mislabelled_control_is_reported_as_a_difference_not_a_refusal(
        self, spec: OperationSpec, deterministic: Mapping[str, Any]
    ) -> None:
        """One kind claim is left to replay, on purpose.

        A deterministic record whose ``agent_kind`` is the *other* registered
        kind is well-formed, and replay re-derives the kind from the registry,
        so the report names the field that diverged. A refusal here would lose
        that: the operator would learn the file was rejected rather than which
        claim in it was false.
        """
        body = copy_of(deterministic)
        body["agent_kind"] = "negative"
        report = replay_artifact(spec, body)
        assert not report.ok
        assert not report.sections["identity"]["agent_kind"]
        assert any("agent_kind" in line for line in report.differences)

    def test_a_playback_names_every_field_it_carried(
        self, spec: OperationSpec, model_artifact: Mapping[str, Any]
    ) -> None:
        report = replay_artifact(spec, model_artifact)
        assert report.ok
        assert set(report.carried_forward) == {
            "agent_execution",
            "agent_id",
            "agent_kind",
            "decisions",
            "decisions_digest_sha256",
            # Contract 7. A replay reaches no provider, so it cannot re-derive
            # which journal witnessed the original session; the binding is
            # carried, and the report names it rather than comparing a field it
            # took from the record it is comparing against.
            "provider_execution",
        }
        assert report.total_comparison is False, (
            "identity_matches on a playback is a verdict on the identity fields "
            "that were re-derived, not a claim that the whole identity was compared"
        )


# ------------------------------------------------- the recorded model identity


class TestARecordedModelIdentityIsLoadBearing:
    @pytest.mark.parametrize(
        ("field", "value"),
        [("model", "a-different-model"), ("max_output_tokens", 512)],
    )
    def test_editing_the_provider_identity_breaks_every_request_digest(
        self,
        spec: OperationSpec,
        model_artifact: Mapping[str, Any],
        field: str,
        value: Any,
    ) -> None:
        body = copy_of(model_artifact)
        body["agent_execution"][field] = value
        if field == "model":
            # The binding names the model too, and the reader refuses a record
            # whose two statements of it disagree — before a replay happens. So
            # the forgery has to be the *consistent* one to reach the claim
            # under test: edit both, and the request digests are still what
            # catch it.
            body["provider_execution"]["model"] = value
        validate_artifact(body)  # well-formed: the forgery is not a shape error
        with pytest.raises(ArtifactError, match="not a label beside them"):
            replay_artifact(spec, body)

    def test_a_binding_and_an_execution_record_that_name_two_models_are_refused(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        """The inconsistent half of the forgery above, refused by the reader."""
        body = copy_of(model_artifact)
        body["agent_execution"]["model"] = "a-different-model"
        with pytest.raises(ArtifactError, match="one run asked one model"):
            validate_artifact(body)

    def test_a_request_digest_that_is_not_the_one_the_run_produces_is_refused(
        self, spec: OperationSpec, model_artifact: Mapping[str, Any]
    ) -> None:
        body = copy_of(model_artifact)
        body["agent_execution"]["attempts"][2]["request_digest_sha256"] = "b" * 64
        with pytest.raises(ArtifactError, match="not a label beside them"):
            replay_artifact(spec, body)

    def test_the_rebuilt_request_check_reaches_no_provider(
        self, spec: OperationSpec, model_artifact: Mapping[str, Any]
    ) -> None:
        from operatebench.agents.playback import tape_from_records
        from operatebench.runner import _reproduce_episode, playback_episode

        tape = tape_from_records(model_artifact["decisions"])
        execution = model_artifact["agent_execution"]
        agent = RecordedModelAgent(
            tape,
            agent_id=MODEL_AGENT_ID,
            model=str(execution["model"]),
            max_output_tokens=int(execution["max_output_tokens"]),
            request_digests=[
                str(attempt["request_digest_sha256"]) for attempt in execution["attempts"]
            ],
        )
        # The private reproduction surface, because this is a reproduction: the
        # persisted identity is what the rebuilt requests are hashed over. The
        # public run API mints its own and could not be used here.
        _reproduce_episode(
            spec,
            "V1",
            MODEL_AGENT_ID,
            operation_instance_id=str(model_artifact["operation_instance_id"]),
            agent_factory=lambda: agent,
            agent_kind=KIND_MODEL,
        )
        assert agent.provider_calls == 0, (
            "a replay that reached a provider would be a second provider run, and "
            "the count is read back rather than assumed"
        )
        # And the ordinary tape playback still refuses a provider call too.
        assert playback_episode(
            spec,
            "V1",
            MODEL_AGENT_ID,
            tape,
            operation_instance_id=str(model_artifact["operation_instance_id"]),
        ).outcome.status

    def test_the_mismatch_is_raised_as_a_playback_refusal(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        assert issubclass(RequestIdentityMismatchError, ArtifactError)


# ------------------------------------------------------------------ exclusion


class TestAnExcludedExecutionHasNoEpisodeArtefact:
    @pytest.mark.parametrize("fault", FAULTS)
    def test_an_excluded_record_is_refused_whatever_the_fault(
        self, model_artifact: Mapping[str, Any], fault: str
    ) -> None:
        body = copy_of(model_artifact)
        body["agent_execution"].update(
            {
                "excluded": True,
                "exclusion_code": fault,
                "exclusion_detail": f"execution stopped at the provider ({fault})",
            }
        )
        with pytest.raises(ArtifactError, match="no episode artefact in this build"):
            validate_artifact(body)

    def test_an_excluded_record_is_refused_even_beside_a_reliable_evaluation(
        self, spec: OperationSpec, model_artifact: Mapping[str, Any]
    ) -> None:
        body = copy_of(model_artifact)
        assert body["evaluation"]["reliable"] is True
        assert body["evaluation"]["legitimate_completion"] is True
        assert body["terminal_outcome"] == "completed_successfully"
        body["agent_execution"].update(
            {
                "excluded": True,
                "exclusion_code": "provider_transport",
                "exclusion_detail": "socket closed",
            }
        )
        with pytest.raises(ArtifactError, match="no episode artefact in this build"):
            replay_artifact(spec, body)

    def test_a_half_stated_exclusion_is_still_named_as_one(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        for excluded, code in ((True, None), (False, FAULTS[0])):
            body = copy_of(model_artifact)
            body["agent_execution"]["excluded"] = excluded
            body["agent_execution"]["exclusion_code"] = code
            with pytest.raises(ArtifactError, match="no third state"):
                validate_artifact(body)

    def test_the_engine_writes_no_artefact_for_a_run_that_stopped_at_the_provider(
        self, spec: OperationSpec
    ) -> None:
        """Why the reader's refusal is fail-closed rather than restrictive.

        A provider fault leaves the runner as a :class:`ProviderFailure`, so
        there is no ``EpisodeRun`` for :func:`build_artifact` to be called on
        and no artefact to write. The refusal above therefore closes a shape
        nothing produces, and the exclusion record travels on the exception —
        with every attempt already made — for a caller that keeps its own
        execution ledger.
        """
        transport = ReferenceDrivenTransport()
        with pytest.raises(ProviderFailure) as caught:
            run_episode(
                spec,
                "V1",
                MODEL_AGENT_ID,
                agent_factory=lambda: ModelAgent(
                    transport,
                    model=TEST_MODEL,
                    agent_id=MODEL_AGENT_ID,
                    max_transport_calls=3,
                ),
                agent_kind=KIND_MODEL,
            )
        record = caught.value.record
        assert record.excluded is True
        assert record.exclusion_code == "provider_budget"
        assert len(record.attempts) == 3
        assert record.max_output_tokens == MAX_OUTPUT_TOKENS


# ------------------------------------------------------------- outcome bodies


class TestEveryRecordedOutcomeHasAnExactBody:
    def test_the_outcome_shapes_cover_exactly_the_recordable_kinds(self) -> None:
        assert outcome_shape_coverage_problem() is None
        assert set(_OUTCOME_SHAPES) == {*OUTCOME_KINDS, MALFORMED_KIND, RETRIEVE_KIND}

    def test_the_row_shapes_cover_exactly_the_ledger_vocabulary(self) -> None:
        assert row_shape_coverage_problem() is None
        assert set(_ROW_SHAPES) == set(RECORD_TYPES), (
            "a record type with no row shape would reach the reader as a KeyError "
            "rather than a named refusal"
        )

    def test_the_checker_notices_a_record_type_with_no_shape(self) -> None:
        import operatebench.artifact as artifact

        original = artifact.RECORD_TYPES
        try:
            artifact.RECORD_TYPES = (*original, "a_type_nobody_described")
            problem = artifact.row_shape_coverage_problem()
        finally:
            artifact.RECORD_TYPES = original
        assert problem is not None and "a_type_nobody_described" in problem

    def test_the_checker_notices_a_kind_with_no_outcome_shape(self) -> None:
        import operatebench.artifact as artifact

        original = artifact._OUTCOME_SHAPES
        try:
            artifact._OUTCOME_SHAPES = {
                name: shape for name, shape in original.items() if name != "COMPLETE"
            }
            problem = artifact.outcome_shape_coverage_problem()
        finally:
            artifact._OUTCOME_SHAPES = original
        assert problem is not None and "COMPLETE" in problem

    @pytest.mark.parametrize("kind", sorted(_OUTCOME_SHAPES))
    def test_every_kind_refuses_a_key_this_build_does_not_write(
        self, model_artifact: Mapping[str, Any], kind: str
    ) -> None:
        decisions = copy_of(model_artifact)["decisions"]
        decisions[0]["outcome"] = {**_example_outcome(kind), "provider_text": "prose"}
        body = with_decisions(model_artifact, decisions)
        with pytest.raises(ArtifactError, match="unknown field"):
            validate_artifact(body)

    @pytest.mark.parametrize("kind", sorted(_OUTCOME_SHAPES))
    def test_every_kind_refuses_a_body_missing_one_of_its_fields(
        self, model_artifact: Mapping[str, Any], kind: str
    ) -> None:
        example = _example_outcome(kind)
        for name in sorted(set(example) - {"kind"}):
            decisions = copy_of(model_artifact)["decisions"]
            decisions[0]["outcome"] = {
                key: value for key, value in example.items() if key != name
            }
            body = with_decisions(model_artifact, decisions)
            with pytest.raises(ArtifactError, match="missing required field"):
                validate_artifact(body)

    @pytest.mark.parametrize(
        ("kind", "field", "value"),
        [
            ("ACT", "action_type", ""),
            ("ACT", "payload", ["not", "a", "mapping"]),
            ("ACT", "evidence_refs", "evidence_1"),
            ("ACT", "rationale", 7),
            ("WAIT", "wake_on", {"supplier_visit_response": True}),
            ("WAIT", "fallback_after_minutes", 0),
            ("WAIT", "fallback_after_minutes", "later"),
            ("ASK", "recipient_actor_id", None),
            ("ASK", "wait", {"kind": "ACT"}),
            ("ESCALATE", "deadline_after_minutes", -1),
            ("ESCALATE", "exception_type", ""),
            ("COMPLETE", "evidence_refs", [""]),
            (MALFORMED_KIND, "code", "MODEL_INVENTED_BY_A_PROVIDER"),
        ],
    )
    def test_a_field_of_the_wrong_shape_is_refused(
        self, model_artifact: Mapping[str, Any], kind: str, field: str, value: Any
    ) -> None:
        decisions = copy_of(model_artifact)["decisions"]
        decisions[0]["outcome"] = {**_example_outcome(kind), field: value}
        body = with_decisions(model_artifact, decisions)
        with pytest.raises(ArtifactError):
            validate_artifact(body)

    def test_a_nested_ask_wait_is_held_to_the_wait_shape(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        decisions = copy_of(model_artifact)["decisions"]
        ask = _example_outcome("ASK")
        ask["wait"] = {**_example_outcome("WAIT"), "urgency": "high"}
        decisions[0]["outcome"] = ask
        body = with_decisions(model_artifact, decisions)
        with pytest.raises(ArtifactError, match="unknown field"):
            validate_artifact(body)

    def test_a_well_formed_body_of_every_kind_passes_the_shape_check(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        """The shapes accept what this build writes, kind by kind.

        Without this, every refusal above could be satisfied by a reader that
        refuses everything. The decisions are swapped in one at a time, so what
        is checked is the body and not the run.
        """
        from operatebench.artifact import _validate_outcome_body

        for kind in sorted(_OUTCOME_SHAPES):
            _validate_outcome_body(_example_outcome(kind), f"example[{kind}]")


def _example_outcome(kind: str) -> dict[str, Any]:
    """One well-formed body per kind, in the exact projection this build writes."""
    bodies: dict[str, dict[str, Any]] = {
        RETRIEVE_KIND: {
            "kind": RETRIEVE_KIND,
            "requests": [{"tool": "list_quotes", "arguments": {}}],
        },
        "ACT": {
            "kind": "ACT",
            "action_type": "book_visit",
            "payload": {"cycle_id": "cycle_1"},
            "evidence_refs": ["evidence_1"],
            "rationale": "the state says a visit is due",
        },
        "WAIT": {
            "kind": "WAIT",
            "reason": "waiting for the supplier",
            "wake_on": ["supplier_visit_response"],
            "fallback_after_minutes": 1440,
        },
        "ASK": {
            "kind": "ASK",
            "recipient_actor_id": "customer_1",
            "message_fixture_id": "msg_cannot_proceed",
            "correlation_id": None,
            "wait": {
                "kind": "WAIT",
                "reason": "waiting for the answer",
                "wake_on": ["customer_resolution_reported"],
                "fallback_after_minutes": None,
            },
        },
        "ESCALATE": {
            "kind": "ESCALATE",
            "checkpoint_id": "checkpoint_1",
            "exception_type": "APPROVAL_REQUIRED",
            "evidence_refs": [],
            "deadline_after_minutes": 1440,
            "rationale": "",
        },
        "COMPLETE": {
            "kind": "COMPLETE",
            "reason": "the work is verified",
            "evidence_refs": ["evidence_1"],
        },
        MALFORMED_KIND: {
            "kind": MALFORMED_KIND,
            "code": sorted(CLASSIFICATIONS)[0],
        },
    }
    return bodies[kind]


class TestTheReadableClassificationVocabulary:
    """Everything the tape can *say* must be something the reader can read.

    The defect this pins is a one-word hole. ``decision_record`` records an
    object that is neither an outcome nor a named classification as the base
    code ``MODEL_OUTPUT_UNCLASSIFIED``, and the reader's vocabulary was built
    from :data:`CLASSIFICATIONS` — the six *parsed* classifications — which does
    not contain it. So this build could write a decision tape it then refused to
    read: a model run whose response was unrecognisable produced an artefact
    that failed its own reader, and the failure looked like a tampered record.

    Closing it is one code, not a widened check: the vocabulary stays closed,
    the code carries no provider text, and every code in it round-trips back to
    the classification playback rebuilds.
    """

    def test_the_base_code_is_readable(self) -> None:
        assert UNCLASSIFIED_CODE == "MODEL_OUTPUT_UNCLASSIFIED"
        assert UNCLASSIFIED_CODE in READABLE_CLASSIFICATIONS

    def test_the_vocabulary_is_the_parsed_classifications_plus_the_base_one(
        self,
    ) -> None:
        readable = set(READABLE_CLASSIFICATIONS)
        assert readable == {*CLASSIFICATIONS, UNCLASSIFIED_CODE}
        assert UNCLASSIFIED_CODE not in CLASSIFICATIONS, (
            "the base code names no parsed misbehaviour; it is what an object "
            "that is not a decision at all is recorded as"
        )

    def test_the_reader_holds_a_malformed_decision_to_exactly_that_vocabulary(
        self,
    ) -> None:
        from operatebench.artifact import _CLASSIFICATION_CODES

        assert _CLASSIFICATION_CODES == READABLE_CLASSIFICATIONS

    def test_what_the_writer_records_for_an_unrecognisable_object_reads_back(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        recorded = decision_record(object())
        assert recorded == {"kind": MALFORMED_KIND, "code": UNCLASSIFIED_CODE}

        decisions = copy_of(model_artifact)["decisions"]
        decisions[0]["outcome"] = recorded
        # The attempt beside it says the provider answered with something that
        # was classified rather than decided, which is what this is.
        body = with_decisions(model_artifact, decisions)
        body["agent_execution"]["attempts"][0]["outcome"] = "classified"
        validate_artifact(body)

    def test_every_readable_code_rebuilds_into_a_classification(self) -> None:
        for code in sorted(READABLE_CLASSIFICATIONS):
            rebuilt = outcome_from_record({"kind": MALFORMED_KIND, "code": code})
            assert isinstance(rebuilt, MalformedModelOutcome)
            assert rebuilt.code == code, (
                "a code the reader accepts and playback rebuilds into a different "
                "classification would replay as a different run"
            )

    def test_a_code_outside_the_vocabulary_is_still_refused(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        decisions = copy_of(model_artifact)["decisions"]
        decisions[0]["outcome"] = {
            "kind": MALFORMED_KIND,
            "code": "MODEL_OUTPUT_UNCLASSIFIED_BUT_INTERESTING",
        }
        with pytest.raises(ArtifactError, match="not a model output classification"):
            validate_artifact(with_decisions(model_artifact, decisions))

    def test_no_readable_code_carries_provider_text(self) -> None:
        # The vocabulary is this build's own words. A code that quoted the model
        # would put unreviewed provider output into durable evidence through the
        # one field a malformed decision is allowed to carry.
        for code in READABLE_CLASSIFICATIONS:
            assert code.startswith("MODEL_")
            assert code == code.upper()
            assert set(code) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ_")


class TestATrajectoryOutcomeIsHeldToTheDecisionShapes:
    """``trajectory[].agent_outcome.outcome`` is a decision, and is read as one.

    The row shape said ``object``. The engine writes exactly the projection the
    tape writes — :func:`~operatebench.core.outcomes.outcome_as_dict` — so a row
    checked only for "is a mapping" accepted an ``ACT`` carrying the prose that
    arrived beside a tool call, an ``ASK`` whose nested wait nobody could
    honour, a ``MALFORMED`` naming a code this build never classifies, and a
    kind that does not exist. The evaluator reads those rows, so the reader has
    to hold them to the same shapes it holds the tape to, and it has to do it
    before anything replays.
    """

    def test_the_row_shape_names_the_outcome_kind_not_a_bare_object(self) -> None:
        assert _ROW_SHAPES["agent_outcome"] == (
            {
                "invocation_index": "counter",
                "turn_index": "counter",
                "outcome": "outcome",
            },
        )

    def test_the_shipped_run_writes_rows_that_satisfy_the_decision_shapes(
        self, deterministic: Mapping[str, Any]
    ) -> None:
        # The positive control: the rows this build actually writes pass, so the
        # refusals below are about the shapes and not about a reader that
        # refuses every trajectory it meets.
        rows = [
            row
            for row in deterministic["trajectory"]
            if row["record_type"] == "agent_outcome"
        ]
        assert rows, "the reference run produces agent outcomes"
        assert {row["outcome"]["kind"] for row in rows} <= set(_OUTCOME_SHAPES)
        validate_artifact(deterministic)

    @pytest.mark.parametrize(
        ("outcome", "refusal"),
        [
            (
                {**_example_outcome("ACT"), "provider_text": "here is why I did that"},
                "unknown field",
            ),
            (
                {**_example_outcome("COMPLETE"), "confidence": 0.9},
                "unknown field",
            ),
            ({"kind": "IMPROVISE"}, "not a decision kind"),
            ({"kind": "ACT"}, "missing required field"),
            (
                {**_example_outcome("ACT"), "payload": ["not", "a", "mapping"]},
                "must be a JSON object",
            ),
            (
                {**_example_outcome("ACT"), "evidence_refs": "evidence_1"},
                "must be a JSON array",
            ),
            (
                {**_example_outcome(MALFORMED_KIND), "code": "MODEL_INVENTED"},
                "not a model output classification",
            ),
        ],
    )
    def test_a_row_outcome_this_build_never_wrote_is_refused(
        self, deterministic: Mapping[str, Any], outcome: dict[str, Any], refusal: str
    ) -> None:
        body = with_trajectory_outcome(deterministic, outcome)
        with pytest.raises(ArtifactError, match=refusal):
            validate_artifact(body)

    def test_a_nested_ask_wait_in_a_row_is_held_to_the_wait_shape(
        self, deterministic: Mapping[str, Any]
    ) -> None:
        ask = copy_of(_example_outcome("ASK"))
        ask["wait"] = {**_example_outcome("WAIT"), "urgency": "high"}
        with pytest.raises(ArtifactError, match="unknown field"):
            validate_artifact(with_trajectory_outcome(deterministic, ask))

        ask = copy_of(_example_outcome("ASK"))
        ask["wait"] = _example_outcome("ACT")
        with pytest.raises(ArtifactError, match="must be 'WAIT'"):
            validate_artifact(with_trajectory_outcome(deterministic, ask))

    def test_the_refusal_names_the_row_it_came_from(
        self, deterministic: Mapping[str, Any]
    ) -> None:
        index = _first_agent_outcome(deterministic)
        body = with_trajectory_outcome(
            deterministic, {**_example_outcome("ACT"), "provider_text": "prose"}
        )
        with pytest.raises(ArtifactError) as raised:
            validate_artifact(body)
        assert f"trajectory[{index}].outcome" in str(raised.value)

    def test_the_row_is_refused_before_the_trajectory_digest_is_recomputed(
        self, spec: OperationSpec, deterministic: Mapping[str, Any]
    ) -> None:
        """Shape first, and before replay — not "the digest caught it anyway".

        The tampered row here is left with the digest recomputed over it, so
        nothing else can be what refuses it. A reader that admitted the row and
        left it to the digest would admit exactly this document.
        """
        body = with_trajectory_outcome(
            deterministic, {**_example_outcome("ACT"), "rationale": 7}
        )
        body["trajectory_digest_sha256"] = _content_digest(body["trajectory"])
        with pytest.raises(ArtifactError, match="must be a string"):
            validate_artifact(body)
        with pytest.raises(ArtifactError, match="must be a string"):
            replay_artifact(spec, body)


def _first_agent_outcome(payload: Mapping[str, Any]) -> int:
    for index, row in enumerate(payload["trajectory"]):
        if row["record_type"] == "agent_outcome":
            return index
    raise AssertionError("the reference run records an agent outcome")


def with_trajectory_outcome(
    payload: Mapping[str, Any], outcome: Mapping[str, Any]
) -> dict[str, Any]:
    """A copy whose first recorded agent outcome carries ``outcome``."""
    body = copy_of(payload)
    body["trajectory"][_first_agent_outcome(body)]["outcome"] = copy_of(outcome)
    return body


# ------------------------------------------------------------------- the CLI


class TestThePublicCommandRefusesByName:
    def _written(self, path: Path, payload: Mapping[str, Any]) -> Path:
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_a_relabelled_artefact_exits_one_with_a_named_message(
        self,
        spec: OperationSpec,
        deterministic: Mapping[str, Any],
        model_artifact: Mapping[str, Any],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        body = copy_of(deterministic)
        body["agent_kind"] = KIND_MODEL
        body["agent_execution"] = copy_of(model_artifact)["agent_execution"]
        path = self._written(tmp_path / "relabelled.json", body)
        assert main(["replay", "--spec", str(FIXTURE), "--run", str(path)]) == EXIT_ERROR
        captured = capsys.readouterr()
        assert "relabels one run as another" in captured.err
        assert "Traceback" not in captured.err

    def test_an_excluded_artefact_exits_one_with_a_named_message(
        self,
        spec: OperationSpec,
        model_artifact: Mapping[str, Any],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        body = copy_of(model_artifact)
        body["agent_execution"].update(
            {
                "excluded": True,
                "exclusion_code": "provider_transport",
                "exclusion_detail": "socket closed",
            }
        )
        path = self._written(tmp_path / "excluded.json", body)
        assert main(["replay", "--spec", str(FIXTURE), "--run", str(path)]) == EXIT_ERROR
        captured = capsys.readouterr()
        assert "no episode artefact in this build" in captured.err
        assert "Traceback" not in captured.err

    def test_an_unknown_agent_exits_one_with_a_named_message(
        self,
        spec: OperationSpec,
        deterministic: Mapping[str, Any],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        body = copy_of(deterministic)
        body["agent_id"] = "no_such_agent_anywhere"
        path = self._written(tmp_path / "unknown.json", body)
        assert main(["replay", "--spec", str(FIXTURE), "--run", str(path)]) == EXIT_ERROR
        captured = capsys.readouterr()
        assert "not an agent this build registers" in captured.err
        assert "Traceback" not in captured.err

    def test_a_forged_model_identity_exits_one_with_a_named_message(
        self,
        spec: OperationSpec,
        model_artifact: Mapping[str, Any],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        body = copy_of(model_artifact)
        body["agent_execution"]["model"] = "a-different-model"
        body["provider_execution"]["model"] = "a-different-model"
        path = self._written(tmp_path / "forged_model.json", body)
        assert main(["replay", "--spec", str(FIXTURE), "--run", str(path)]) == EXIT_ERROR
        captured = capsys.readouterr()
        assert "not a label beside them" in captured.err
        assert "Traceback" not in captured.err

    def test_a_playback_replay_says_at_the_terminal_what_it_carried(
        self,
        model_artifact: Mapping[str, Any],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path = self._written(tmp_path / "model.json", model_artifact)
        assert main(["replay", "--spec", str(FIXTURE), "--run", str(path)]) == 0
        out = capsys.readouterr().out
        assert "replay OK" in out
        assert "carried forward" in out
        assert "not total" in out, (
            "an operator reading 'replay OK' has to be able to see that some "
            "fields were taken from the record rather than re-derived"
        )

    def test_a_deterministic_replay_claims_no_carried_fields(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        output = tmp_path / "reference.json"
        main(
            [
                "run-maintenance",
                "--spec",
                str(FIXTURE),
                "--scenario",
                "V1",
                "--agent",
                "reference",
                "--output",
                str(output),
            ]
        )
        capsys.readouterr()
        assert main(["replay", "--spec", str(FIXTURE), "--run", str(output)]) == 0
        out = capsys.readouterr().out
        assert "carried forward" not in out
        assert "reproduced by      rerun" in out

    def test_a_genuine_run_still_writes_and_replays_through_the_cli(
        self, tmp_path: Path
    ) -> None:
        output = tmp_path / "run.json"
        assert (
            main(
                [
                    "run-maintenance",
                    "--spec",
                    str(FIXTURE),
                    "--scenario",
                    "V1",
                    "--agent",
                    "reference",
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
        assert main(["replay", "--spec", str(FIXTURE), "--run", str(output)]) == 0


# ------------------------------------------------------- cross-version notes


class TestV1CompatibilityIsStated:
    def test_the_frozen_v1_artefact_still_validates_here(self) -> None:
        body = read_artifact(FROZEN_V1)
        assert body["artifact_version"] == 1

    def test_replaying_it_here_is_a_named_incompatibility(
        self, spec: OperationSpec
    ) -> None:
        """Stated by digest, because that is what makes it checkable.

        The frozen record describes a run of the operation as it was authored
        before this build. Its spec digest and the shipped fixture's are
        different values, so the refusal names both and says what comparing
        them would mean. Nothing about it is narrowed, guessed or downgraded to
        a difference report.
        """
        recorded = read_artifact(FROZEN_V1)
        frozen_digest = recorded["operation"]["spec_digest_sha256"]
        assert frozen_digest != spec.spec_digest_sha256

        with pytest.raises(ArtifactError) as raised:
            replay_artifact(spec, recorded)

        message = str(raised.value)
        assert "artefact contract 1" in message
        assert "reproduces contract(s) [8]" in message
        assert "compare two different contracts" in message

    def test_a_current_v1_shape_cannot_invent_the_retracted_wait_claim(
        self, spec: OperationSpec, deterministic: Mapping[str, Any]
    ) -> None:
        with pytest.raises(ArtifactError) as raised:
            replay_artifact(spec, as_v1_shape(deterministic))
        message = str(raised.value)
        # Refused before it reaches the wait row, and refused for the same
        # reason: a contract-1 runtime performed no reads, so a document that
        # carries one describes a run its own contract had no way to produce.
        assert "agent_invoked" in message or "derived_context_invalidated" in message
        assert "contract 1" in message

    def test_a_v1_artefact_carries_no_execution_record_to_bind(self) -> None:
        body = read_artifact(FROZEN_V1)
        assert "agent_execution" not in body
        assert body["agent_kind"] in DETERMINISTIC_AGENT_KINDS


class TestFieldsALaterContractBuildsOn:
    def test_every_delivered_event_must_carry_a_non_empty_verdict_code(
        self, deterministic: Mapping[str, Any]
    ) -> None:
        assert deterministic["events"], "the fixture delivers events"
        for event in deterministic["events"]:
            assert isinstance(event["verdict_code"], str) and event["verdict_code"]
        for empty in ("", None, 0):
            body = copy_of(deterministic)
            body["events"][0]["verdict_code"] = empty
            with pytest.raises(ArtifactError, match="verdict_code"):
                validate_artifact(body)

    def test_the_evaluation_reader_refuses_a_further_dimension(
        self, deterministic: Mapping[str, Any]
    ) -> None:
        """The dimension vector is read by name and by order, per contract.

        This test asserted the opposite until the vector became a contract, on
        the reasoning that a later evaluator adding a dimension should not need
        the reader changed. That is true of a *widening*, and the result vector
        does not widen: it is versioned, and a dimension arrives with a contract
        bump — ``environment_integrity`` at 2, ``retrieval_discipline`` at 6. A
        reader that accepted an extra entry at any version was accepting a
        document claiming a dimension its own contract never graded, and leaving
        the divergence to be discovered at replay, which reads records that
        never replay.

        The added entry is a well-formed dimension. It is refused for what it
        claims, not for how it is shaped.
        """
        body = copy_of(deterministic)
        body["evaluation"]["dimensions"].append(
            {
                "name": "environment_integrity",
                "ok": True,
                "findings": [],
                "counts": {"checked": 1},
                "note": "",
            }
        )
        with pytest.raises(ArtifactError, match=r"evaluation\.dimensions"):
            validate_artifact(body)

    def test_a_malformed_further_dimension_is_still_refused(
        self, deterministic: Mapping[str, Any]
    ) -> None:
        body = copy_of(deterministic)
        body["evaluation"]["dimensions"].append(
            {"name": "environment_integrity", "ok": "probably"}
        )
        with pytest.raises(ArtifactError):
            validate_artifact(body)


# --------------------------------------------------- the boundary of the claim


class TestConsistencyIsNotAttestation:
    """What the binding above establishes, and the thing it cannot.

    Every test before this one is an inconsistent record meeting a named
    refusal. This one is the other half of the same statement, asserted rather
    than left to be discovered by a reader who assumes more than is true: a
    party who can run this build can compute the same request digests from the
    same observations and mint a record that is consistent throughout, and this
    reader accepts it. Nothing in a self-describing local file distinguishes a
    provider that answered from a provider that was never asked; closing that
    needs evidence the provider signs, which this format does not hold.

    So the test passes on acceptance *on purpose*. If a later change makes this
    record refused, the claim has grown and the docstring in
    :mod:`operatebench.artifact` that says it has not needs to change with it.
    """

    def test_a_record_minted_from_a_deterministic_run_is_consistent_and_accepted(
        self, spec: OperationSpec, deterministic: Mapping[str, Any]
    ) -> None:
        digests: dict[int, str] = {}
        rebuilder = ModelAgent(
            ForbiddenTransport(),
            model=TEST_MODEL,
            agent_id="minted",
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        class Sniffing(RecordedOutcomeAgent):
            """Playback that also records what request the forger would send.

            Keyed by tape position rather than appended, because an episode is
            executed twice and the second pass must not double the list.
            """

            def decide(self, observation: Any) -> Any:
                digests[self.cursor] = rebuilder.build_request(
                    observation
                ).request_digest_sha256
                return super().decide(observation)

        body = copy_of(deterministic)
        tape = tape_from_records(body["decisions"])
        # Reproduction of a persisted record, under the identity it carries, so
        # the private surface rather than the public one.
        from operatebench.runner import _reproduce_episode

        _reproduce_episode(
            spec,
            str(body["scenario_id"]),
            "minted",
            operation_instance_id=str(body["operation_instance_id"]),
            agent_factory=lambda: Sniffing(tape, agent_id="minted"),
            agent_kind=KIND_MODEL,
        )
        assert len(digests) == len(body["decisions"])

        body["agent_id"] = "minted"
        body["agent_kind"] = KIND_MODEL
        body["agent_execution"] = {
            "outcome_source": SOURCE_MODEL,
            "model": TEST_MODEL,
            "protocol_version": REPRODUCIBLE_MODEL_PROTOCOLS[0],
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "transport_calls": len(body["decisions"]),
            "retry_count": RETRY_COUNT,
            "attempts": [
                {
                    "invocation_index": decision["invocation_index"],
                    "turn_index": decision["turn_index"],
                    "request_digest_sha256": digests[position],
                    "outcome": (
                        "classified"
                        if decision["outcome"]["kind"] == MALFORMED_KIND
                        else "decided"
                    ),
                    "fault": None,
                }
                for position, decision in enumerate(body["decisions"])
            ],
            "excluded": False,
            "exclusion_code": None,
            "exclusion_detail": "",
        }
        # Contract 7 makes the forger mint one thing more: the binding to the
        # journal that witnessed the session. It costs nothing to mint a
        # *coherent* one — every value here is derivable from the record being
        # forged — and that is exactly the point being made. A binding is a
        # reference, not evidence: what it cannot do without also fabricating a
        # whole hash-chained journal is survive
        # :func:`~operatebench.execution_bundle.audit_execution_bundle`, which is
        # why that audit is a separate surface reading a separate file. See
        # :mod:`tests.test_execution_bundle_audit`.
        body["provider_execution"] = {
            "execution_run_id": "exec_" + "0" * 32,
            "execution_ledger_version": EXECUTION_LEDGER_VERSION,
            "execution_ledger_digest_sha256": "0" * 64,
            "provider": "openai",
            "api": "responses",
            "model": TEST_MODEL,
            "settings_digest_sha256": "1" * 64,
            "pricing_digest_sha256": "2" * 64,
            "provider_calls": len(body["decisions"]),
            "attempts": len(body["decisions"]),
            "measured_cost_usd": "0",
            "forfeited_reservation_usd": "0",
            "input_tokens": 0,
            "output_tokens": 0,
            "decision_call_index": list(range(len(body["decisions"]))),
        }

        validate_artifact(body)
        report = replay_artifact(spec, body)
        assert report.ok
        # And the report does not pretend otherwise: the fields this replay took
        # from the record are named, and it says the comparison was not total.
        assert report.total_comparison is False
        assert "agent_execution" in report.carried_forward
        assert "agent_id" in report.carried_forward
