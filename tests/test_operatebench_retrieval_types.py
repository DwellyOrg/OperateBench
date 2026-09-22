"""Phase 1A: the retrieval batch contract, its canonical form and its refusals.

The object under test is deliberately *not* an ``AgentOutcome``. A retrieval is
not a decision the operation records as business; it is a read the environment
performs on the agent's behalf, and the whole point of giving it its own type is
that :func:`~operatebench.core.outcomes.outcome_contract_problem` keeps refusing
anything it does not recognise.

What is asserted here is the part a later phase grades on: a request set is
byte-identical however it was typed, a duplicate collapses, a conflicting
duplicate is refused rather than silently resolved, and every way a batch can be
wrong has a name.
"""

from __future__ import annotations

import pytest

from boundarybench.jsonsafe import canonical_json_bytes
from operatebench.core.outcomes import Wait, outcome_contract_problem
from operatebench.core.retrieval import (
    INVALIDATION_CAUSES,
    MAX_REQUESTS_PER_BATCH,
    MIN_REQUESTS_PER_BATCH,
    RETRIEVAL_AUTHORITIES,
    RETRIEVAL_BATCH_TOO_LARGE,
    RETRIEVAL_BUDGET_EXHAUSTED,
    RETRIEVAL_CONFLICTING_ARGUMENTS,
    RETRIEVAL_EMPTY_BATCH,
    RETRIEVAL_INITIATORS,
    RETRIEVAL_MALFORMED_REQUEST,
    RETRIEVAL_REFUSAL_CODES,
    RETRIEVAL_UNKNOWN_TOOL,
    RETRIEVE_KIND,
    RetrievalRequest,
    RetrieveBatch,
    ToolResult,
    canonical_requests,
    retrieval_batch_problem,
    tool_result_from_dict,
)
from operatebench.domains.lettings.maintenance.retrieval import (
    MAINTENANCE_RETRIEVAL_TOOL_NAMES,
    maintenance_retrieval_catalogue,
)

CATALOGUE = maintenance_retrieval_catalogue()


def batch(*tools: str) -> RetrieveBatch:
    return RetrieveBatch(tuple(RetrievalRequest(tool=tool) for tool in tools))


def problem(candidate: RetrieveBatch, *, remaining: int = 6) -> tuple[str, str] | None:
    return retrieval_batch_problem(
        candidate, catalogue=CATALOGUE, batch_budget_remaining=remaining
    )


# -- the type ---------------------------------------------------------------


class TestTheBatchIsNotAnOutcome:
    def test_a_batch_declares_the_retrieve_kind(self) -> None:
        assert RetrieveBatch(()).kind == RETRIEVE_KIND == "RETRIEVE"
        assert RetrievalRequest(tool="list_quotes").arguments == {}

    def test_the_outcome_contract_still_refuses_a_batch(self) -> None:
        # The union is unchanged, so the engine's public boundary keeps naming
        # anything that is not one of the five.
        assert outcome_contract_problem(batch("list_quotes")) is not None
        assert outcome_contract_problem(Wait(reason="", fallback_after_minutes=1)) is None

    def test_a_request_and_a_result_project_to_plain_builtins(self) -> None:
        request = RetrievalRequest(tool="list_quotes")
        assert request.as_dict() == {"tool": "list_quotes", "arguments": {}}
        result = ToolResult(
            tool="list_quotes",
            source="quoting_service",
            authority="system_of_record",
            record_id="quoting_service:maintenance_case",
            record_version="0" * 16,
            as_of="2025-01-01T00:00:00Z",
            schema_id="maintenance.list_quotes.v1",
            records={"quotes": {}},
        )
        assert tool_result_from_dict(result.as_dict()) == result
        assert result.ok is True
        assert result.error_code is None


# -- canonical form ---------------------------------------------------------


class TestCanonicalForm:
    def test_ordering_is_permutation_invariant(self) -> None:
        first = canonical_requests(
            batch("list_quotes", "get_case_record", "list_billing")
        )
        second = canonical_requests(
            batch("list_billing", "list_quotes", "get_case_record")
        )
        assert first == second
        assert [request.tool for request in first] == sorted(
            ["list_quotes", "get_case_record", "list_billing"]
        )

    def test_identical_duplicates_collapse(self) -> None:
        collapsed = canonical_requests(batch(*["list_quotes"] * 12, "get_case_record"))
        assert [request.tool for request in collapsed] == [
            "get_case_record",
            "list_quotes",
        ]

    def test_canonicalisation_is_idempotent_and_byte_stable(self) -> None:
        once = canonical_requests(batch("list_quotes", "get_case_record"))
        twice = canonical_requests(RetrieveBatch(once))
        assert once == twice
        rows = [
            canonical_json_bytes(
                [request.as_dict() for request in canonical_requests(candidate)],
                "retrieval batch",
            )
            for candidate in (
                batch("list_quotes", "get_case_record", "list_billing"),
                batch("list_billing", "get_case_record", "list_quotes"),
                batch("list_quotes", "list_billing", "list_quotes", "get_case_record"),
            )
        ]
        assert len(set(rows)) == 1

    def test_the_whole_vocabulary_is_one_legal_batch(self) -> None:
        assert len(MAINTENANCE_RETRIEVAL_TOOL_NAMES) == MAX_REQUESTS_PER_BATCH == 8
        assert problem(batch(*MAINTENANCE_RETRIEVAL_TOOL_NAMES)) is None


# -- refusals ---------------------------------------------------------------


class TestEveryWayABatchCanBeWrongHasAName:
    def test_the_refusal_vocabulary_is_closed(self) -> None:
        assert (
            tuple(
                sorted(
                    {
                        RETRIEVAL_BATCH_TOO_LARGE,
                        RETRIEVAL_BUDGET_EXHAUSTED,
                        RETRIEVAL_CONFLICTING_ARGUMENTS,
                        RETRIEVAL_EMPTY_BATCH,
                        RETRIEVAL_MALFORMED_REQUEST,
                        RETRIEVAL_UNKNOWN_TOOL,
                    }
                )
            )
            == RETRIEVAL_REFUSAL_CODES
        )
        assert RETRIEVAL_INITIATORS == ("agent",)
        assert INVALIDATION_CAUSES == ("effect_accepted", "wake")
        assert tuple(sorted(RETRIEVAL_AUTHORITIES)) == RETRIEVAL_AUTHORITIES

    def test_an_empty_batch_is_refused_by_name(self) -> None:
        assert MIN_REQUESTS_PER_BATCH == 1
        code, detail = problem(RetrieveBatch(()))
        assert code == RETRIEVAL_EMPTY_BATCH
        assert detail

    def test_more_than_eight_distinct_requests_is_refused_by_name(self) -> None:
        # Size is checked before membership: nine requests are refused for being
        # nine, whatever they name, so an oversize batch cannot be re-labelled as
        # an unknown-tool refusal by including one.
        oversize = RetrieveBatch(
            (
                *(
                    RetrievalRequest(tool=name)
                    for name in MAINTENANCE_RETRIEVAL_TOOL_NAMES
                ),
                RetrievalRequest(tool="ninth_service"),
            )
        )
        code, detail = problem(oversize)
        assert code == RETRIEVAL_BATCH_TOO_LARGE
        assert "9" in detail

    def test_size_is_decided_before_membership_semantics(self) -> None:
        """A compound-invalid batch answers with the size refusal, as documented.

        Nine distinct requests where one pair is a conflicting duplicate is
        wrong in two ways at once, and the published order of the checks says
        which answer the agent gets: structure, then size, then membership. A
        batch refused for its conflict would tell an agent to drop one argument
        set and retry a batch that is still oversize — which is a refusal that
        teaches the wrong repair.
        """
        compound = RetrieveBatch(
            (
                *(
                    RetrievalRequest(tool=name)
                    for name in MAINTENANCE_RETRIEVAL_TOOL_NAMES
                ),
                RetrievalRequest(tool="list_quotes", arguments={"cycle_id": "x"}),
            )
        )
        code, detail = problem(compound)
        assert code == RETRIEVAL_BATCH_TOO_LARGE
        assert "9" in detail

    def test_size_is_decided_before_the_unknown_and_conflicting_pair(self) -> None:
        # The same compound batch with the ninth request naming a tool outside
        # the catalogue *and* conflicting with itself. Size still answers first.
        compound = RetrieveBatch(
            (
                *(
                    RetrievalRequest(tool=name)
                    for name in MAINTENANCE_RETRIEVAL_TOOL_NAMES
                ),
                RetrievalRequest(tool="list_quotes", arguments={"cycle_id": "x"}),
                RetrievalRequest(tool="ninth_service"),
            )
        )
        code, _ = problem(compound)
        assert code == RETRIEVAL_BATCH_TOO_LARGE

    def test_malformed_and_empty_still_precede_size(self) -> None:
        # The documented order does not change for the two checks that come
        # before size: a request that is not a request cannot be counted, and an
        # empty batch is empty however the bound is set.
        oversize_and_malformed = RetrieveBatch(
            (
                *(
                    RetrievalRequest(tool=name)
                    for name in MAINTENANCE_RETRIEVAL_TOOL_NAMES
                ),
                RetrievalRequest(tool=""),
            )
        )
        code, _ = problem(oversize_and_malformed)
        assert code == RETRIEVAL_MALFORMED_REQUEST
        assert problem(RetrieveBatch(()))[0] == RETRIEVAL_EMPTY_BATCH

    def test_budget_is_still_decided_last(self) -> None:
        # An oversize batch on a spent budget is refused for its size: charging
        # a budget for a batch that was never legal would be a charge for
        # nothing, and the budget refusal is the one an agent can act on only
        # once the batch it sent was one the environment would have served.
        oversize = RetrieveBatch(
            (
                *(
                    RetrievalRequest(tool=name)
                    for name in MAINTENANCE_RETRIEVAL_TOOL_NAMES
                ),
                RetrievalRequest(tool="ninth_service"),
            )
        )
        assert problem(oversize, remaining=0)[0] == RETRIEVAL_BATCH_TOO_LARGE
        assert problem(batch("list_quotes"), remaining=0)[0] == (
            RETRIEVAL_BUDGET_EXHAUSTED
        )

    def test_an_unknown_tool_is_refused_and_never_echoed(self) -> None:
        code, detail = problem(batch("select * from quotes"))
        assert code == RETRIEVAL_UNKNOWN_TOOL
        assert "select * from quotes" not in detail

    def test_a_conflicting_duplicate_is_refused_rather_than_resolved(self) -> None:
        conflicting = RetrieveBatch(
            (
                RetrievalRequest(tool="list_quotes"),
                RetrievalRequest(tool="list_quotes", arguments={"cycle_id": "x"}),
            )
        )
        code, _ = problem(conflicting)
        assert code == RETRIEVAL_CONFLICTING_ARGUMENTS

    @pytest.mark.parametrize(
        "requests",
        [
            "list_quotes",
            (RetrievalRequest(tool=""),),
            ({"tool": "list_quotes"},),
            (RetrievalRequest(tool="list_quotes", arguments={"": 1}),),
        ],
    )
    def test_a_malformed_request_is_a_named_refusal_not_an_exception(
        self, requests: object
    ) -> None:
        code, _ = problem(RetrieveBatch(requests))  # type: ignore[arg-type]
        assert code == RETRIEVAL_MALFORMED_REQUEST

    def test_an_exhausted_budget_is_a_named_refusal(self) -> None:
        code, detail = problem(batch("list_quotes"), remaining=0)
        assert code == RETRIEVAL_BUDGET_EXHAUSTED
        assert detail

    def test_arguments_outside_the_published_schema_are_refused(self) -> None:
        # This vertical publishes an empty argument schema for every tool: the
        # eight reads are whole-collection projections and take no parameter. An
        # invented key is interface guessing, so it is named rather than ignored.
        assert all(entry["arguments"] == {} for entry in CATALOGUE.values())
        code, _ = problem(
            RetrieveBatch((RetrievalRequest(tool="list_quotes", arguments={"page": 1}),))
        )
        assert code == RETRIEVAL_MALFORMED_REQUEST
