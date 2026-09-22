"""What a ledger field may hold, asked one primitive at a time.

The chain proves a row was not edited and the state machine proves the rows
describe a run that could have happened. Neither of them says anything about
whether a *value* is the kind of value this build writes, and that is where a
durable record quietly stops being comparable: a duration that is a string, a
count that is ``True``, an amount rendered two ways, an instant nothing can
parse, a settings mapping holding an object no JSON reader can carry.

Each of those is refused here by name, and the refusals are asserted as *types*
rather than as prose: :class:`LedgerSchemaError` for a value outside the
schema's domain, :class:`LedgerChainError` for a stated digest that does not
cover what it sits beside, :class:`LedgerSecretError` for material that must
never reach durable evidence at all.

Every constructor in this module is the real record type, and the fixtures are
the shared ones, so a field that stopped being validated fails here rather than
being discovered by a reader long after the run it describes.
"""

from __future__ import annotations

from typing import Any

import pytest

from operatebench.execution_ledger import (
    ABORTED_CODE,
    EXECUTION_RUN_ID_PREFIX,
    TERMINAL_ABORTED,
    TERMINAL_EXCLUDED,
    TERMINAL_SCORED,
    LedgerAttempt,
    LedgerCall,
    LedgerChainError,
    LedgerControls,
    LedgerHeader,
    LedgerProviderIdentity,
    LedgerSchemaError,
    LedgerSecretError,
    LedgerTerminal,
    assert_no_secret_material,
    check_call_amounts,
    check_execution_run_id,
    check_terminal_admissible,
    ledger_digest,
    new_execution_run_id,
    recompute_totals,
)
from operatebench.providers.faults import (
    ATTEMPT_OUTCOME_FAULT,
    ATTEMPT_OUTCOME_RESPONSE,
    ATTEMPT_OUTCOME_UNCLASSIFIED,
    ATTEMPT_SETTLEMENT_FORFEITED,
    ATTEMPT_SETTLEMENT_MEASURED,
    TURN_END_RESPONSE,
    TURN_END_RETRIES_EXHAUSTED,
    TURN_END_UNCLASSIFIED,
)
from tests.execution_ledger_fixtures import (
    DIGEST_A,
    DIGEST_B,
    DIGEST_C,
    DIGEST_D,
    INSTANT,
    LATER,
    MEASURED_USD,
    RESERVATION_USD,
    answered_attempt,
    call,
    controls,
    decision,
    faulted_attempt,
    header,
    pricing,
    provider_identity,
    settings,
)


def identity(**overrides: Any) -> LedgerProviderIdentity:
    """The shared provider identity, with one field replaced."""
    fields = {
        name: value
        for name, value in provider_identity().as_dict().items()
        if name != "settings_digest_sha256"
    }
    fields.update(overrides)
    return LedgerProviderIdentity(**fields)


def unclassified_attempt(**overrides: Any) -> LedgerAttempt:
    """One attempt that met something this build never classified at all."""
    fields: dict[str, Any] = {
        "attempt_index": 1,
        "outcome": ATTEMPT_OUTCOME_UNCLASSIFIED,
        "fault": None,
        "http_status": None,
        "latency_seconds": 0.5,
        "response_received": False,
        "usage_reported": False,
        "cost_reservation_usd": RESERVATION_USD,
        "cost_settlement": ATTEMPT_SETTLEMENT_FORFEITED,
        "cost_measured_usd": None,
        "cost_forfeited_usd": RESERVATION_USD,
        "input_tokens": None,
        "output_tokens": None,
        "response_model": None,
        "response_id_digest_sha256": None,
        "response_stop_classification": None,
        "response_normalized_digest_sha256": None,
    }
    fields.update(overrides)
    return LedgerAttempt(**fields)


# -- the digest this module takes ----------------------------------------------


def test_a_payload_that_cannot_be_canonically_encoded_is_this_modules_refusal() -> None:
    """The encoder's error is translated, so callers meet one refusal vocabulary."""
    with pytest.raises(LedgerSchemaError):
        ledger_digest({"settings": {1: "a non-string key"}})


def test_the_digest_is_over_content_and_not_over_authoring_order() -> None:
    assert ledger_digest({"a": 1, "b": 2}) == ledger_digest({"b": 2, "a": 1})


# -- an execution run identity -------------------------------------------------


def test_a_minted_run_identity_is_labelled_so_it_cannot_be_read_as_a_digest() -> None:
    minted = new_execution_run_id()
    assert minted.startswith(EXECUTION_RUN_ID_PREFIX)
    assert check_execution_run_id(minted, "header") == minted
    assert new_execution_run_id() != minted


@pytest.mark.parametrize("value", ["a" * 32, 17, None, "run_" + "a" * 32, ""])
def test_an_identity_without_this_builds_label_is_refused(value: Any) -> None:
    with pytest.raises(LedgerSchemaError):
        check_execution_run_id(value, "header.execution_run_id")


@pytest.mark.parametrize(
    "body", ["a" * 31, "a" * 33, "A" * 32, "g" * 32, "0123456789abcdef" * 3]
)
def test_an_identity_whose_body_is_not_lowercase_hex_of_the_right_width_is_refused(
    body: str,
) -> None:
    with pytest.raises(LedgerSchemaError):
        check_execution_run_id(f"{EXECUTION_RUN_ID_PREFIX}{body}", "header")


# -- the scan that runs beside the schema --------------------------------------


def test_a_payload_nested_past_the_scans_floor_is_refused_rather_than_half_scanned() -> (
    None
):
    """A partial scan is the failure mode: material below the floor is unseen."""
    deep: Any = "authorization: Bearer sk-deadbeefdeadbeef"
    for _ in range(40):
        deep = [deep]
    with pytest.raises(LedgerSchemaError):
        assert_no_secret_material(deep, "row 0")


def test_a_field_name_that_is_not_text_is_refused_before_it_is_scanned() -> None:
    with pytest.raises(LedgerSchemaError):
        assert_no_secret_material({"provider": {7: "openai"}}, "row 0")


def test_the_scan_reaches_material_buried_under_a_field_nobody_validated() -> None:
    """Two nets: a field added without a validator still meets this one."""
    with pytest.raises(LedgerSecretError):
        assert_no_secret_material(
            {"provider": {"settings": {"extras": ["x-api-key: 1"]}}}, "row 0"
        )


# -- counts, flags and durations -----------------------------------------------


@pytest.mark.parametrize("index", [True, 0, -1, "1", 1.0, None])
def test_an_attempt_index_that_is_not_a_count_from_one_is_refused(index: Any) -> None:
    """``bool`` is an ``int`` subclass, so the type test is exact rather than kind."""
    with pytest.raises(LedgerSchemaError):
        answered_attempt(attempt_index=index)


@pytest.mark.parametrize("status", [True, 99, -1, "500", 500.0])
def test_an_http_status_below_the_lowest_real_one_is_refused(status: Any) -> None:
    with pytest.raises(LedgerSchemaError):
        faulted_attempt(http_status=status)


@pytest.mark.parametrize("flag", [1, 0, "true", None, "yes"])
def test_a_boolean_field_that_is_not_a_boolean_is_refused(flag: Any) -> None:
    with pytest.raises(LedgerSchemaError):
        answered_attempt(response_received=flag)


def test_an_unmeasured_latency_is_null_and_that_is_not_zero() -> None:
    """``None`` means the implementation did not measure it; zero is a claim."""
    unmeasured = faulted_attempt(latency_seconds=None)
    assert unmeasured.latency_seconds is None
    assert unmeasured.as_dict()["latency_seconds"] is None


@pytest.mark.parametrize("seconds", [True, "0.5", -0.001, -1])
def test_a_duration_that_is_not_a_non_negative_number_is_refused(seconds: Any) -> None:
    with pytest.raises(LedgerSchemaError):
        answered_attempt(latency_seconds=seconds)


# -- amounts -------------------------------------------------------------------


@pytest.mark.parametrize("amount", [0.0593425, 5, b"1", ["0.0593425"]])
def test_a_usd_amount_that_is_not_a_decimal_string_is_refused(amount: Any) -> None:
    """A JSON number is a binary float to most readers, and is not the amount."""
    with pytest.raises(LedgerSchemaError):
        answered_attempt(cost_reservation_usd=amount)


@pytest.mark.parametrize("amount", ["", "one", "0x10", "1,5"])
def test_a_string_that_is_not_a_decimal_amount_is_refused(amount: str) -> None:
    with pytest.raises(LedgerSchemaError):
        answered_attempt(cost_reservation_usd=amount)


@pytest.mark.parametrize("amount", ["-0.01", "NaN", "Infinity"])
def test_a_negative_or_non_finite_amount_is_refused(amount: str) -> None:
    with pytest.raises(LedgerSchemaError):
        answered_attempt(cost_reservation_usd=amount)


@pytest.mark.parametrize("amount", ["0.05934250", "0.0593425000", "+0.0593425"])
def test_an_amount_rendered_a_second_way_is_refused_rather_than_normalised(
    amount: str,
) -> None:
    """One rendering, so two reports of one amount compare equal as strings."""
    with pytest.raises(LedgerSchemaError):
        answered_attempt(cost_reservation_usd=amount)


@pytest.mark.parametrize("cap", [1.0, 100, None, ["1.00"]])
def test_a_cost_cap_that_is_not_a_decimal_string_is_refused(cap: Any) -> None:
    with pytest.raises(LedgerSchemaError):
        controls(cost_cap_usd=cap)


@pytest.mark.parametrize("cap", ["one dollar", "", "£1"])
def test_a_cost_cap_that_is_not_a_decimal_is_refused(cap: str) -> None:
    with pytest.raises(LedgerSchemaError):
        controls(cost_cap_usd=cap)


@pytest.mark.parametrize("cap", ["0", "0.00", "-1", "NaN"])
def test_a_cap_of_zero_or_less_is_refused_rather_than_read_as_no_limit(cap: str) -> None:
    with pytest.raises(LedgerSchemaError):
        controls(cost_cap_usd=cap)


@pytest.mark.parametrize("typed", ["1", "1.00", "1.000000"])
def test_a_cap_an_operator_typed_is_stored_in_one_rendering(typed: str) -> None:
    """``1.00`` and ``1`` are one authorisation written two ways, not two caps."""
    assert controls(cost_cap_usd=typed).cost_cap_usd == "1"


# -- instants ------------------------------------------------------------------


@pytest.mark.parametrize(
    "instant",
    ["2020-13-45T00:00:00Z", "not a time", "", 1_900_000_000, "2030-01-01 00:00:00"],
)
def test_an_instant_this_build_cannot_parse_is_refused(instant: Any) -> None:
    with pytest.raises(LedgerSchemaError):
        header(started_at=instant)


# -- mappings ------------------------------------------------------------------


@pytest.mark.parametrize("table", [["1.25"], "1.25", 5, None])
def test_a_rate_table_that_is_not_a_json_object_is_refused(table: Any) -> None:
    with pytest.raises(LedgerSchemaError):
        controls(pricing=table)


def test_a_settings_value_json_cannot_carry_is_refused_rather_than_coerced() -> None:
    """A set is not a JSON value; storing one would be storing a guess about it."""
    with pytest.raises(LedgerSchemaError):
        identity(settings={**settings(), "tools": {"act", "wait"}})


def test_a_settings_mapping_is_carried_whole_and_digested_from_what_it_holds() -> None:
    stated = provider_identity()
    assert stated.settings_digest_sha256 == ledger_digest(dict(stated.settings))
    assert stated.as_dict()["settings"] == dict(settings())


# -- an attempt states one coherent thing --------------------------------------


def test_an_attempt_names_a_fault_exactly_when_its_outcome_is_one() -> None:
    with pytest.raises(LedgerSchemaError):
        answered_attempt(fault="provider_timeout")
    with pytest.raises(LedgerSchemaError):
        faulted_attempt(fault=None)


def test_an_attempt_whose_outcome_is_a_response_received_one() -> None:
    with pytest.raises(LedgerSchemaError):
        answered_attempt(response_received=False)


def test_a_reservation_and_its_settlement_are_stated_together_or_not_at_all() -> None:
    """A reservation with no settlement is money the run cannot account for."""
    with pytest.raises(LedgerSchemaError):
        answered_attempt(cost_settlement=None)
    with pytest.raises(LedgerSchemaError):
        answered_attempt(cost_reservation_usd=None)


# -- a call row ----------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com/v1/responses?api_key=sk-x",
        "http://api.openai.com/v1/responses",
        "https://api.openai.com/v1/responses#fragment",
        "api.openai.com/v1/responses",
    ],
)
def test_a_recorded_endpoint_carrying_a_query_or_no_scheme_is_refused(url: str) -> None:
    with pytest.raises(LedgerSchemaError):
        call(0, wire_request_url=url)


def test_a_call_that_ended_before_it_started_is_refused() -> None:
    with pytest.raises(LedgerSchemaError):
        call(0, started_at=LATER, ended_at=INSTANT)


@pytest.mark.parametrize("attempts", [(), [answered_attempt()], None])
def test_a_call_row_with_no_attempts_describes_a_request_never_dispatched(
    attempts: Any,
) -> None:
    with pytest.raises(LedgerSchemaError):
        call(0, attempts=attempts)


def test_an_attempt_slot_holding_something_that_is_not_an_attempt_is_refused() -> None:
    with pytest.raises(LedgerSchemaError):
        call(0, attempts=({"attempt_index": 1, "outcome": "response"},))


def test_the_exact_wire_request_is_stated_in_full_or_not_at_all() -> None:
    """Half a wire record cannot say which bytes the recorded answer came from."""
    complete = call(0)
    assert complete.wire_request_digest_sha256 is not None
    with pytest.raises(LedgerSchemaError):
        call(0, wire_request_bytes=None)
    with pytest.raises(LedgerSchemaError):
        call(0, wire_request_url=None)
    none_at_all = call(
        0,
        wire_request_digest_sha256=None,
        wire_request_bytes=None,
        wire_request_method=None,
        wire_request_url=None,
    )
    assert none_at_all.wire_request_digest_sha256 is None


def test_a_decision_slot_holding_a_plain_mapping_is_refused() -> None:
    with pytest.raises(LedgerSchemaError):
        call(0, decision={"kind": "RETRIEVE", "decision_digest_sha256": DIGEST_D})


@pytest.mark.parametrize("kind", ["a kind with spaces", "", 7, "x" * 300])
def test_a_decision_kind_that_is_not_a_bounded_identifier_is_refused(kind: Any) -> None:
    with pytest.raises(LedgerSchemaError):
        decision(kind=kind)


@pytest.mark.parametrize("digest", ["D" * 64, "d" * 63, "z" * 64, 7, ""])
def test_a_decision_digest_that_is_not_a_lowercase_sha256_is_refused(
    digest: Any,
) -> None:
    with pytest.raises(LedgerSchemaError):
        decision(decision_digest_sha256=digest)


def test_an_error_this_build_never_classified_ends_the_turn_it_happened_in() -> None:
    """A later attempt after one describes a turn that cannot have happened."""
    provisional = call(
        0,
        attempts=(
            unclassified_attempt(attempt_index=1),
            faulted_attempt(attempt_index=2),
        ),
        terminal_reason=TURN_END_RETRIES_EXHAUSTED,
        decision=None,
    )
    with pytest.raises(LedgerSchemaError):
        LedgerCall.from_row(provisional.as_dict(), "row 1.call", ledger_version=3)


def test_a_turn_that_states_no_exit_is_not_checked_against_its_last_attempt() -> None:
    """``None`` is a turn whose request never reached the retry loop at all."""
    unstated = call(
        0, attempts=(unclassified_attempt(),), terminal_reason=None, decision=None
    )
    assert unstated.terminal_reason is None


def test_a_turn_that_ended_unclassified_says_so_and_nothing_else() -> None:
    contradictory = call(
        0,
        attempts=(unclassified_attempt(),),
        terminal_reason=TURN_END_RETRIES_EXHAUSTED,
        decision=None,
    )
    with pytest.raises(LedgerSchemaError):
        LedgerCall.from_row(contradictory.as_dict(), "row 1.call", ledger_version=3)
    placed = call(
        0,
        attempts=(unclassified_attempt(),),
        terminal_reason=TURN_END_UNCLASSIFIED,
        decision=None,
    )
    placed.validate_for_ledger_version(3, "call 0")
    assert placed.attempts[0].outcome == ATTEMPT_OUTCOME_UNCLASSIFIED


# -- rebuilding a row from what a reader decoded --------------------------------


def _call_payload(**overrides: Any) -> dict[str, Any]:
    payload = call(0).as_dict()
    payload.update(overrides)
    return payload


def test_a_call_row_whose_attempts_are_not_an_array_is_refused() -> None:
    with pytest.raises(LedgerSchemaError):
        LedgerCall.from_row(_call_payload(attempts={"attempt_index": 1}), "row 1.call")


def test_a_call_row_whose_attempt_is_not_an_object_is_refused() -> None:
    with pytest.raises(LedgerSchemaError):
        LedgerCall.from_row(_call_payload(attempts=[["attempt_index", 1]]), "row 1.call")


def test_a_call_row_whose_decision_is_not_an_object_is_refused() -> None:
    with pytest.raises(LedgerSchemaError):
        LedgerCall.from_row(_call_payload(decision="RETRIEVE"), "row 1.call")


def test_a_call_row_survives_the_round_trip_it_was_written_through() -> None:
    original = call(0)
    assert LedgerCall.from_row(original.as_dict(), "row 1.call") == original


# -- the provider identity ------------------------------------------------------


@pytest.mark.parametrize(
    "base_url",
    ["http://api.openai.com/v1", "api.openai.com", "https://api openai.com", ""],
)
def test_a_base_url_that_is_not_a_bare_https_endpoint_is_refused(base_url: str) -> None:
    with pytest.raises(LedgerSchemaError):
        identity(base_url=base_url)


def test_a_settings_digest_that_does_not_cover_the_settings_beside_it_is_refused() -> (
    None
):
    """A digest a caller states independently can be made to agree with anything."""
    payload = provider_identity().as_dict()
    payload["settings_digest_sha256"] = DIGEST_A
    with pytest.raises(LedgerChainError):
        LedgerProviderIdentity.from_row(payload, "row 0.header.provider")


def test_a_provider_identity_survives_the_round_trip_it_was_written_through() -> None:
    original = provider_identity()
    assert LedgerProviderIdentity.from_row(original.as_dict(), "p") == original


# -- the control block ----------------------------------------------------------


def test_an_expected_call_count_above_the_safety_bound_is_refused() -> None:
    """It would make the bound a forecast the run is already known to break."""
    with pytest.raises(LedgerSchemaError):
        controls(max_provider_calls=60, expected_provider_calls=61)


@pytest.mark.parametrize("rate", [1.25, 125, None, ["1.25"]])
def test_a_pinned_rate_that_is_not_an_exact_decimal_string_is_refused(
    rate: Any,
) -> None:
    """Without it no amount on any row below can be re-derived."""
    with pytest.raises(LedgerSchemaError):
        controls(pricing=pricing(input_usd_per_mtok=rate))


def test_a_rate_table_missing_a_rate_altogether_is_refused_at_the_header() -> None:
    table = pricing()
    del table["output_usd_per_mtok"]
    with pytest.raises(LedgerSchemaError):
        controls(pricing=table)


@pytest.mark.parametrize("rate", ["1,25", "one", ""])
def test_a_rate_string_that_is_not_a_decimal_is_refused(rate: str) -> None:
    with pytest.raises(LedgerSchemaError):
        controls(pricing=pricing(output_usd_per_mtok=rate))


@pytest.mark.parametrize("rate", ["-0.01", "NaN", "-Infinity"])
def test_a_negative_or_non_finite_rate_is_refused(rate: str) -> None:
    with pytest.raises(LedgerSchemaError):
        controls(pricing=pricing(input_usd_per_mtok=rate))


def test_a_pricing_digest_that_does_not_cover_the_table_beside_it_is_refused() -> None:
    payload = controls().as_dict()
    payload["pricing_digest_sha256"] = DIGEST_B
    with pytest.raises(LedgerChainError):
        LedgerControls.from_row(payload, "row 0.header.controls")


def test_a_control_block_survives_the_round_trip_it_was_written_through() -> None:
    original = controls()
    assert LedgerControls.from_row(original.as_dict(), "c") == original


# -- the header -----------------------------------------------------------------


@pytest.mark.parametrize(
    "instance", ["not-an-instance", "opinst_" + "z" * 32, "", "opinst_1"]
)
def test_a_header_naming_something_that_is_not_an_operation_instance_is_refused(
    instance: str,
) -> None:
    with pytest.raises(LedgerSchemaError):
        header(operation_instance_id=instance)


def test_a_header_whose_provider_block_is_a_plain_mapping_is_refused() -> None:
    with pytest.raises(LedgerSchemaError):
        header(provider=provider_identity().as_dict())


def test_a_header_whose_control_block_is_a_plain_mapping_is_refused() -> None:
    with pytest.raises(LedgerSchemaError):
        header(controls=controls().as_dict())


def test_a_header_row_whose_provider_or_controls_are_not_objects_is_refused() -> None:
    payload = header().as_dict()
    with pytest.raises(LedgerSchemaError):
        LedgerHeader.from_row({**payload, "provider": "openai"}, "row 0.header")
    with pytest.raises(LedgerSchemaError):
        LedgerHeader.from_row({**payload, "controls": ["max_provider_calls"]}, "row 0")


def test_a_header_survives_the_round_trip_it_was_written_through() -> None:
    original = header()
    assert LedgerHeader.from_row(original.as_dict(), "row 0.header") == original


# -- amounts have to follow from the numbers beside them ------------------------


def test_a_run_with_no_cost_guard_states_no_amounts_and_that_is_admissible() -> None:
    """No cap means no amount was authorised — never that nothing was authorised."""
    unpriced = call(
        0,
        attempts=(
            answered_attempt(
                cost_reservation_usd=None,
                cost_settlement=None,
                cost_measured_usd=None,
            ),
        ),
    )
    check_call_amounts(unpriced, controls=controls())
    assert unpriced.attempts[0].cost_reservation_usd is None


def test_a_measured_settlement_with_no_counts_is_refused_by_the_accounting_net() -> None:
    """Two independent nets: the schema refuses this, and so does the arithmetic.

    The attempt is built valid and then mutated in place, which is the only way
    to ask the second net a question the first one would have answered first. A
    reader that trusted the schema alone would report an amount nothing measured.
    """
    priced = call(0)
    object.__setattr__(priced.attempts[0], "input_tokens", None)
    with pytest.raises(LedgerChainError):
        check_call_amounts(priced, controls=controls())


def test_a_measured_cost_that_is_not_its_own_counts_at_the_run_s_rates_is_refused() -> (
    None
):
    edited = call(0, attempts=(answered_attempt(cost_measured_usd=RESERVATION_USD),))
    with pytest.raises(LedgerChainError):
        check_call_amounts(edited, controls=controls())
    honest = call(0, attempts=(answered_attempt(cost_measured_usd=MEASURED_USD),))
    check_call_amounts(honest, controls=controls())


# -- an ending the rows above it have to support --------------------------------


def test_a_scored_run_over_an_unclassified_attempt_is_refused() -> None:
    """It never reached a business terminal, so nothing scored it."""
    with pytest.raises(LedgerChainError):
        check_terminal_admissible(
            TERMINAL_SCORED,
            None,
            (
                call(
                    0,
                    attempts=(unclassified_attempt(),),
                    terminal_reason=TURN_END_UNCLASSIFIED,
                    decision=None,
                ),
            ),
        )


def test_an_excluded_run_cannot_borrow_this_modules_own_abandonment_code() -> None:
    with pytest.raises(LedgerChainError):
        check_terminal_admissible(TERMINAL_EXCLUDED, ABORTED_CODE, ())


# -- the terminal row ------------------------------------------------------------


def _totals() -> Any:
    return recompute_totals((call(0),))


def test_a_terminal_row_whose_totals_are_a_plain_mapping_is_refused() -> None:
    with pytest.raises(LedgerSchemaError):
        LedgerTerminal(kind=TERMINAL_SCORED, ended_at=LATER, totals=_totals().as_dict())


def test_an_aborted_run_states_this_modules_own_abort_code_and_no_other() -> None:
    with pytest.raises(LedgerSchemaError):
        LedgerTerminal(
            kind=TERMINAL_ABORTED,
            ended_at=LATER,
            exclusion_code="provider_transport",
            totals=_totals(),
        )
    stated = LedgerTerminal(
        kind=TERMINAL_ABORTED,
        ended_at=LATER,
        exclusion_code=ABORTED_CODE,
        totals=_totals(),
    )
    assert stated.exclusion_code == ABORTED_CODE


def test_a_terminal_detail_is_this_builds_own_fixed_sentence_for_its_kind() -> None:
    """The one field that reads like prose carries no text from anywhere else."""
    with pytest.raises(LedgerSchemaError):
        LedgerTerminal(
            kind=TERMINAL_EXCLUDED,
            ended_at=LATER,
            exclusion_code="provider_transport",
            exclusion_detail="the provider said the account was over its quota",
            totals=_totals(),
        )


def test_a_terminal_row_whose_totals_block_is_not_an_object_is_refused() -> None:
    payload = LedgerTerminal(
        kind=TERMINAL_SCORED, ended_at=LATER, totals=_totals()
    ).as_dict()
    with pytest.raises(LedgerSchemaError):
        LedgerTerminal.from_row({**payload, "totals": "none"}, "row 2.terminal")


def test_a_terminal_row_survives_the_round_trip_it_was_written_through() -> None:
    original = LedgerTerminal(
        kind=TERMINAL_EXCLUDED,
        ended_at=LATER,
        exclusion_code="provider_transport",
        totals=_totals(),
    )
    assert LedgerTerminal.from_row(original.as_dict(), "row 2.terminal") == original


def test_a_terminal_row_has_no_field_a_business_result_could_be_put_in() -> None:
    stated = LedgerTerminal(kind=TERMINAL_SCORED, ended_at=LATER, totals=_totals())
    assert set(stated.as_dict()) == {
        "kind",
        "ended_at",
        "exclusion_code",
        "exclusion_detail",
        "totals",
    }
    assert ATTEMPT_OUTCOME_RESPONSE not in stated.as_dict()
    assert ATTEMPT_OUTCOME_FAULT not in stated.as_dict()
    assert ATTEMPT_SETTLEMENT_MEASURED not in stated.as_dict()
    assert TURN_END_RESPONSE not in stated.as_dict()
    assert DIGEST_C not in stated.as_dict()
