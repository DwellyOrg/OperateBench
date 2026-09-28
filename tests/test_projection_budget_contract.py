# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Engine-bound offline projection accounting, not provider admission limits."""

import copy
import json

import pytest

from operatebench.version import OPERATEBENCH_VERSION

HISTORICAL_BASELINE_BYTES_V1 = 295_404
HISTORICAL_BYTE_RATIO_CEILING = 1.25
HISTORICAL_CAP_V1 = 369_255
# Independently recorded pre-0.13 guidance census; never derived from the run.
INITIAL_GUIDANCE_FIELDS_V1 = {"policy": 15_732, "action_schemas": 79_793}
GUIDANCE_ALLOWANCES_0_13 = {"policy": 17_000, "action_schemas": 13_000}
INITIAL_ENVELOPE_BYTES_V1 = 11_040


def assert_projection_budget(projections, *, engine_version, scenario_id):
    assert engine_version == "0.13.0", "projection budget needs explicit version review"
    full = sum(len(json.dumps(p, sort_keys=True, default=str)) for p in projections)
    fields = {}
    for projection in projections:
        for name, value in projection.items():
            fields[name] = fields.get(name, 0) + len(
                json.dumps(value, sort_keys=True, default=str)
            )
    assert scenario_id in {"V1", "V2", "V3"}
    if scenario_id != "V1":
        # No incremental allowance for V2/V3; retain the legacy ceiling.
        assert full < HISTORICAL_CAP_V1, "legacy projection budget"
        return
    # Maintenance delivery-observability.v2 / delivery-recovery.v3 introduce
    # two separately identified disclosures. Compare the old projection after
    # removing ONLY those exact new leaves, not by increasing its old ceiling.
    # Their own serialized bounds are independently pinned below. Old guidance
    # and all non-guidance fields still spend the historical allowances.
    delivery_bytes = 0
    appearances = 0
    for projection in projections:
        schemas = projection.get("action_schemas", {})
        if not isinstance(schemas, dict):
            continue
        send = schemas.get("send_message", {})
        exceptions = schemas.get("request_exception_resolution", {})
        if "dispatch_status" not in send.get("field_guidance", {}):
            continue
        appearances += 1
        contract = projection["policy"]["maintenance_contract_version"]
        assert contract in {"delivery-observability.v2", "delivery-recovery.v3"}
        extra = {"dispatch_status": send["field_guidance"]["dispatch_status"]}
        exception_guidance = exceptions["field_guidance"]
        assert len(json.dumps(extra, sort_keys=True)) <= 350, (
            "delivery dispatch disclosure budget"
        )
        assert len(json.dumps(exception_guidance, sort_keys=True)) <= 273, (
            "exception disclosure budget"
        )
        reduced = copy.deepcopy(schemas)
        del reduced["send_message"]["field_guidance"]["dispatch_status"]
        del reduced["request_exception_resolution"]["field_guidance"]
        if contract == "delivery-recovery.v3":
            assert reduced["send_message"]["optional"].pop("recovery_of") == "integer"
        delivery_bytes += len(json.dumps(schemas, sort_keys=True)) - len(
            json.dumps(reduced, sort_keys=True)
        )
    assert appearances <= 23, "delivery disclosure repetition budget"
    fields["action_schemas"] = fields.get("action_schemas", 0) - delivery_bytes
    increments = {
        name: max(0, fields.get(name, 0) - initial)
        for name, initial in INITIAL_GUIDANCE_FIELDS_V1.items()
    }
    for name, increment in increments.items():
        assert increment <= GUIDANCE_ALLOWANCES_0_13[name], f"{name} guidance budget"
    incremental = sum(increments.values())
    # Guidance savings cannot pay for growth in the initial projection.
    assert full - delivery_bytes - incremental < HISTORICAL_CAP_V1, (
        "initial full projection budget"
    )
    assert sum(fields.values()) - incremental < (
        HISTORICAL_CAP_V1 - INITIAL_ENVELOPE_BYTES_V1
    ), "initial field projection budget"


@pytest.fixture(scope="module")
def v1_projections():
    from tests.test_operatebench_retrieval_reference import SPEC, census_run, load_spec

    return census_run(load_spec(SPEC), "V1")[1].projections


@pytest.mark.parametrize("leaf", ["dispatch", "exception", "unknown_contract", "repeat"])
def test_versioned_disclosures_cannot_hide_growth(v1_projections, leaf):
    projections = copy.deepcopy(v1_projections)
    ready = next(
        p for p in projections if "required" in p["action_schemas"]["send_message"]
    )
    if leaf == "dispatch":
        ready["action_schemas"]["send_message"]["field_guidance"]["dispatch_status"] += (
            "x"
        )
    elif leaf == "exception":
        ready["action_schemas"]["request_exception_resolution"]["field_guidance"][
            "exception_type"
        ] += "x"
    elif leaf == "unknown_contract":
        ready["policy"]["maintenance_contract_version"] = "unreviewed.v4"
    else:
        projections.append(copy.deepcopy(ready))
    with pytest.raises(AssertionError):
        assert_projection_budget(
            projections, engine_version=OPERATEBENCH_VERSION, scenario_id="V1"
        )


def test_historical_contract_is_unchanged():
    assert HISTORICAL_BASELINE_BYTES_V1 * HISTORICAL_BYTE_RATIO_CEILING == 369_255
    assert sum(GUIDANCE_ALLOWANCES_0_13.values()) == 30_000


@pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
def test_current_version_budget(scenario_id):
    from tests.test_operatebench_retrieval_reference import SPEC, census_run, load_spec

    projections = census_run(load_spec(SPEC), scenario_id)[1].projections
    assert_projection_budget(
        projections, engine_version=OPERATEBENCH_VERSION, scenario_id=scenario_id
    )


@pytest.mark.parametrize("engine_version", ["0.12.0", "0.13.1", "0.14.0"])
def test_allowance_does_not_follow_other_versions(v1_projections, engine_version):
    with pytest.raises(AssertionError, match="explicit version review"):
        assert_projection_budget(
            v1_projections, engine_version=engine_version, scenario_id="V1"
        )


@pytest.mark.parametrize(
    ("field", "growth", "failure"),
    [
        ("policy", 17_001, "policy guidance budget"),
        ("action_schemas", 13_001, "action_schemas guidance budget"),
        ("retrieval", 1_017, "initial full projection budget"),
    ],
)
def test_future_growth_fails(v1_projections, field, growth, failure):
    projections = copy.deepcopy(v1_projections)
    projections[0][field]["budget_regression_padding"] = "x" * growth
    with pytest.raises(AssertionError, match=failure):
        assert_projection_budget(
            projections, engine_version=OPERATEBENCH_VERSION, scenario_id="V1"
        )


def test_required_disclosures_survive_budget_revision(v1_projections):
    for projection in v1_projections:
        policy = projection["policy"]
        assert policy["human_checkpoint_budget"] == 1
        assert "maximum total number" in policy["human_checkpoint_guidance"]
        assert "Exceeding it fails" in policy["human_checkpoint_guidance"]
    schemas = json.dumps([p["action_schemas"] for p in v1_projections])
    assert "a new completion duty" in schemas
    assert "they do not discharge duties created later" in schemas


@pytest.mark.parametrize("field", ["policy", "action_schemas"])
def test_fixed_increment_boundary_is_inclusive_and_one_more_byte_fails(field):
    # Synthetic boundary independent of the current reference's measured size.
    size = INITIAL_GUIDANCE_FIELDS_V1[field] + GUIDANCE_ALLOWANCES_0_13[field]
    projections = [{field: "x" * (size - 2)}]  # JSON string quotes count too.
    assert_projection_budget(
        projections, engine_version=OPERATEBENCH_VERSION, scenario_id="V1"
    )
    projections[0][field] += "x"
    with pytest.raises(AssertionError, match=f"{field} guidance budget"):
        assert_projection_budget(
            projections, engine_version=OPERATEBENCH_VERSION, scenario_id="V1"
        )


@pytest.mark.parametrize("scenario_id", ["V2", "V3"])
def test_other_variants_cannot_spend_v1_allowance(scenario_id):
    overhead = len(json.dumps({"padding": ""}, sort_keys=True, default=str))
    projections = [{"padding": "x" * (HISTORICAL_CAP_V1 - overhead - 1)}]
    assert_projection_budget(
        projections, engine_version=OPERATEBENCH_VERSION, scenario_id=scenario_id
    )
    projections[0]["padding"] += "x"
    with pytest.raises(AssertionError, match="legacy projection budget"):
        assert_projection_budget(
            projections, engine_version=OPERATEBENCH_VERSION, scenario_id=scenario_id
        )
