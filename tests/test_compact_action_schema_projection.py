# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Compact model disclosure preserves the full nonempty action contract."""

import copy

import pytest

from operatebench.domains.lettings.maintenance.spec import action_schema_view
from tests.test_operatebench_phase2_state_removal import SPEC, census_run, load_spec


@pytest.mark.parametrize("scenario", ["V1", "V2", "V3"])
def test_omitted_empty_declarations_reconstruct_the_unabridged_contract(scenario):
    _outcome, census = census_run(load_spec(SPEC), scenario)
    original = action_schema_view()
    count = 0
    for projection in census.projections:
        for name, schema in projection["action_schemas"].items():
            if "required" not in schema:
                continue  # The existing unmet-read boundary is unchanged.
            restored = copy.deepcopy(schema)
            restored.setdefault("optional", {})
            restored.setdefault("evidence_refs", {"required": {}})
            expected = {
                key: value for key, value in original[name].items() if key != "reads"
            }
            assert restored == expected
            assert schema.get("optional") != {}
            assert schema.get("evidence_refs") != {"required": {}}
            count += 1
    assert count > 0
