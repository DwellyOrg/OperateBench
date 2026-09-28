"""Contribution exact-key payload validation remains isolated from providers."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from operatebench.contributions.prospire.property_compliance_profiles.operation import (
    SCHEMAS,
    ComplianceDomain,
)

from .test_pack import FIXTURE, load_spec


@pytest.mark.parametrize("action", SCHEMAS)
def test_extra_missing_and_invalid_payloads_never_mutate(action):
    domain = ComplianceDomain(load_spec(FIXTURE), "EN_normal")
    state = domain.initial_state()
    before = deepcopy(state)
    context = SimpleNamespace(now="2026-09-26T09:00:00Z", record=lambda *args: None)
    valid_shape = dict.fromkeys(SCHEMAS[action], "synthetic-id")
    invalid = [valid_shape | {"invented": "extra"}]
    for field in SCHEMAS[action]:
        invalid += [
            {k: v for k, v in valid_shape.items() if k != field},
            valid_shape | {field: ""},
            valid_shape | {field: None},
        ]
    for payload in invalid:
        verdict = domain.apply_action(state, action, payload, (), context)
        assert not verdict.accepted and verdict.code == "PAYLOAD"
        assert (
            verdict.detail == domain.action_schemas()[action]["field_guidance"]["payload"]
        )
        assert state == before
