# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Mock safety-gate predicates for frozen operators, not historical execution proof.

Only imported gate versions and model-visible disclosures are restored for these tests.
Core, evaluator, durable delivery receipts and recorder identities remain current.
No production activation knob, request mapping, ceiling or historical fixture is changed.
"""

import importlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from operatebench.domains.lettings.maintenance import operation


@pytest.fixture()
def historical_canary_gate_build(request, monkeypatch):
    frozen = json.loads(
        (
            Path(__file__).parent
            / "fixtures/maintenance-card-v2/source-owner-projection-v4.json"
        ).read_text()
    )["action_contracts"]
    current = operation.action_schema_view

    def historical_guidance(*, recovery_enabled=False):
        # Frozen canaries exercise the legacy envelope, never the successor.
        assert not recovery_enabled
        schemas = current(recovery_enabled=False)
        for action, schema in schemas.items():
            schema.pop("field_guidance", None)
            if "field_guidance" in frozen[action]:
                schema["field_guidance"] = deepcopy(frozen[action]["field_guidance"])
        return schemas

    current_policy = operation.MaintenanceOperation.policy_view

    def historical_policy(self):
        policy = dict(current_policy(self))
        # These 0.13 disclosures were absent from the frozen request envelope.
        # The actual scenario budget and evaluator enforcement remain unchanged.
        policy.pop("human_checkpoint_budget")
        policy.pop("human_checkpoint_guidance")
        policy.pop("maintenance_contract_version")
        return policy

    current_retrieval = operation.serve_maintenance_retrieval

    def historical_retrieval(record, requests, as_of):
        # Frozen V1 SDK envelopes predate per-message delivery disclosure.
        # Strip only the detached read projection, never state or receipt evidence.
        record = deepcopy(record)
        for message in record.get("communications", []):
            assert message.pop("dispatch_status") == "DELIVERED"
            assert "recovery_of" not in message
        return current_retrieval(record, requests, as_of)

    monkeypatch.setattr(operation, "serve_maintenance_retrieval", historical_retrieval)
    monkeypatch.setattr(operation.MaintenanceOperation, "policy_view", historical_policy)
    monkeypatch.setattr(operation, "action_schema_view", historical_guidance)
    modules = [
        getattr(request.module, alias, None)
        for alias in (
            "canary",
            "haiku_canary",
            "sonnet_canary",
            "openai_canary",
            "proven",
        )
    ]
    if all(module is None for module in modules):
        # Shared envelope tests select their operator by parameter instead of import.
        modules.append(
            importlib.import_module("tools." + request.node.callspec.params["name"])
        )
    for module in modules:
        if module is not None and hasattr(module, "OPERATEBENCH_VERSION"):
            monkeypatch.setattr(module, "OPERATEBENCH_VERSION", "0.12.0")
