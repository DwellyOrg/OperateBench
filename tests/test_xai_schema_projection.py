# SPDX-License-Identifier: Apache-2.0
"""xAI-only nonempty spelling; the canonical schema is untouched."""

import json
import re

import pytest

from operatebench.agents.lifecycle_contract import outcome_tools
from operatebench.agents.xai_responses import build_model_payload
from tests.test_grok46_successor import request_for


def patterns(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "pattern":
                yield (*path, key), child
            else:
                yield from patterns(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from patterns(child, (*path, index))


@pytest.mark.parametrize("model,ceiling", [("grok-4.5", 4096), ("grok-4.6", 8192)])
def test_projection_only_changes_all_fifteen_patterns(model, ceiling):
    original = outcome_tools()
    tools = build_model_payload(request_for(model, ceiling), model=model)["tools"]
    before, after = dict(patterns(original)), dict(patterns(tools))
    assert len(before) == len(after) == 15
    assert before.keys() == after.keys()
    assert set(before.values()) == {r"[\s\S]"}
    assert set(after.values()) == {r"^[\s\S]+$"}
    # Exact equality outside those leaves preserves strict/null/array rules.
    assert json.dumps(tools).replace(r"^[\\s\\S]+$", r"[\\s\\S]") == json.dumps(original)
    assert outcome_tools() == original


@pytest.mark.parametrize(
    "text", ["", "x", "get_case_record", "a\nb", "\n", "a\n", " ", "é"]
)
def test_json_schema_search_semantics(text):
    assert (
        bool(re.search(r"[\s\S]", text))
        == bool(re.search(r"^[\s\S]+$", text))
        == bool(text)
    )


@pytest.mark.parametrize("cell,bad_name", [("grok45", "I"), ("grok46", "g")])
@pytest.mark.parametrize("negative", [True, False])
def test_actual_sdk_keeps_name_before_canonical_fixture(
    cell, bad_name, negative, tmp_path
):
    from decimal import Decimal

    import httpx

    from operatebench.agents.evidence import WireCaptureTransport
    from tools import diagnose_aggregate_budget as runner
    from tools.aggregate_budget import Budget

    tmp_path.chmod(0o700)
    budget = Budget(
        tmp_path, cap=Decimal("10"), namespace="offline-test-name", create=True
    )
    module = runner.module(cell)
    script = module.ScriptedProvider(max_calls=100)
    bodies, replies = [], []
    first = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        assert set(dict(patterns(body["tools"])).values()) == {r"^[\s\S]+$"}
        if len(bodies) == 1:
            response = script(request)
            first.append(response)
            data = json.loads(response.content)
            call = next(v for v in data["output"] if v["type"] == "function_call")
            arguments = json.loads(call["arguments"])
            assert arguments["requests"][0]["tool"] == "get_case_record"
            if negative:
                arguments["requests"][0]["tool"] = bad_name
            call["arguments"] = json.dumps(arguments)
            replies.append(arguments["requests"][0]["tool"])
            return httpx.Response(200, json=data)
        if negative and len(bodies) == 2:
            # The SDK and Core saw the unmodified one-character name before
            # the canonical reference response is allowed to repair it.
            assert bad_name in body["input"][0]["content"]
            assert "UNKNOWN_RETRIEVAL_TOOL" in body["input"][0]["content"]
            return first[0]
        return script(request)

    wire = WireCaptureTransport()
    wire.attach(runner.DispatchTransport(httpx.MockTransport(handler), budget, cell))
    client = runner.client_for(cell, runner.PLACEHOLDER, wire)
    try:
        result = runner.execute_cell(
            cell, tmp_path, budget, client, wire, episode100=True
        )
        assert replies == [bad_name if negative else "get_case_record"]
        assert result.get("bundle_ok") and result.get("replay_ok"), result
        assert len(bodies) == (47 if negative else 46)
    finally:
        client.close()
        budget.close()
