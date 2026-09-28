"""Finite campaign choices do not change existing declarations or SDK bytes."""

import hashlib
import json
from pathlib import Path

import pytest

from operatebench.agents.transport import ModelRequest, content_digest
from tests.test_three_flow_providers import fixture
from tools import three_flow_campaign as campaign
from tools import three_flow_http as http
from tools.three_flow_admission import registry

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures/campaign_profile_goldens.json").read_text()
)


def test_campaign_selection_has_one_owner():
    assert campaign.selected_roster.__module__ == "tools.three_flow_profiles"
    assert http.minimum_effort.__module__ == "tools.three_flow_profiles"


@pytest.mark.parametrize("key,expected", GOLDEN["registry"].items())
def test_registry_semantic_golden(key, expected):
    mode, policy, roster = key.split("/")
    assert (
        registry(mode_profile=mode, transport_policy=policy, roster_profile=roster)
        == expected
    )


@pytest.mark.parametrize("key,expected", GOLDEN["http"].items())
def test_sdk_bytes_and_settings_golden(tmp_path, monkeypatch, key, expected):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    provider, model, mode, policy = key.split("/")
    budget, _, transport, _trial = fixture(
        tmp_path, provider, model=model, mode_profile=mode, transport_policy=policy
    )
    try:
        prompt = expected["prompt"]
        observation = prompt["observation"]
        from operatebench.agents.lifecycle_contract import AGENT_TOOL_NAMES

        request = ModelRequest(
            model=model,
            protocol_version=prompt["protocol_version"],
            max_output_tokens=transport.max_output_tokens,
            tool_names=AGENT_TOOL_NAMES,
            observation_digest_sha256=content_digest(observation),
            prompt_digest_sha256=content_digest(prompt),
            invocation_index=observation["invocation_index"],
            turn_index=observation["turn_index"],
            prompt=prompt,
        )
        transport.send(request)
        assert transport.settings == expected["settings"]
        assert transport.request_mapping == expected["mapping"]
        assert [
            hashlib.sha256(r["request_utf8"].encode()).hexdigest()
            for r in transport.wire.captures
        ] == [expected["wire"]]
    finally:
        transport.close()
        budget.close()


def test_unknown_legacy_defaults_and_refusals():
    assert http.minimum_effort("unrecognized") == "low"
    with pytest.raises(ValueError, match=r"^unknown roster profile$"):
        campaign.selected_roster("unrecognized")
    with pytest.raises(ValueError, match=r"^unknown transport policy$"):
        registry(transport_policy="bad", mode_profile="bad", roster_profile="bad")
    with pytest.raises(ValueError, match=r"^unknown reasoning mode profile$"):
        registry(mode_profile="bad", roster_profile="bad")
