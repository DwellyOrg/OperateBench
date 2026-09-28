"""Finite campaign choices do not change existing declarations or SDK bytes."""

import hashlib
import json

import pytest

from operatebench.agents.transport import ModelRequest, content_digest
from tests.campaign_profile_goldens import load_campaign_profile_goldens
from tests.test_three_flow_providers import fixture
from tools import three_flow_campaign as campaign
from tools import three_flow_http as http
from tools.three_flow_admission import registry

GOLDEN = load_campaign_profile_goldens()


def test_profile_fixture_lossless_serialization():
    # SHA256 of the original checked-in, expanded capture, not new SUT output.
    # Its UTF-8 JSON recipe preserves recursive key order (no sort_keys).
    # This proves lossless transformation, not historical executable provenance.
    encoded = (json.dumps(GOLDEN, indent=2, ensure_ascii=True) + "\n").encode()
    assert hashlib.sha256(encoded).hexdigest() == (
        "f2eae8ae8f02150440851487d42eeabb1f8767796d62b62955c98daaa6fa62a3"
    )


def test_profile_fixture_rejects_unknown_schema(monkeypatch):
    monkeypatch.setattr(
        json,
        "loads",
        lambda _: {"schema_version": 2, "http": {}, "registry": {}},
    )
    with pytest.raises(ValueError, match="unsupported campaign golden schema"):
        load_campaign_profile_goldens()


def test_profile_fixture_copies_are_independent():
    loaded = load_campaign_profile_goldens()
    cells = list(loaded["http"].values())
    cells[0]["prompt"]["observation"]["operation_instance_id"] = "changed"
    cells[0]["settings"].clear()
    matrices = list(loaded["registry"].values())
    matrices[0][0]["request_settings"]["stop"].append("changed")
    assert cells[1] == list(GOLDEN["http"].values())[1]
    assert matrices[1] == list(GOLDEN["registry"].values())[1]
    assert load_campaign_profile_goldens() == GOLDEN


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


def test_selection_catalog_drives_http_and_declaration(tmp_path, monkeypatch):
    from dataclasses import replace

    from tools import three_flow_profiles as profiles

    original = profiles.SELECTIONS
    monkeypatch.setattr(
        profiles,
        "SELECTIONS",
        tuple(
            replace(s, api="selected-api", settings_source="selected-source")
            if s.provider == "openai"
            else s
            for s in original
        ),
    )
    budget, _, transport, _ = fixture(tmp_path, "openai")
    try:
        assert transport.api == "selected-api"
        assert transport.settings_source == "selected-source"
        assert transport.selection in profiles.SELECTIONS
        row = next(r for r in registry() if r["provider"] == "openai")
        assert row["settings_source"] == "selected-source"
    finally:
        transport.close()
        budget.close()


def test_http_admission_controls_share_payload_owner(monkeypatch):
    from tools import three_flow_profiles as profiles

    original = profiles.http_mode_fields

    def controls(provider, model, mode):
        return {**original(provider, model, mode), "owner_probe": provider}

    monkeypatch.setattr(profiles, "http_mode_fields", controls)
    for row in registry(mode_profile="off-or-minimum-v1"):
        if row["provider"] != "river":
            assert row["request_settings"]["owner_probe"] == row["provider"]


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"transport_policy": "bad", "mode_profile": "bad"}, "unknown transport policy"),
        (
            {"transport_policy": "paced-safe-errors-v1", "mode_profile": "bad"},
            "pacing policy is Mistral-only",
        ),
        ({"mode_profile": "bad", "model": "unknown"}, "unknown reasoning mode profile"),
        ({"model": "unknown"}, "unsupported HTTP provider/model pair"),
        (
            {"provider": "river", "model": "Qwen/Qwen3.8-27B-FP8"},
            "unsupported HTTP provider/model pair",
        ),
        (
            {"model": "gpt-6-luna", "max_output_tokens": 0},
            "new models require explicit off-or-minimum profile",
        ),
        ({"max_output_tokens": 0}, "explicit output profile required"),
        ({"max_output_tokens": 1}, "successor flagship output profile is 128000"),
    ],
)
def test_http_refusal_order_before_client_construction(overrides, message):
    from types import SimpleNamespace

    kwargs = {
        "provider": "openai",
        "model": "gpt-6-astra",
        "api_key": "synthetic",
        "inner": None,
        "guard": SimpleNamespace(_max_output_tokens=1),
        "max_output_tokens": 128000,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError) as caught:
        http.HTTPCampaignTransport(**kwargs)
    assert str(caught.value) == message


def test_unknown_legacy_defaults_and_refusals():
    assert http.minimum_effort("unrecognized") == "low"
    with pytest.raises(ValueError, match=r"^unknown roster profile$"):
        campaign.selected_roster("unrecognized")
    with pytest.raises(ValueError, match=r"^unknown transport policy$"):
        registry(transport_policy="bad", mode_profile="bad", roster_profile="bad")
    with pytest.raises(ValueError, match=r"^unknown reasoning mode profile$"):
        registry(mode_profile="bad", roster_profile="bad")
