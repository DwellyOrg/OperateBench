"""Bounded 420b984 oracles: exact ordered settings, rendering and 66 SDK RPCs.

Expected hashes come from the real baseline, not a candidate re-recording.
The synthetic first prompt is reused verbatim from the HTTP baseline fixture.
Run the optional native test with the reviewed River SDK and explicit local assets.
"""

import hashlib
import json
import os
import socket
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

from tools import three_flow_profiles as profiles

FIXTURES = Path(__file__).parent / "fixtures"
NATIVE = json.loads((FIXTURES / "campaign_native_goldens.json").read_text())
HTTP = json.loads((FIXTURES / "campaign_profile_goldens.json").read_text())


def test_selections_equal_supported_golden_matrix():
    expected = {tuple(key.split("/")[:3]) for key in HTTP["http"]} | {
        ("river", c["model"], c["mode"]) for c in NATIVE["cells"]
    }
    actual = [(s.provider, s.model, s.mode_profile) for s in profiles.SELECTIONS]
    assert len(actual) == len(set(actual))
    assert set(actual) == expected
    for selection in profiles.SELECTIONS:
        assert (
            profiles.selection_for(
                *(selection.provider, selection.model, selection.mode_profile)
            )
            is selection
        )
    assert profiles.selection_for("river", "unknown", "legacy-v1") is None
    assert profiles.selection_for("openai", "gpt-6-luna", "legacy-v1") is None


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="official River SDK requires Python >=3.12"
)
@pytest.mark.parametrize(
    "cell", NATIVE["cells"], ids=lambda c: c["model"] + "/" + c["mode"]
)
def test_native_baseline_hashes(tmp_path, monkeypatch, cell):
    assets = os.environ.get("THREE_FLOW_RIVER_ASSETS")
    if assets is None:
        pytest.skip("explicit pinned local River asset cache required")

    def deny(*args, **kwargs):
        raise AssertionError("offline test: network/client forbidden")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    import grpc
    import river_client

    monkeypatch.setattr(grpc, "secure_channel", deny)
    monkeypatch.setattr(grpc, "insecure_channel", deny)
    monkeypatch.setattr(river_client, "Client", deny)
    from operatebench.agents.lifecycle_contract import AGENT_TOOL_NAMES
    from operatebench.agents.transport import ModelRequest, content_digest
    from tests.test_three_flow_river import fixture
    from tools import three_flow_river_mock as mock
    from tools.three_flow_river_assets import family_render

    for package, expected in NATIVE["versions"].items():
        assert version(package) == expected
    monkeypatch.setattr(mock, "uuid4", lambda: "00000000-0000-0000-0000-000000000001")
    prompt = HTTP["http"][NATIVE["prompt_fixture"]]["prompt"]
    model, mode = cell["model"], cell["mode"]
    rendered = family_render(model, prompt, Path(assets), mode_profile=mode).encode()
    assert len(rendered) == cell["render_bytes"]
    assert hashlib.sha256(rendered).hexdigest() == cell["render_sha256"]
    budget, channel, transport, _ = fixture(
        tmp_path, Path(assets), model, polls=1, mode_profile=mode
    )
    try:
        observation = prompt["observation"]
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
        ordered = json.dumps(
            transport.settings, ensure_ascii=False, separators=(",", ":")
        )
        assert hashlib.sha256(ordered.encode()).hexdigest() == cell["settings_sha256"]
        assert [r["request_sha256"] for r in transport.wire.captures] == cell[
            "wire_sha256"
        ]
        assert len(channel.submissions) == 1
        assert len(channel.polls) == 2
        proto = channel.submissions[0]
        assert [f.name for f, _ in proto.ListFields()] == cell["protobuf_present_fields"]
        assert [f.name for f, _ in proto.prompts[0].ListFields()] == cell[
            "prompt_present_fields"
        ]
        assert transport.reasoning_prefilled is cell["reasoning_prefilled"]
        assert transport.selection == profiles.selection_for("river", model, mode)
        assert transport.api == transport.selection.api
        assert transport.settings_source == transport.selection.settings_source
        suffix = "" if mode == "legacy-v1" else "-" + mode
        assert transport.request_mapping == "three-flow-river-native-v1" + suffix
        assert transport.settings["stop"] == (["<|im_end|>"] if "Kimi" in model else [])
        assert all(r["state"] == "settled" for r in budget.records.values())
    finally:
        transport.close()
        budget.close()
