"""Ownership checks run without SDKs or pinned checkpoint assets."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tools import three_flow_profiles as profiles
from tools import three_flow_river as river
from tools import three_flow_river_native as native


def test_river_selection_and_legacy_controls(tmp_path, monkeypatch):
    model = profiles.RIVER_MODELS[0]
    monkeypatch.setattr(
        profiles,
        "SELECTIONS",
        tuple(
            replace(s, api="selected-api", settings_source="selected-source")
            if s.provider == "river"
            else s
            for s in profiles.SELECTIONS
        ),
    )
    monkeypatch.setattr(river, "BudgetChannel", lambda *a: None)
    monkeypatch.setattr(river, "family_tokenizer", lambda *a: None)
    monkeypatch.setattr(
        river, "sdk", lambda: (None, SimpleNamespace(RiverServiceStub=lambda _: None))
    )
    monkeypatch.setattr(river, "version", lambda _: "synthetic")
    monkeypatch.setattr(
        profiles,
        "river_legacy_fields",
        lambda _: {
            "temperature": 0.25,
            "top_p": 0.9,
            "top_k": 7,
            "thinking": False,
            "stop": ["sentinel"],
            "deepseek_reasoning_effort": 9,
        },
    )
    # Resolve the owner dynamically, not a second local copy of its decisions.
    if hasattr(river, "river_legacy_fields"):
        monkeypatch.setattr(river, "river_legacy_fields", profiles.river_legacy_fields)
    transport = river.RiverCampaignTransport(
        model=model,
        assets=tmp_path,
        channel=SimpleNamespace(offline_mock=True),
        guard=SimpleNamespace(_max_output_tokens=32768),
        max_output_tokens=32768,
        channel_options=river.GRPC_OPTIONS,
    )
    assert transport.api == "selected-api"
    assert transport.settings_source == "selected-source"
    assert transport.selection in profiles.SELECTIONS
    assert transport.settings["temperature"] == 0.25
    assert transport.settings["stop"] == ["sentinel"]
    keys = list(transport.settings)
    start = keys.index("temperature")
    assert keys[start : start + 6] == [
        "temperature",
        "top_p",
        "top_k",
        "stop",
        "thinking",
        "deepseek_reasoning_effort",
    ]


def test_renderer_and_recorded_mode_share_template_owner(monkeypatch, tmp_path):
    model = profiles.RIVER_MODELS[0]
    helper = getattr(profiles, "river_template_kwargs", None)
    assert callable(helper), "one template-mode owner is required"
    assert native.river_template_kwargs is helper
    monkeypatch.setattr(
        profiles, "river_template_kwargs", lambda *a: {"owner_probe": "shared"}
    )
    monkeypatch.setattr(native, "river_template_kwargs", profiles.river_template_kwargs)
    monkeypatch.setattr(
        native,
        "_asset",
        lambda m, a, name: b"{}" if name.endswith("json") else b"{{ owner_probe }}",
    )
    monkeypatch.setattr(native, "_messages", lambda *a: [])
    assert profiles.river_mode_fields(model, "off-or-minimum-v1")["template_kwargs"] == {
        "owner_probe": "shared"
    }
    assert (
        native.render_messages(model, [], [], tmp_path, mode_profile="off-or-minimum-v1")
        == "shared"
    )


def test_native_family_remains_strict_and_kimi_separate():
    assert not hasattr(profiles, "native_family")
    for model in ("unknown", "nvidia/Kimi-K2.6-NVFP4"):
        with pytest.raises(native.InterfaceAmbiguityError):
            native._family(model)
