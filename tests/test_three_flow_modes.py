"""Versioned successor modes; offline real SDK serialization, never paid calls."""

import pytest

from tests.test_three_flow_providers import fixture
from tools.three_flow_runtime import run_trial


@pytest.mark.parametrize("provider", ["openai", "anthropic", "mistral"])
def test_registry_binds_minimum_http(provider):
    from tools.three_flow_admission import registry

    legacy = registry()
    current = registry(mode_profile="off-or-minimum-v1")
    assert len(current) == 14
    assert sum(r["request_settings"]["reasoning_mode"] == "OFF" for r in current) == 10
    assert sum(r["request_settings"]["reasoning_mode"] == "MINIMUM" for r in current) == 4
    row = next(r for r in current if r["provider"] == provider)
    assert row["mapping"].endswith("-off-or-minimum-v1")
    assert "mode_profile" not in legacy[0]["request_settings"]
    assert registry() == legacy
    with pytest.raises(ValueError):
        registry(mode_profile="unrecognized")


@pytest.mark.parametrize("provider", ["openai", "anthropic", "mistral"])
def test_minimum_http_wire(tmp_path, provider):
    budget, mock, transport, trial = fixture(
        tmp_path, provider, mode_profile="off-or-minimum-v1"
    )
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        assert record["evaluation"]["reliable"]
        assert mock.calls
        expected = {
            "openai": {"reasoning": {"effort": "low"}},
            "anthropic": {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "low"},
            },
            "mistral": {"reasoning_effort": "none"},
        }[provider]
        for call in mock.calls:
            assert all(call[k] == v for k, v in expected.items())
            assert "prompt_mode" not in call
        assert all(transport.settings[k] == v for k, v in expected.items())
        assert transport.settings["mode_profile"] == "off-or-minimum-v1"
        assert transport.settings["reasoning_mode"] == "MINIMUM"
        assert transport.request_mapping.endswith("-off-or-minimum-v1")
        if provider == "openai":
            assert "reasoning" not in transport.settings["omitted"]
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("provider", ["openai", "anthropic", "mistral"])
def test_mode_admission_and_factory_binding(tmp_path, monkeypatch, provider):
    from decimal import Decimal

    from tests.test_three_flow_live import Network
    from tests.test_three_flow_live import fixture as admission_fixture
    from tools import three_flow_live as live
    from tools.three_flow_admission import VARIABLES, registry, validate
    from tools.three_flow_budget import CampaignBudget
    from tools.three_flow_live_factory import live_factory

    monkeypatch.setattr(live, "source_identity", lambda: "a" * 40)
    _, admission, slot = admission_fixture(tmp_path, provider)
    admission["mode_profile"] = "off-or-minimum-v1"
    admission["registry"] = registry(mode_profile="off-or-minimum-v1")
    assert validate(admission, "a" * 40, synthetic=True) == admission
    (tmp_path / "campaign").mkdir(mode=0o700)
    budget = CampaignBudget(
        tmp_path / "campaign", campaign_id="test-series", cap=Decimal("1000"), create=True
    )
    try:
        transport = live_factory(
            admission, {VARIABLES[provider]: "offline-dummy"}, network=Network()
        )(slot, budget)
        try:
            assert transport.settings["mode_profile"] == "off-or-minimum-v1"
            assert transport.request_mapping == next(
                r["mapping"] for r in admission["registry"] if r["model"] == slot["model"]
            )
        finally:
            transport.close()
    finally:
        budget.close()
    admission["registry"] = registry()
    with pytest.raises(ValueError, match="source/runtime/scope/gates"):
        validate(admission, "a" * 40, synthetic=True)
