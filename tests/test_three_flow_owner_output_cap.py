"""Explicit River operational selection, never a provider maximum."""

from types import SimpleNamespace

import pytest

from tests.test_three_flow_context import diagnostic
from tests.test_three_flow_live import fixture
from tools.three_flow_admission import validate_profile
from tools.three_flow_campaign import ROSTER
from tools.three_flow_live_factory import AdmittedGuard


@pytest.mark.parametrize("provider,model", ROSTER)
def test_all_registry_identity_mutants_refused(tmp_path, monkeypatch, provider, model):
    from tools.three_flow_admission import validate

    monkeypatch.setenv("THREE_FLOW_RIVER_ASSETS", "/unused")
    _, admission, _ = fixture(tmp_path, "mistral")
    row = next(r for r in admission["registry"] if r["model"] == model)
    row["request_settings"]["owner_output_cap_tokens"] = 30000
    with pytest.raises(ValueError):
        validate(admission, "a" * 40, synthetic=True)


def test_cap_change_invalidates_owner_authorization(tmp_path, monkeypatch):
    from tests.test_three_flow_context import bind_combined
    from tests.test_three_flow_owner_pricing import assumed_fixture

    monkeypatch.setenv("THREE_FLOW_RIVER_ASSETS", "/unused")
    _, admission, slot = assumed_fixture(tmp_path, "river")
    p = diagnostic(admission["profiles"][slot["model"]])
    p["owner_output_cap_tokens"] = 30000
    bind_combined(p)
    validate_profile(p, "river")
    p["owner_output_cap_tokens"] = 30001
    with pytest.raises(ValueError):
        validate_profile(p, "river")


@pytest.fixture(autouse=True)
def synthetic_source(monkeypatch):
    from tools import three_flow_live

    monkeypatch.setattr(three_flow_live, "source_identity", lambda: "a" * 40)


def test_owner_output_cap_admission_and_context(tmp_path, monkeypatch):
    monkeypatch.setenv("THREE_FLOW_RIVER_ASSETS", "/unused")
    _, admission, slot = fixture(tmp_path, "river")
    p = diagnostic(admission["profiles"][slot["model"]])
    p["owner_output_cap_tokens"] = 30000
    validate_profile(p, "river")
    guard = object.__new__(AdmittedGuard)
    guard.profile = p
    guard.budget = SimpleNamespace(totals=lambda: {"remaining": None})
    assert guard.output_bound(6307) == 30000
    assert guard.output_bound(p["context_max"] - 17) == 17
    guard.check_context(6307, 30000)
    with pytest.raises(ValueError):
        guard.check_context(6307, 30001)
    assert p["output_max"] is None
    assert p["guarantees"]["provider_bounds_verified"] is False
    del p["owner_output_cap_tokens"]
    validate_profile(p, "river")
    assert guard.output_bound(6307) == p["context_max"] - 6307


@pytest.mark.parametrize("cap", [True, None, 0, -1, 30000.0, "30000"])
def test_invalid_owner_cap(tmp_path, monkeypatch, cap):
    monkeypatch.setenv("THREE_FLOW_RIVER_ASSETS", "/unused")
    _, admission, slot = fixture(tmp_path, "river")
    p = diagnostic(admission["profiles"][slot["model"]])
    p["owner_output_cap_tokens"] = cap
    with pytest.raises(ValueError):
        validate_profile(p, "river")


@pytest.mark.parametrize("provider", ["openai", "anthropic", "mistral"])
def test_http_cannot_select_river_cap(tmp_path, provider):
    _, admission, slot = fixture(tmp_path, provider)
    p = admission["profiles"][slot["model"]]
    p["owner_output_cap_tokens"] = 30000
    with pytest.raises(ValueError):
        validate_profile(p, provider)
