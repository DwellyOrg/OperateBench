"""Public preview ships fictional permission examples, not access records."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_intake_is_only_a_fictional_test_fixture():
    assert not (ROOT / ".github/partner-intake").exists()
    fixture = ROOT / "tests/fixtures/partner-intake/companies/fictional_example.json"
    record = json.loads(fixture.read_text())
    assert record["company_id"] == "fictional_example"
    assert record["contact"]["email"].endswith(".invalid")
    assert record["identity"]["website_url"] == "https://fictional.example/"
    assert record["access"]["provider_model_execution"] == "HOLD"
    assert record["provenance"]["partner_confirmed"] is False


def test_timeout_demo_has_no_inherited_paid_authority():
    from decimal import Decimal

    import pytest

    from tools import episode100_grok46_timeout120 as demo
    from tools import run_episode100 as launch

    assert demo.PROFILE.startswith("synthetic-demo-")
    assert demo.SERIES["series_id"].startswith("synthetic-demo-")
    assert demo.SERIES["authorized_total_usd"] == "20"
    assert demo.SERIES["committed_exposure_usd"] == "2"
    assert demo.SERIES["new_cap_usd"] == "18"
    with pytest.raises(ValueError, match="new external authority required"):
        launch.consume(None, output=Path("not-created"), cap=Decimal("18"), mock=False)
