# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Current preview disclosures, distinct from historical evidence."""

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_current_claims_are_scoped_and_normative():
    methodology = (ROOT / "docs/METHODOLOGY.md").read_text()
    verification = (ROOT / "PUBLIC_CANDIDATE_VERIFICATION.md").read_text()
    readme = (ROOT / "README.md").read_text()
    assert "| Execution ledger | 3 |" in verification
    assert "lifecycle_openai_responses_model_request_v9" in verification
    assert "`reliable=false`" in verification
    assert "not mean three reliable references" in verification
    assert "Stateful time-removed implementation is pending" in methodology
    assert "only a bundled lifecycle-protocol differential" in methodology
    assert "not sufficient construct validation" in methodology
    assert "Supplying neither is refused" in methodology
    assert "or both" in methodology
    for text in (methodology, readme):
        assert "Every agent in this build is deterministic" not in text
        assert "Model-capable" in text
