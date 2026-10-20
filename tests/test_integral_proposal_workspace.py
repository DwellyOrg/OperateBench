"""The pending proposal has no canonical contribution or authorship authority."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_integral_proposal_workspace_is_pending_and_test_only():
    proposal = ROOT / "tests/proposals/integral/bookkeeping_reconciliation"
    for tree in (
        "src/operatebench/contributions",
        "tests/contributions",
        "examples/contributions",
    ):
        assert not (ROOT / tree / "prospire/bookkeeping_reconciliation").exists()
        assert not (ROOT / tree / "integral/bookkeeping_reconciliation").exists()
    for name in ("__init__.py", "agents.py", "evaluator.py", "operation.py", "spec.py"):
        assert (proposal / name).is_file()
    for name in ("OWNER.yaml", "CONTRIBUTION.yaml", "pack.py"):
        assert not list(proposal.rglob(name))
    fixtures = ROOT / "examples/proposals/integral/bookkeeping_reconciliation"
    assert (fixtures / "operation.yaml").is_file()
    assert (fixtures / "negative_controls.json").is_file()
    doc = (ROOT / "docs/proposals/integral/bookkeeping_reconciliation.md").read_text()
    for claim in (
        "Maintainer-authored proposal for Integral feedback",
        "pending acceptance",
        "not their contribution or endorsement",
        "not an official pack",
        "not a tax-law test",
        "substantively review, adopt or edit",
        "sign their contribution",
    ):
        assert claim in doc
