"""Public policy documentation describes executable invariants and scope."""

from pathlib import Path


def test_refund_policy_and_unimplemented_scope_are_self_contained():
    text = (
        Path(__file__).resolve().parents[1] / "docs/commerce-return-refund-profiles.md"
    ).read_text()
    policy = text.split("## Synthetic operational policy (not law)\n", 1)[1]
    approval = next(b for b in policy.split("\n- ") if b.startswith("Every refund"))
    assert " ".join(approval.split()) == (
        "Every refund requires authenticated simulated supervisor authority after "
        "carrier proof. The grant binds order, profile, exact amount, currency, original "
        "tender and proof. There is no statutory or executable "
        "low-value approval exemption."
    )
    scope = text.split("## Not implemented\n", 1)[1].strip()
    assert scope.startswith("Unimplemented branches include UK CRA faulty-goods")
    for branch in (
        "German",
        "other EU countries",
        "California",
        "AU major",
        "multi-line/partial refunds",
        "chargebacks",
    ):
        assert branch in scope
    assert (
        "Unknown payment query, duplicate webhook, failed tender, CAS rejection" in scope
    )
