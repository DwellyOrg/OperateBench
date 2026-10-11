"""Candidate identity only; intentionally no SDK command adapter or registration.

Executable Engine integration lives in the maintainer's targeted offline tests.
This object is not an OperationPack implementation and must not be dispatched.
"""

from operatebench.sdk import OperationPackMetadata


class BookkeepingCandidate:
    metadata = OperationPackMetadata(
        pack_id="accounting.bookkeeping_reconciliation.synthetic.v1",
        operation_type="accounting.bookkeeping_reconciliation.synthetic.v1",
        operation_id="bookkeeping_reconciliation_synthetic_v1",
        pack_version="0.1.0",
        display_name="Synthetic month-end bookkeeping reconciliation development slice",
        owner_id="prospire",
        owner_display_name="PROSPIRE TECHNOLOGIES LTD",
        contribution_kind="maintainer",
        status="incubator",
        privacy_status="SYNTHETIC_ONLY",
        default_spec="examples/contributions/prospire/bookkeeping_reconciliation/operation.yaml",
        agent_ids=(
            "alternate",
            "failed_zero",
            "premature_publish",
            "reference",
            "reuse_report",
            "skip_approval",
        ),
        aliases=(),
        evidence_eligible=False,
    )


PACK = BookkeepingCandidate()
