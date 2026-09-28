"""Static contribution metadata; execution belongs to the shared SDK lane."""

from operatebench.sdk import (
    CheckRequest,
    CommandResult,
    OperationPackMetadata,
    ReplayRequest,
    RunRequest,
    ValidateRequest,
)
from operatebench.sdk.profile_packs.pack import COMPLIANCE_COMMANDS


class ComplianceProfilesPack:
    metadata = OperationPackMetadata(
        pack_id="lettings.property_compliance.profiles.v1",
        operation_type="lettings.property_compliance.profiles",
        operation_id="lettings_property_compliance_profiles_v1",
        pack_version="0.1.0",
        display_name="Property compliance jurisdiction profiles",
        owner_id="prospire",
        owner_display_name="PROSPIRE TECHNOLOGIES LTD",
        contribution_kind="maintainer",
        status="incubator",
        privacy_status="SYNTHETIC_ONLY",
        default_spec="examples/contributions/prospire/property_compliance_profiles/operation.yaml",
        agent_ids=(
            "alternate_delivery",
            "delay_remedy",
            "force_access",
            "premature_complete",
            "reference",
            "reference-mock",
            "skip_council",
            "skip_reads",
            "stale_version",
            "trust_bad_report",
            "trust_repair_claim",
            "unnecessary_review",
            "wrong_recipient",
        ),
        aliases=("lettings.property_compliance.profiles",),
        evidence_eligible=False,
    )

    def validate(self, request: ValidateRequest) -> CommandResult:
        return COMPLIANCE_COMMANDS.validate(request)

    def run(self, request: RunRequest) -> CommandResult:
        return COMPLIANCE_COMMANDS.run(request)

    def replay(self, request: ReplayRequest) -> CommandResult:
        return COMPLIANCE_COMMANDS.replay(request)

    def check(self, request: CheckRequest) -> CommandResult:
        return COMPLIANCE_COMMANDS.check(request)


PACK = ComplianceProfilesPack()
