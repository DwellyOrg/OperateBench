"""Reviewed integration-owned factories for the two finite profile processes."""

from pathlib import Path

from operatebench.contributions.prospire import property_compliance_profiles as compliance
from operatebench.contributions.prospire.property_compliance_profiles.agents import (
    AGENT_IDS as COMPLIANCE_AGENTS,
)
from operatebench.contributions.prospire.property_compliance_profiles.spec import (
    Spec as ComplianceSpec,
)
from operatebench.core.errors import SpecSchemaError
from operatebench.domains.commerce import return_refund_profiles as commerce
from operatebench.domains.commerce.return_refund_profiles.agents import (
    AGENT_IDS as COMMERCE_AGENTS,
)
from operatebench.sdk.api import (
    CheckRequest,
    CommandResult,
    OperationPackMetadata,
    ReplayRequest,
    RunRequest,
    ValidateRequest,
)
from operatebench.sdk.development_pack import DevelopmentCommands
from operatebench.sdk.development_runtime import DevelopmentFactories


def _compliance_spec(path: str | Path) -> ComplianceSpec:
    # The finite contribution loader deliberately uses ValueError for input
    # refusal. Translate only at that parsing boundary, not around execution.
    try:
        return compliance.load_spec(path)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise SpecSchemaError(f"invalid compliance profile fixture: {exc}") from exc


COMMERCE_COMMANDS = DevelopmentCommands(
    DevelopmentFactories(
        "commerce.return_refund.profiles.v1",
        "0.1.0",
        "commerce.return_refund.profiles",
        commerce.load_spec,
        commerce.build_domain,
        commerce.build_agent,
        commerce.evaluate_episode,
        commerce.profile_identity,
        (Path(commerce.__file__).parent,),
    ),
    tuple(sorted(COMMERCE_AGENTS)),
    ("reference", "reference_alternative"),
    commerce.REFERENCE_SCENARIOS,
    commerce.NEGATIVE_CONTROLS,
)
COMPLIANCE_COMMANDS = DevelopmentCommands(
    DevelopmentFactories(
        "lettings.property_compliance.profiles.v1",
        "0.1.0",
        "lettings.property_compliance.profiles",
        _compliance_spec,
        compliance.build_domain,
        compliance.build_agent,
        compliance.evaluate_episode,
        compliance.profile_identity,
        (Path(compliance.__file__).parent,),
    ),
    tuple(sorted(COMPLIANCE_AGENTS)),
    ("reference", "alternate_delivery"),
    compliance.REFERENCE_SCENARIOS,
    compliance.NEGATIVE_CONTROLS,
)


class CommerceProfilesPack:
    metadata = OperationPackMetadata(
        pack_id="commerce.return_refund.profiles.v1",
        operation_type="commerce.return_refund.profiles",
        operation_id="commerce_return_refund_profiles",
        pack_version="0.1.0",
        display_name="Commerce return and refund profiles",
        owner_id="prospire",
        owner_display_name="PROSPIRE TECHNOLOGIES LTD",
        contribution_kind="maintainer",
        status="development",
        privacy_status="SYNTHETIC_ONLY",
        default_spec="examples/operatebench/commerce_return_refund_profiles/operation.yaml",
        agent_ids=tuple(sorted((*COMMERCE_AGENTS, "reference-mock"))),
        aliases=("commerce.return_refund.profiles",),
    )

    def validate(self, request: ValidateRequest) -> CommandResult:
        return COMMERCE_COMMANDS.validate(request)

    def run(self, request: RunRequest) -> CommandResult:
        return COMMERCE_COMMANDS.run(request)

    def replay(self, request: ReplayRequest) -> CommandResult:
        return COMMERCE_COMMANDS.replay(request)

    def check(self, request: CheckRequest) -> CommandResult:
        return COMMERCE_COMMANDS.check(request)


COMMERCE_PROFILES_PACK = CommerceProfilesPack()
