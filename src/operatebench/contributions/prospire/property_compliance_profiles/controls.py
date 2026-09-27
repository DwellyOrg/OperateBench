"""Independently authored controls. Expected rejected proposal then recovery.

Each dict: agent_id, scenario_id, failed_dimensions, finding_codes; must_pass is
all remaining dimensions. These are not generated from reference executions.
"""

from .evaluator import DIMENSIONS

_ROWS = (
    ("delay_remedy", "EN_remediation", "temporal", "LATE_REMEDIATION"),
    ("skip_reads", "EN_normal", "retrieval", "ACTION_WITHOUT_RETRIEVAL"),
    ("premature_complete", "EN_normal", "scope", "REPORTS_MISSING"),
    ("trust_bad_report", "EN_wrong_property", "scope", "PROPERTY_SCOPE"),
    ("trust_bad_report", "SC_qualification", "qualification", "QUALIFICATION"),
    ("trust_bad_report", "AU_qualification", "qualification", "QUALIFICATION"),
    ("trust_bad_report", "AU_gas_expired", "temporal", "EXPIRY"),
    ("trust_bad_report", "NZ_assessor_only", "qualification", "SIGNATURE"),
    ("trust_repair_claim", "EN_remediation", "verification", "VERIFICATION"),
    ("wrong_recipient", "NZ_normal", "delivery", "RECIPIENT"),
    ("skip_council", "EN_remediation", "delivery", "DELIVERY"),
    ("stale_version", "EN_reopen", "versioning", "VERSION"),
    ("force_access", "SC_access", "access", "ACCESS"),
    ("unnecessary_review", "AU_normal", "decision_validity", "UNNECESSARY_REVIEW"),
)
NEGATIVE_CONTROLS = tuple(
    {
        "agent_id": agent,
        "scenario_id": scenario,
        "failed_dimensions": (dimension,),
        "finding_codes": (code,),
        "must_pass": tuple(d for d in DIMENSIONS if d != dimension),
    }
    for agent, scenario, dimension, code in _ROWS
)
