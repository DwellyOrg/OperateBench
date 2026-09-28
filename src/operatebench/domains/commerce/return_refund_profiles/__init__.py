"""Research-grounded, synthetic, non-evidence Commerce development process."""

from .agents import build_agent
from .evaluator import evaluate_episode
from .operation import build_domain
from .oracle import NEGATIVE_CONTROLS
from .spec import OPERATION_TYPE, PACK_ID, VERSION, Spec, load_spec, profile_identity

__all__ = [
    "NEGATIVE_CONTROLS",
    "OPERATION_TYPE",
    "PACK_ID",
    "REFERENCE_SCENARIOS",
    "VERSION",
    "Spec",
    "build_agent",
    "build_domain",
    "evaluate_episode",
    "load_spec",
    "profile_identity",
]

REFERENCE_SCENARIOS = (
    "UK_NORMAL",
    "UK_LABEL_UNKNOWN",
    "UK_MISSING_DISCLOSURE",
    "UK_PAYMENT_FAILED",
    "DE_NORMAL",
    "DE_WITHHOLDING",
    "DE_MISSING_DISCLOSURE",
    "DE_FAULT_REVIEW",
    "CA_POLICY",
    "CA_NO_POLICY",
    "CA_UNKNOWN_DISPLAY",
    "CA_FAULT_REVIEW",
    "AU_MAJOR",
    "AU_MINOR",
    "AU_CHANGE_OF_MIND",
    "AU_UNCERTAIN_FAULT",
)
