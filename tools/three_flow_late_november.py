"""Candidate contracts for the late-November roster, not dispatch authority.

Rates are USD per million tokens. Higher Meta rates form the conservative
reservation envelope regardless of prompt size. Accounting is not an invoice.
"""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class Candidate:
    provider: str
    model: str
    route: str
    context: int
    output: int
    reasoning: str
    input_rate: str
    cached_rate: str
    output_rate: str
    standard_only: bool = True


CANDIDATES = MappingProxyType(
    {
        "gemini-4-hydrogen": Candidate(
            "gemini",
            "gemini-4-hydrogen",
            "generate_content",
            1048576,
            65536,
            "low",
            "1.5",
            "0.075",
            "7.5",
        ),
        "gemini-4-neon": Candidate(
            "gemini",
            "gemini-4-neon",
            "generate_content",
            1048576,
            65536,
            "low",
            "3",
            "0.15",
            "15",
        ),
        "muse-ember-1.0": Candidate(
            "meta",
            "muse-ember-1.0",
            "responses",
            1048576,
            524288,
            "minimal",
            "0.30",
            "0.012",
            "1.50",
        ),
        "moonshotai/Kimi-K3.1-Pro": Candidate(
            "river_http",
            "moonshotai/Kimi-K3.1-Pro",
            "chat_completions",
            1048576,
            30000,
            "none",
            "2",
            "0.2",
            "10",
        ),
        "ricursiveAI/Ripple-65B-Preview-NVFP4": Candidate(
            "river",
            "ricursiveAI/Ripple-65B-Preview-NVFP4",
            "grpc_inference_generate",
            262144,
            30000,
            "provider-default",
            "1",
            "0.2",
            "3",
        ),
    }
)

# Meta's lower tier applies at <=200K input tokens. The envelope above covers
# both tiers; cache is never assumed in an upper-bound reservation.
META_SHORT_TIER = MappingProxyType(
    {"input": "0.10", "cached": "0.004", "output": "0.50", "input_tokens_at_most": 200000}
)
