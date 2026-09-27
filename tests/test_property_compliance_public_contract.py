"""Core integration: contribution contract through real provider projection."""

import json

from operatebench.agents.anthropic_messages import build_model_payload
from operatebench.agents.model import ModelAgent
from operatebench.contributions.prospire.property_compliance_profiles.agents import (
    ComplianceAgent,
)
from operatebench.contributions.prospire.property_compliance_profiles.operation import (
    SCHEMAS,
)
from operatebench.core.outcomes import Act
from operatebench.core.protocol import AgentObservation, model_projection
from tests.contributions.prospire.property_compliance_profiles.test_pack import run_case


class NoTransport:
    def send(self, request):
        raise AssertionError("no provider calls allowed")


def provider_public(observation):
    # Reconstruct from JSON exactly as a provider-side observation consumer does.
    observation = AgentObservation(
        **json.loads(json.dumps(model_projection(observation)))
    )
    request = ModelAgent(NoTransport(), model="claude-sonnet-5").build_request(
        observation
    )
    body = json.loads(json.dumps(build_model_payload(request, model="claude-sonnet-5")))
    return json.loads(body["messages"][0]["content"])["observation"]


def test_exact_contract_reaches_provider_after_real_read():
    seen = []

    class Agent(ComplianceAgent):
        def decide(self, observation):
            public = provider_public(observation)
            if "required" in public["action_schemas"]["request_inspection"]:
                seen.append(public)
            return super().decide(AgentObservation(**public))

    _, grade = run_case("EN_normal", agent=Agent("reference"))
    assert grade["reliable"]
    assert seen
    for public in seen:
        for action, fields in SCHEMAS.items():
            schema = public["action_schemas"][action]
            assert schema["required"] == fields
            guidance = schema.get("field_guidance", {}).get("payload", "")
            assert "No additional fields" in guidance
            assert "all required" in guidance
            assert (", ".join(sorted(fields)) if fields else "{}") in guidance
            assert "non-empty strings" in guidance
        assert public["policy"]["benchmark_policy"]["public_contract_version"] == 2


def test_blocked_extra_field_recovery_keeps_effects_but_fails_safety():
    class Recover(ComplianceAgent):
        attempted = False

        def decide(self, observation):
            result = super().decide(AgentObservation(**provider_public(observation)))
            if isinstance(result, Act) and not self.attempted:
                self.attempted = True
                return Act(
                    result.action_type,
                    dict(result.payload) | {"property_id": "property-1"},
                    evidence_refs=result.evidence_refs,
                    rationale=result.rationale,
                )
            return result

    good, good_grade = run_case("EN_normal")
    recovered, grade = run_case("EN_normal", agent=Recover("reference"))
    assert good_grade["reliable"] and not grade["reliable"]
    assert recovered.final_state == good.final_state
    assert recovered.terminal_outcome == good.terminal_outcome
    assert "PAYLOAD" in {f["code"] for f in grade["findings"]}
    for episode in (good, recovered):
        assert sum(t["record_type"] == "action_rejected" for t in episode.trajectory) == (
            0 if episode is good else 1
        )
