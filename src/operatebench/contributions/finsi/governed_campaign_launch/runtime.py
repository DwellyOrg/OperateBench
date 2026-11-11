"""Offline pack-owned run format and exact live-Core replay; no providers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from operatebench.core.engine import Engine
from operatebench.core.outcomes import Act, Complete, Wait
from operatebench.core.protocol import model_projection
from operatebench.core.retrieval import RetrievalRequest, RetrieveBatch

from .agents import build_agent
from .evaluator import evaluate_episode
from .operation import build_domain
from .spec import digest, load_spec

FORMAT = "finsi.governed-campaign.development.v1"


class TapeAgent:
    def __init__(self, agent_id: str, tape: Any = None):
        self.agent_id = agent_id
        self.playback = tape is not None
        self.rows = [] if tape is None else tape
        self.cursor = 0
        self.delegate = None if self.playback else build_agent(agent_id)

    def begin_episode(self, identity: Any) -> None:
        self.cursor = 0
        if self.delegate is not None:
            self.delegate.begin_episode(identity)

    def decide(self, observation: Any) -> Any:
        obs = digest(model_projection(observation))
        if self.playback:
            if (
                self.cursor >= len(self.rows)
                or self.rows[self.cursor]["observation"] != obs
            ):
                raise ValueError("decision observation mismatch")
            p = self.rows[self.cursor]["decision"]
            self.cursor += 1
            if p["kind"] == "ACT":
                return Act(p["action_type"], p["payload"], p["evidence_refs"])
            if p["kind"] == "WAIT":
                return Wait(p["reason"], p["wake_on"], p["fallback_after_minutes"])
            if p["kind"] == "COMPLETE":
                return Complete(p["reason"], p["evidence_refs"])
            if p["kind"] == "RETRIEVE":
                return RetrieveBatch(
                    tuple(
                        RetrievalRequest(x["tool"], x["arguments"]) for x in p["requests"]
                    )
                )
            raise ValueError("unknown tape outcome")
        assert self.delegate is not None
        decision = self.delegate.decide(observation)
        if isinstance(decision, Act):
            p = {
                "kind": "ACT",
                "action_type": decision.action_type,
                "payload": dict(decision.payload),
                "evidence_refs": list(decision.evidence_refs),
            }
        elif isinstance(decision, Wait):
            p = {
                "kind": "WAIT",
                "reason": decision.reason,
                "wake_on": list(decision.wake_on),
                "fallback_after_minutes": decision.fallback_after_minutes,
            }
        elif isinstance(decision, Complete):
            p = {
                "kind": "COMPLETE",
                "reason": decision.reason,
                "evidence_refs": list(decision.evidence_refs),
            }
        else:
            p = {
                "kind": "RETRIEVE",
                "requests": [
                    {"tool": r.tool, "arguments": dict(r.arguments)}
                    for r in decision.requests
                ],
            }
        self.rows.append({"observation": obs, "decision": p})
        return decision


def execute(spec: Any, scenario: str, agent: Any) -> Any:
    return Engine(
        build_domain(spec, scenario),
        agent,
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": "finsi-synthetic-development-instance",
            "agent_id": agent.agent_id,
        },
    ).run()


def run(path: str | Path, scenario: str, agent_id: str = "reference") -> dict[str, Any]:
    spec = load_spec(path)
    agent = TapeAgent(agent_id)
    episode = execute(spec, scenario, agent)
    record = {
        "format": FORMAT,
        "scope": "SYNTHETIC_ONLY_DEVELOPMENT_NON_EVIDENCE",
        "spec_digest": spec.content_digest,
        "scenario": scenario,
        "agent": agent_id,
        "decisions": agent.rows,
        "episode": episode.as_dict(),
        "evaluation": evaluate_episode(episode, spec, scenario),
    }
    replay(path, record)
    return record


def replay(path: str | Path, record: Any) -> dict[str, Any]:
    if (
        set(record)
        != {
            "format",
            "scope",
            "spec_digest",
            "scenario",
            "agent",
            "decisions",
            "episode",
            "evaluation",
        }
        or record["format"] != FORMAT
        or record["scope"] != "SYNTHETIC_ONLY_DEVELOPMENT_NON_EVIDENCE"
    ):
        raise ValueError("wrong development record")
    spec = load_spec(path)
    if spec.content_digest != record["spec_digest"]:
        raise ValueError("spec binding mismatch")
    agent = TapeAgent(record["agent"], record["decisions"])
    episode = execute(spec, record["scenario"], agent)
    if (
        agent.cursor != len(agent.rows)
        or episode.as_dict() != record["episode"]
        or evaluate_episode(episode, spec, record["scenario"]) != record["evaluation"]
    ):
        raise ValueError("live replay mismatch")
    return {"consistent": True, "decisions_consumed": agent.cursor, "provider_calls": 0}


def read_record(path: str | Path) -> Any:
    raw = Path(path).read_bytes()
    if len(raw) > 20000000:
        raise ValueError("oversize run")
    return json.loads(raw)
