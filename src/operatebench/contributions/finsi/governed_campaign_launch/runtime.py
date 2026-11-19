"""Offline pack-owned run format and exact live-Core replay; no providers."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from operatebench.core.engine import Engine
from operatebench.core.outcomes import Act, Complete, Wait
from operatebench.core.protocol import model_projection
from operatebench.core.retrieval import RetrievalRequest, RetrieveBatch

from .agents import AGENT_IDS, build_agent
from .evaluator import evaluate_episode
from .input import InputError, read_json
from .operation import START, build_domain
from .spec import REFERENCE_SCENARIOS, digest, load_spec

FORMAT = "finsi.governed-campaign.development.v1"


def validate_selection(scenario: Any, agent: Any) -> None:
    if not isinstance(scenario, str) or scenario not in REFERENCE_SCENARIOS:
        raise InputError("unknown scenario")
    if not isinstance(agent, str) or agent not in AGENT_IDS:
        raise InputError("unknown agent")


def is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def decision_from_input(p: Any) -> Any:
    if not isinstance(p, dict) or not isinstance(p.get("kind"), str):
        raise InputError("malformed decision")
    fields = {
        "ACT": {"kind", "action_type", "payload", "evidence_refs"},
        "WAIT": {"kind", "reason", "wake_on", "fallback_after_minutes"},
        "COMPLETE": {"kind", "reason", "evidence_refs"},
        "RETRIEVE": {"kind", "requests"},
    }
    kind = p["kind"]
    if kind not in fields or set(p) != fields[kind]:
        raise InputError("malformed decision variant")
    if kind == "RETRIEVE":
        if not isinstance(p["requests"], list):
            raise InputError("malformed retrieval requests")
        requests = []
        for r in p["requests"]:
            if (
                not isinstance(r, dict)
                or set(r) != {"tool", "arguments"}
                or not isinstance(r["tool"], str)
                or not r["tool"]
                or not isinstance(r["arguments"], dict)
                or any(not isinstance(k, str) or not k for k in r["arguments"])
            ):
                raise InputError("malformed retrieval request")
            requests.append(RetrievalRequest(r["tool"], r["arguments"]))
        return RetrieveBatch(tuple(requests))
    outcome: Any
    if kind == "WAIT":
        deadline = p["fallback_after_minutes"]
        if (
            not isinstance(p["wake_on"], list)
            or (deadline is not None and (type(deadline) is not int or deadline <= 0))
            or (not p["wake_on"] and deadline is None)
        ):
            raise InputError("malformed WAIT bound")
        validate_wait_time(START, deadline)
        outcome = Wait(p["reason"], p["wake_on"], deadline)
    elif kind == "ACT":
        if not isinstance(p["action_type"], str) or not p["action_type"]:
            raise InputError("malformed action type")
        outcome = Act(p["action_type"], p["payload"], p["evidence_refs"])
    else:
        outcome = Complete(p["reason"], p["evidence_refs"])
    refs = p["wake_on"] if kind == "WAIT" else p["evidence_refs"]
    if (
        not isinstance(refs, list)
        or any(not isinstance(ref, str) or not ref for ref in refs)
        or (kind == "ACT" and not isinstance(p["payload"], dict))
        or (kind != "ACT" and not isinstance(p["reason"], str))
    ):
        raise InputError("malformed outcome fields")
    return outcome


def validate_wait_time(now: str, minutes: int | None) -> None:
    if minutes is not None:
        try:
            datetime.fromisoformat(now.replace("Z", "+00:00")) + timedelta(
                minutes=minutes
            )
        except OverflowError as exc:
            raise InputError("WAIT fallback exceeds timestamp range") from exc


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
                raise InputError("decision observation mismatch")
            p = self.rows[self.cursor]["decision"]
            self.cursor += 1
            decision = decision_from_input(p)
            if isinstance(decision, Wait):
                validate_wait_time(
                    model_projection(observation)["now"], decision.fallback_after_minutes
                )
            return decision
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
    validate_selection(scenario, agent_id)
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
        not isinstance(record, dict)
        or set(record)
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
        raise InputError("wrong development record")
    validate_selection(record["scenario"], record["agent"])
    if (
        not is_digest(record["spec_digest"])
        or not isinstance(record["episode"], dict)
        or not isinstance(record["evaluation"], dict)
        or not isinstance(record["decisions"], list)
    ):
        raise InputError("malformed development envelope")
    from .evaluator import GRADING_VERSION

    if record["evaluation"].get("grading_version") != GRADING_VERSION:
        raise InputError("grading version mismatch")
    for row in record["decisions"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"observation", "decision"}
            or not is_digest(row["observation"])
        ):
            raise InputError("malformed decision row")
        decision_from_input(row["decision"])
    spec = load_spec(path)
    if spec.content_digest != record["spec_digest"]:
        raise InputError("spec binding mismatch")
    agent = TapeAgent(record["agent"], record["decisions"])
    episode = execute(spec, record["scenario"], agent)
    if (
        agent.cursor != len(agent.rows)
        or episode.as_dict() != record["episode"]
        or evaluate_episode(episode, spec, record["scenario"]) != record["evaluation"]
    ):
        raise InputError("live replay mismatch")
    return {"consistent": True, "decisions_consumed": agent.cursor, "provider_calls": 0}


def read_record(path: str | Path) -> Any:
    return read_json(path, 20000000)
