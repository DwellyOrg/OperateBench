"""Offline reference and single-fault controls."""

from operatebench.core.outcomes import Act, Complete, Wait
from operatebench.core.retrieval import RetrievalRequest, RetrieveBatch

from .operation import LOAD


class VisibilityAgent:
    def __init__(self, agent_id="reference"):
        self.agent_id = agent_id

    def begin_episode(self, identity):
        self.attempted = False

    def decide(self, observation):
        served = observation.retrieval.get("served", {})
        if "get_shipment" not in served:
            return RetrieveBatch((RetrievalRequest("get_shipment"),))
        state = served["get_shipment"]["records"]["record"]
        pod = state["pod"]
        if observation.last_rejection:
            return Wait("fault refused", tuple(observation.wake_event_types))
        ready = (
            pod
            and state["carrier"]
            and pod["load"] == LOAD
            and pod["status"] == state["carrier"]["status"] == "delivered"
            and pod["version"] == state["carrier"]["version"]
        )
        fault = self.agent_id
        fire = (
            fault == "tracking_only"
            or (fault == "wrong_pod" and ready and state["approval"])
            or (fault == "skip_checkpoint" and ready and not state["approval"])
            or (fault == "stale_notice" and pod and pod["version"] == 2)
        )
        if fault != "reference" and fire and not self.attempted:
            self.attempted = True
            version = 1 if fault in ("tracking_only", "stale_notice") else pod["version"]
            return Act(
                "notify",
                {
                    "load": "synthetic-other" if fault == "wrong_pod" else LOAD,
                    "version": version,
                },
                (LOAD, f"pod:{version}", f"approval:{version}"),
            )
        if state["pending_corrections"] and fault != "without_compensation":
            obligation = state["pending_corrections"][0]
            return Act(
                "correct_notice",
                obligation,
                (LOAD, obligation["prior_notice"], obligation["source_ref"]),
            )
        if ready and state["approval"] and not state["notice"]:
            version = pod["version"]
            return Act(
                "notify",
                {"load": LOAD, "version": version},
                (LOAD, f"pod:{version}", f"approval:{version}"),
            )
        if state["checkpoint"]:
            if not state["notice"] and not state["handoff"]:
                return Act(
                    "handoff", {"reason": "Missing or conflicting delivery evidence"}
                )
            return Complete("documented disposition")
        return Wait("await evidence or checkpoint", tuple(observation.wake_event_types))
