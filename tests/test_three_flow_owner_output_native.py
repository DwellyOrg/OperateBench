"""One real business prompt per catalog model, native SDK mock; no live calls."""

import copy
import gc
import os

import pytest

from tests.test_three_flow_context import diagnostic
from tests.test_three_flow_live import fixture
from tests.test_three_flow_optional_limits import UnlimitedNetwork, consent
from tests.test_three_flow_river import no_network
from tools.three_flow_budget import CampaignBudget
from tools.three_flow_live_factory import live_factory
from tools.three_flow_river_assets import catalog
from tools.three_flow_runtime import Trial, _execute, _identity, _load, provider_identity

__all__ = ["no_network"]


@pytest.mark.parametrize("model", [r["model"] for r in catalog()["models"]])
def test_owner_cap_first_business_wire(tmp_path, monkeypatch, model):
    if "THREE_FLOW_RIVER_ASSETS" not in os.environ:
        pytest.skip("explicit native asset cache required")
    import tools.three_flow_river_mock as mock
    from operatebench.agents.model import ModelAgent
    from operatebench.agents.playback import decision_record
    from tools import three_flow_live
    from tools.three_flow_admission import VARIABLES, digest

    monkeypatch.setattr(three_flow_live, "source_identity", lambda: "a" * 40)
    _, admission, slot = fixture(tmp_path, "river")
    p = diagnostic(admission["profiles"][slot["model"]])
    p.update(
        owner_output_cap_tokens=30000,
        network_timeout_seconds=None,
        execution_policy=consent()["mode"],
    )
    slot.update(model=model, flow="commerce", scenario="UK_NORMAL")
    admission.update(
        profiles={model: p},
        mode_profile="off-or-minimum-v1",
        cap_usd=None,
        limit_policy=consent(),
    )
    for name in ("frame", "synthetic_frame"):
        original = getattr(mock, name)

        def selected(*args, original=original):
            raw = original(*args)
            return raw if "GLM-5.3" in model else raw.removeprefix("</think>")

        monkeypatch.setattr(mock, name, selected)
    tmp_path.chmod(0o700)
    budget = CampaignBudget(
        tmp_path, campaign_id="offline", cap=None, limit_policy=consent(), create=True
    )
    network = UnlimitedNetwork()
    transport = live_factory(
        admission, {VARIABLES["river"]: "offline-dummy"}, network=network
    )(slot, budget)
    requests, decisions = [], []

    class FirstComplete(Exception):
        pass

    class FirstAgent(ModelAgent):
        def build_request(self, observation):
            request = transport.prepare_request(super().build_request(observation))
            self.max_output_tokens = request.max_output_tokens
            requests.append(request)
            return request

        def decide(self, observation):
            decisions.append(decision_record(super().decide(observation)))
            raise FirstComplete

    try:
        identity = provider_identity(transport)
        assert identity["settings"]["owner_output_cap_tokens"] == 30000
        assert identity["settings"]["dedicated_output_max"] is None
        assert identity["settings"]["model_contract_verified"] is False
        changed = copy.deepcopy(identity)
        changed["settings"]["owner_output_cap_tokens"] += 1
        assert digest(changed) != digest(identity)
        trial = Trial("offline", "first", "commerce", "UK_NORMAL", model, 30000, identity)
        factories, spec, _ = _load(trial)
        agent = FirstAgent(
            transport,
            model=model,
            agent_id="campaign-model",
            max_output_tokens=30000,
            max_transport_calls=None,
        )
        with pytest.raises(FirstComplete):
            _execute(factories, spec, trial, agent, _identity(spec, trial, "offline"))
        assert len(requests) == len(decisions) == 1
        assert decisions[0]["kind"].lower() in (
            "retrieve",
            "act",
            "wait",
            "ask",
            "complete",
            "escalate",
        )
        from tools.three_flow_river import sdk

        pb, _ = sdk()
        raw = [raw for path, raw in network.rpcs if path.endswith("InferenceGenerate")]
        assert len(raw) == 1
        wire = pb.InferenceGenerateRequest.FromString(raw[0])
        assert wire.base_model == model
        prompt = wire.prompts[0]
        assert prompt.max_tokens == requests[0].max_output_tokens == 30000
        assert len(prompt.input_ids) + prompt.max_tokens <= p["context_max"]
        rendered = transport.tokenizer.decode(list(prompt.input_ids))
        assert "observation" in rendered
        assert "Reasoning Effort: 75" not in rendered
        assert list(prompt.stop) == (["<|im_end|>"] if "Kimi" in model else [])
        assert (prompt.temperature, prompt.top_p, prompt.top_k) == (0, 1, -1)
        assert budget.totals()["unknown"] == budget.totals()["pending"] == 0
        assert budget.totals()["remaining"] is None
    finally:
        transport.close()
        budget.close()
        gc.collect()
