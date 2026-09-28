"""Offline extraction closure; goldens captured on immutable 420b984."""

import dataclasses
import hashlib
import importlib
import inspect
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

from boundarybench.providers import xai_openai_compat as boundary
from operatebench.agents import openai_responses, xai_responses
from operatebench.providers import xai_openai_compat as kernel
from tests.test_i7_xai_differential_corpus import body, corpus, exercise


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True).encode()).hexdigest()


def result(call):
    try:
        value = call()
    except Exception as error:
        return [
            type(error).__module__,
            type(error).__qualname__,
            getattr(error, "fault", None),
            str(error),
        ]
    return dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value


def snapshot():
    from operatebench.agents import mistral_chat, xai_chat_completions
    from operatebench.agents.model import ModelAgent
    from tests.test_lifecycle_xai_chat_completions import observation

    captured = {}
    for module in (kernel, boundary):
        captured[f"{module.__name__}:shapes"] = {
            name: [shape.kind, sorted(shape.allowed), list(shape.required)]
            for name in sorted(dir(module))
            if name.endswith("_WIRE_SHAPE")
            for shape in (getattr(module, name),)
        }
    for module in (openai_responses, xai_responses):
        provider = importlib.import_module(module.__name__.replace("agents", "providers"))
        for model in provider.MODEL_REQUEST_PROFILES:
            request = ModelAgent(None, model=model, agent_id="diagnostic").build_request(
                observation()
            )
            if module is xai_responses:
                request = dataclasses.replace(
                    request,
                    max_output_tokens=module.lifecycle_output_tokens(
                        model=model, contract=module._request_contract(model)
                    ),
                )
            payload = module.build_model_payload(request, model=model)
            captured[f"{module.__name__}:{model}"] = digest(payload)
    captured["function_definitions"] = digest(mistral_chat.mistral_outcome_tools())
    assert (
        mistral_chat.mistral_outcome_tools() == xai_chat_completions.xai_outcome_tools()
    )
    for module in (kernel, boundary):
        outcomes = []
        for name, raw, typed in corpus():
            for path in ("wire", "agreement", "admissible"):
                outcomes.append(
                    [name, path, result(partial(exercise, module, path, raw, typed))]
                )
        for value in (None, True, -1, 0, 1, 9007199254740991, 9007199254740992, "1", 1.0):
            for field in ("prompt_tokens", "completion_tokens"):
                raw = body()
                raw["usage"][field] = value
                outcomes.append(
                    [
                        field,
                        value,
                        result(partial(module.response_usage, SimpleNamespace(wire=raw))),
                    ]
                )
        for model in ("grok-4.5", "other", None):
            outcomes.append(
                [
                    "model",
                    model,
                    result(
                        partial(
                            module.check_response_model,
                            SimpleNamespace(model=model),
                            model="grok-4.5",
                        )
                    ),
                ]
            )
        captured[module.__name__] = digest(outcomes)
    captured["boundary_settings"] = digest(boundary.xai_settings(model="grok-4.5"))
    captured["lifecycle_settings"] = digest(
        xai_chat_completions.lifecycle_xai_settings(
            model="grok-4.5",
            deadline_seconds=60.0,
            profile=kernel.request_profile_for("grok-4.5"),
        )
    )
    return captured


def test_baseline_projection_and_exact_fault_digests():
    expected = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("a34_projection_digests.json")
        .read_text()
    )
    assert snapshot() == expected


def test_responses_have_one_pure_projection_owner():
    names = (
        "_response_invalid",
        "_response_id_digest",
        "_bounded_text",
        "_message_text",
        "_tool_call_from",
    )
    for name in names:
        assert getattr(openai_responses, name) is getattr(xai_responses, name)
        assert (
            getattr(openai_responses, name).__module__
            == "operatebench.agents.responses_projection"
        )


def test_lifecycle_function_definitions_have_one_owner():
    from operatebench.agents import lifecycle_contract, mistral_chat, xai_chat_completions

    assert hasattr(lifecycle_contract, "outcome_function_definitions")
    assert (
        mistral_chat.mistral_outcome_tools
        is lifecycle_contract.outcome_function_definitions
    )
    assert (
        xai_chat_completions.xai_outcome_tools
        is lifecycle_contract.outcome_function_definitions
    )


@pytest.mark.parametrize("module", [kernel, boundary])
def test_wire_wrapper_requires_and_binds_track_contract(monkeypatch, module):
    assert hasattr(kernel, "check_wire_completion_under_contract")
    helper = kernel.check_wire_completion_under_contract
    parameter = inspect.signature(helper).parameters["contract"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="contract"):
        helper({})
    calls = []
    monkeypatch.setattr(
        module,
        "check_wire_completion_under_contract",
        lambda raw, *, contract: calls.append((raw, contract)),
    )
    raw = {}
    module.check_wire_completion(raw)
    assert calls == [(raw, module.RESPONSE_EXTENSION_CONTRACT)]
    assert calls[0][1] is module.RESPONSE_EXTENSION_CONTRACT


def test_wire_shapes_model_and_usage_have_one_owner():
    for name in dir(kernel):
        if name.endswith("_WIRE_SHAPE"):
            assert getattr(kernel, name) is getattr(boundary, name)
    assert kernel.check_response_model is boundary.check_response_model
    assert kernel.response_usage is boundary.response_usage
    assert kernel._check_client_state is not boundary._check_client_state
    assert kernel.XAIConfigurationError is not boundary.XAIConfigurationError
