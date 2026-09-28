"""Shared SDK mechanics retain public aliases and complete fault semantics."""

import importlib

import httpx
import openai
import pytest

MODULES = (
    "operatebench.providers.openai_responses",
    "operatebench.providers.xai_responses",
    "operatebench.providers.xai_openai_compat",
    "boundarybench.providers.xai_openai_compat",
)


def test_sdk_classifier_has_one_owner():
    owners = [importlib.import_module(n).classify_exception for n in MODULES]
    assert len(set(owners)) == 1
    assert owners[0].__module__ == "operatebench.providers.openai_sdk"


@pytest.mark.parametrize("module", MODULES)
def test_sdk_classifier_preserves_response_received(module):
    classifier = importlib.import_module(module).classify_exception
    request = httpx.Request("POST", "https://offline.invalid")
    assert classifier(ValueError("internal bug")) is None
    timeout = classifier(openai.APITimeoutError(request=request))
    network = classifier(openai.APIConnectionError(request=request))
    invalid = classifier(openai.APIResponseValidationError(
        response=httpx.Response(200, request=request), body={}
    ))
    assert timeout.fault == "provider_timeout"
    assert network.fault == "provider_network_error"
    assert timeout.retryable and network.retryable
    assert invalid.fault == "provider_response_invalid"
    assert invalid.response_received and not invalid.retryable
