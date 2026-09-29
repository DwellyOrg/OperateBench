"""Fault classification shared by integrations using the OpenAI SDK."""

import openai

from operatebench.providers.faults import (
    PROVIDER_FAULT_NETWORK_ERROR,
    PROVIDER_FAULT_RESPONSE_INVALID,
    PROVIDER_FAULT_TIMEOUT,
    Fault,
    classify_http_status,
)


def classify_openai_sdk_exception(exception: Exception) -> Fault | None:
    """Map one SDK error onto the contract's fault set, or decline.

    ``None`` means "not a provider fault": something inside this integration
    broke, and the runner should record it as the adapter failure it is rather
    than have it dressed up as an outage.
    """
    # Checked before its base class: an SDK timeout *is* an
    # ``APIConnectionError``, and the two are worth telling apart.
    if isinstance(exception, openai.APITimeoutError):
        return Fault(PROVIDER_FAULT_TIMEOUT, True, None)
    if isinstance(exception, openai.APIConnectionError):
        return Fault(PROVIDER_FAULT_NETWORK_ERROR, True, None)
    if isinstance(exception, openai.APIResponseValidationError):
        # A whole body arrived and the SDK refused to read it. Recorded as an
        # attempt that *received a response*, because it did: saying otherwise
        # would report a transport failure that did not happen and would hide
        # the one case where the provider answered with something unreadable.
        return Fault(PROVIDER_FAULT_RESPONSE_INVALID, False, None, response_received=True)
    if isinstance(exception, openai.APIStatusError):
        return classify_http_status(exception.status_code)
    return None
