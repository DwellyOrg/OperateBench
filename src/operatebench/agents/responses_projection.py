"""Pure Responses projections shared by the Lifecycle API-specific transports.

Full response conversion and item/stop policies remain in each API adapter.
"""

from __future__ import annotations

from typing import Any

from openai.types.responses import Response

from operatebench.agents.lifecycle_contract import (
    MAX_TRANSIENT_TEXT_CHARACTERS,
    ToolArgumentsUndecodable,
    UndecodableArguments,
    elide_null_optionals,
)
from operatebench.agents.transport import ToolCall
from operatebench.providers.faults import (
    PROVIDER_FAULT_RESPONSE_INVALID,
    AdapterProviderError,
)
from operatebench.providers.toolcalls import decode_tool_arguments
from operatebench.providers.wire import WireResponse


def _response_invalid(detail: str) -> AdapterProviderError:
    """A body this build cannot read as an answer at all, as a provider fault.

    Not a classification. Every :class:`~operatebench.agents.model.
    MalformedModelOutcome` is a statement about how the *model* behaved, and a
    response that stopped for a reason this build does not classify or carries
    an output item this scaffold never asked for is a statement about the
    service — so filing it under the model would put a service-side fault in the
    bucket that is supposed to mean "the model answered badly". It reaches
    ``ModelAgent`` as an execution fault, which excludes the run with its
    attempts intact rather than completing it with a decision nobody made.
    """
    return AdapterProviderError(PROVIDER_FAULT_RESPONSE_INVALID, detail)


def _response_id_digest(wire: WireResponse[Response] | None) -> str | None:
    """The digest of the provider's own response identifier, or ``None``.

    A digest rather than the identifier, and for the reason every other
    provider-controlled string in this build is treated the same way: it is a
    value nobody here reviewed. The digest is enough to say "these two rows are
    the same answer" and "this row is not that answer", which is what durable
    evidence needs it for.

    The body has already been parsed by the time this runs, so reading it costs
    nothing; a turn that never got one, or whose answer this build refused,
    states ``None`` rather than an invented value.
    """
    return None if wire is None else wire.response_id_digest_sha256


def _bounded_text(parts: list[str]) -> str:
    """The prose beside a tool call, bounded and never read for a decision."""
    return "\n".join(parts)[:MAX_TRANSIENT_TEXT_CHARACTERS]


def _message_text(item: Any) -> list[str]:
    """Whatever text one assistant message states, and nothing else.

    Read defensively and typed as ``Any`` on purpose. This SDK's output item is
    a union of some thirty models and only a few of them carry ``content`` at
    all, so a reader that named one of them would be asserting which member the
    SDK resolved rather than reading what arrived. Nothing here is a decision:
    the value only ever reaches
    :attr:`~operatebench.agents.transport.ModelResponse.text`.
    """
    content = getattr(item, "content", None)
    if not isinstance(content, list):
        return []
    return [part.text for part in content if isinstance(getattr(part, "text", None), str)]


def _tool_call_from(item: Any) -> ToolCall:
    """One ``function_call`` output item as one structured outcome.

    The arguments arrive as a JSON *string* on this API, so they have more ways
    to be wrong than a decoded object does — not JSON at all, JSON that is not an
    object, an object naming a key twice, a value canonical JSON cannot
    round-trip. Every one of them is the model failing the protocol rather than
    the transport failing, so the shared decoder's refusal becomes
    :class:`UndecodableArguments` and the existing parser classifies the call as
    ``MODEL_MALFORMED_ARGUMENTS``. Nothing the model wrote is quoted.
    """
    name = item.name
    try:
        decoded: Any = decode_tool_arguments(
            item.arguments,
            label="lifecycle tool call arguments",
            error=ToolArgumentsUndecodable,
        )
    except ToolArgumentsUndecodable:
        return ToolCall(name, UndecodableArguments())
    return ToolCall(name, elide_null_optionals(name, decoded))
