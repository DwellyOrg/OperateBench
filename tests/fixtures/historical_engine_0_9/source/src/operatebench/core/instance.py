"""The identity a run gives the agent instead of the identity of the answer.

An episode has two identities and they are not the same thing. ``scenario_id``
is the index into the authored expectation: it selects the case, and with it the
semantic scenario id, the expected terminal, the oracle control and the whole
authored event list. Handing that to the agent is handing it the answer's name,
and an agent that reads it can be tuned per case without ever running the
operation. It stays in the artefact, the evaluator and the run identity, where a
grader reads it.

What the agent gets is the *operation instance* identity: this run, and nothing
about which run it is. It has to be three things at once, and each one rules out
an easier construction.

**Opaque.** Nothing in it is derived from the scenario, the semantic scenario
id, the operation id or the spec digest, so it cannot be inverted into the case
being run. That is why it is random rather than hashed from run inputs: a digest
of the inputs is a stable label per case, which is the leak wearing a hat — the
same case would carry the same identity in every run and a lookup table would
rebuild the mapping in one pass.

**Stable across one run.** It is minted once and reused by the determinism
self-check and by recorded playback, because a decision is bound to the digest
of the observation that produced it and the identity is inside that observation.
An identity minted per *execution* would move every digest and make a run fail
to reproduce itself.

**Well-formed.** The prefix and the fixed-width hexadecimal body are checked at
the artefact boundary, so a record carrying something else is a named refusal
rather than a value read back as an identity.
"""

from __future__ import annotations

import secrets

from operatebench.core.errors import OperateBenchError

#: What every operation instance identity starts with. A label rather than a
#: bare hex string so a reader meeting one in a log knows what it is, and so a
#: digest pasted into the field is refused instead of being read as an identity.
OPERATION_INSTANCE_ID_PREFIX = "opinst_"

#: How much entropy the identity carries. 128 bits: enough that two runs
#: colliding is not a thing anyone has to reason about, and short enough to read.
OPERATION_INSTANCE_ID_ENTROPY_BYTES = 16

#: The exact length of the hexadecimal body, derived from the entropy above so
#: the two cannot drift apart.
OPERATION_INSTANCE_ID_BODY_LENGTH = OPERATION_INSTANCE_ID_ENTROPY_BYTES * 2

_HEX = frozenset("0123456789abcdef")


class OperationInstanceError(OperateBenchError):
    """A value is not an operation instance identity this build writes."""


def new_operation_instance_id() -> str:
    """Mint one. Random, non-semantic, and never derived from the case."""
    body = secrets.token_hex(OPERATION_INSTANCE_ID_ENTROPY_BYTES)
    return f"{OPERATION_INSTANCE_ID_PREFIX}{body}"


def operation_instance_id_problem(value: object) -> str | None:
    """Why ``value`` is not an operation instance identity, or ``None``."""
    if not isinstance(value, str):
        return (
            f"is a {type(value).__name__}, not the text of an operation instance identity"
        )
    if not value.startswith(OPERATION_INSTANCE_ID_PREFIX):
        return (
            f"does not start with {OPERATION_INSTANCE_ID_PREFIX!r}; an operation "
            "instance identity is labelled so it cannot be confused with a digest "
            "or with the identity of the case being run"
        )
    body = value[len(OPERATION_INSTANCE_ID_PREFIX) :]
    if len(body) != OPERATION_INSTANCE_ID_BODY_LENGTH:
        return (
            f"carries a {len(body)}-character body against the "
            f"{OPERATION_INSTANCE_ID_BODY_LENGTH} this build writes"
        )
    if not set(body) <= _HEX:
        return "carries a body that is not lower-case hexadecimal"
    return None


def require_operation_instance_id(value: object, where: str) -> str:
    """The identity, or a named refusal. The one place callers convert."""
    problem = operation_instance_id_problem(value)
    if problem is not None:
        raise OperationInstanceError(f"{where}: {problem}")
    assert isinstance(value, str)
    return value


__all__ = [
    "OPERATION_INSTANCE_ID_BODY_LENGTH",
    "OPERATION_INSTANCE_ID_ENTROPY_BYTES",
    "OPERATION_INSTANCE_ID_PREFIX",
    "OperationInstanceError",
    "new_operation_instance_id",
    "operation_instance_id_problem",
    "require_operation_instance_id",
]
