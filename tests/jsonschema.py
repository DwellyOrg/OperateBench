"""A JSON Schema validator for the keywords this build's tool surface states.

**What this is evidence of.** One question, asked locally: does a schema this
build generates, read under JSON Schema's own semantics, admit exactly the
values the Core outcome contract admits? That is a comparison between two
artefacts in this repository — a schema and a parser — and it is the only claim
any test using this module is entitled to make.

**What it is not evidence of.** It is not a claim that a provider enforces these
schemas. The Lifecycle outcome tools are sent as **non-strict** function tools,
because Core's semantics are not expressible in OpenAI's strict subset (see
``tests/test_lifecycle_bridge_blockers.py``), and a non-strict schema is
guidance to the model rather than a constraint any server applies. Nothing
downstream of the wire may assume it was applied: this build's fail-closed
parser, :func:`~operatebench.agents.model.parse_tool_call`, and Core's own
:func:`~operatebench.core.outcomes.outcome_contract_problem` are what decide
whether an answer is a decision, and they are asked directly beside every
assertion made with this module.

The keyword set is deliberately small — exactly the keywords the projection
emits. A validator that understood keywords this build never states would be
proving something about a library rather than about these schemas, and a
validator that silently ignored one it *does* state would be certifying a
constraint nobody checked; so an unknown keyword raises rather than passing.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

#: The JSON Schema keywords this validator reads, which are exactly the ones
#: this build's tool surface states.
VALIDATED_KEYWORDS: frozenset[str] = frozenset(
    {
        "type",
        "description",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "anyOf",
        "pattern",
        "minimum",
        "minItems",
    }
)


def _is_type(value: Any, name: str) -> bool:
    if name == "null":
        return value is None
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "string":
        return isinstance(value, str)
    if name == "array":
        return isinstance(value, list)
    if name == "object":
        return isinstance(value, Mapping)
    raise AssertionError(f"{name!r} is not a JSON type this validator knows")


def schema_problem(schema: Mapping[str, Any], value: Any, path: str = "$") -> str | None:
    """Why ``value`` does not satisfy ``schema``, or ``None`` if it does.

    ``anyOf`` is read as JSON Schema defines it — an additional constraint, not
    a replacement for the sibling keywords — which is what makes a root object
    carrying both a closed property set *and* a liveness ``anyOf`` mean the
    conjunction of the two.

    A property that is not stated is not checked, which is JSON Schema's own
    rule and the one that makes an omitted optional field legal. Absence is
    ``required``'s question and only ``required``'s.
    """
    unknown = sorted(set(schema) - VALIDATED_KEYWORDS)
    if unknown:
        raise AssertionError(
            f"{path} states keyword(s) this validator ignores: {unknown}"
        )
    branches = schema.get("anyOf")
    if branches is not None and all(
        schema_problem(branch, value, path) is not None for branch in branches
    ):
        return f"{path} satisfies none of the {len(branches)} anyOf branch(es)"
    declared = schema.get("type")
    if declared is not None:
        names = [declared] if isinstance(declared, str) else list(declared)
        if not any(_is_type(value, name) for name in names):
            return f"{path} is not any of {names}"
    if (
        isinstance(value, str)
        and "pattern" in schema
        and re.search(schema["pattern"], value) is None
    ):
        return f"{path} does not match the declared pattern"
    if _is_type(value, "number") and "minimum" in schema and value < schema["minimum"]:
        return f"{path} is below the declared minimum"
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            return f"{path} states fewer than {schema['minItems']} item(s)"
        items = schema.get("items")
        if items is not None:
            for index, item in enumerate(value):
                problem = schema_problem(items, item, f"{path}[{index}]")
                if problem is not None:
                    return problem
    if isinstance(value, Mapping):
        properties = schema.get("properties")
        if properties is not None:
            for name in schema.get("required", ()):
                if name not in value:
                    return f"{path}.{name} is required and was not stated"
            if schema.get("additionalProperties") is False:
                for name in value:
                    if name not in properties:
                        return f"{path}.{name} is not a property this schema declares"
            for name, child in properties.items():
                if name in value:
                    problem = schema_problem(child, value[name], f"{path}.{name}")
                    if problem is not None:
                        return problem
    return None


__all__ = ["VALIDATED_KEYWORDS", "schema_problem"]
