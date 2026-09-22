"""One offered interface, three vendors: the affordances must be the same one.

``tests/test_action_surface.py`` and ``tests/test_fact_affordances.py`` hold the
Anthropic integration to two properties that are about the *benchmark* rather
than about a vendor: the tools a request declares are exactly the actions the
:class:`~boundarybench.adapter.TurnRequest` offers, and a fact-acquisition
argument is a closed enum of the keys the Cube can actually deliver rather than
an open string graded against a set the model never saw. Those two properties
were proven for one integration because there was one.

With four, proving them once is not enough, and not because the projection code
is untrusted: a divergence here would be silent and would look exactly like a
result. A model offered a tool the environment does not accept, or an open
string where another model got an enum, is answering a different question from
the other four — and the matrix's whole purpose is to compare *methodology*
across vendors, which requires the question to be identical.

So each of the three new adapters is driven through one real turn, its request
serialised by its own installed SDK over an in-process transport, and the wire
body is checked against the neutral request the runner built. Nothing here
contacts a provider, and none of it is evidence about a model.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from boundarybench.adapter import TurnDeadline, TurnRequest, build_turn_request
from boundarybench.compiler import Variant, compile_cube
from boundarybench.environment import Environment
from boundarybench.providers.mistral_chat import MistralChatAdapter
from boundarybench.providers.openai_responses import OpenAIResponsesAdapter
from boundarybench.providers.xai_openai_compat import XAIOpenAICompatAdapter
from boundarybench.scaffold import STANDARD_SCAFFOLD, Scaffold, load_scaffold
from boundarybench.schema import ConstructCard
from tests import mistral_transport, openai_transport
from tests.conftest import minimal_card

#: The action whose argument is a closed set, and the argument itself.
FACT_ACTION = "ask_user"
FACT_PARAMETER = "fact"


def _scaffold() -> Scaffold:
    return load_scaffold(STANDARD_SCAFFOLD)


def _variant() -> Variant:
    return compile_cube(ConstructCard.from_dict(minimal_card())).variants[0]


def _turn_request() -> TurnRequest:
    return build_turn_request(
        scaffold=_scaffold(), environment=Environment(_variant()), turns_remaining=12
    )


def _deadline() -> TurnDeadline:
    return TurnDeadline(remaining_seconds=30.0, cancelled=lambda: False)


def _openai_turn(request: TurnRequest) -> dict[str, Any]:
    model = "gpt-test-20990101"
    transport, client = openai_transport.scripted_client(
        openai_transport.responses_body(
            [openai_transport.function_call_item("read_records", "{}")], model=model
        )
    )
    OpenAIResponsesAdapter(model=model, client=client).next_call(request, _deadline())
    return transport.bodies[0]


def _xai_turn(request: TurnRequest) -> dict[str, Any]:
    model = "grok-test-20990101"
    transport, client = openai_transport.xai_scripted_client(
        openai_transport.chat_body(
            [openai_transport.tool_call("read_records", "{}")], model=model
        )
    )
    XAIOpenAICompatAdapter(model=model, client=client).next_call(request, _deadline())
    return transport.bodies[0]


def _mistral_turn(request: TurnRequest) -> dict[str, Any]:
    model = "mistral-test-20990101"
    transport, client = mistral_transport.scripted_client(
        mistral_transport.chat_body(
            [mistral_transport.tool_call("read_records", "{}")], model=model
        )
    )
    MistralChatAdapter(model=model, client=client).next_call(request, _deadline())
    return transport.bodies[0]


def _flat_tools(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The Responses API's tool shape: name and parameters beside ``type``."""
    return {tool["name"]: tool["parameters"] for tool in body["tools"]}


def _nested_tools(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The Chat Completions tool shape: both nested under ``function``."""
    return {
        tool["function"]["name"]: tool["function"]["parameters"] for tool in body["tools"]
    }


def _responses_case(body: dict[str, Any]) -> dict[str, Any]:
    parsed = json.loads(body["input"][0]["content"])
    assert isinstance(parsed, dict)
    return parsed


def _messages_case(body: dict[str, Any]) -> dict[str, Any]:
    parsed = json.loads(body["messages"][1]["content"])
    assert isinstance(parsed, dict)
    return parsed


#: Each adapter, with the two projections that differ between the surfaces: how
#: its tools are shaped, and where the observed case travels in its body.
Turn = Callable[[TurnRequest], dict[str, Any]]
Tools = Callable[[dict[str, Any]], dict[str, dict[str, Any]]]
Case = Callable[[dict[str, Any]], dict[str, Any]]

ADAPTERS: tuple[tuple[str, Turn, Tools, Case], ...] = (
    ("openai", _openai_turn, _flat_tools, _responses_case),
    ("xai", _xai_turn, _nested_tools, _messages_case),
    ("mistral", _mistral_turn, _nested_tools, _messages_case),
)


@pytest.mark.parametrize(("name", "turn", "tools_of", "case_of"), ADAPTERS)
def test_the_tools_on_the_wire_are_exactly_the_actions_the_request_offers(
    name: str, turn: Turn, tools_of: Tools, case_of: Case
) -> None:
    """Three lists, one set: the request's, the wire's and the model-visible one.

    An extra tool is an action the environment will refuse after the model has
    spent a turn on it; a missing one is an action the run will never observe
    being taken. Either makes the episode measure the harness.
    """
    request = _turn_request()
    body = turn(request)
    tools = tools_of(body)

    assert set(tools) == {action.name for action in request.actions}
    assert sorted(tools) == sorted(request.available_actions)
    assert case_of(body)["available_actions"] == list(request.available_actions)


@pytest.mark.parametrize(("name", "turn", "tools_of", "case_of"), ADAPTERS)
def test_the_fact_argument_is_the_cubes_closed_enum_and_not_an_open_string(
    name: str, turn: Turn, tools_of: Tools, case_of: Case
) -> None:
    """The keys offered are the keys the Cube can deliver, in the same order.

    This is the fact-affordance defect at the vendor boundary: an open string
    graded against an undisclosed set measures the harness rather than the agent,
    and an enum that differed by vendor would mean two models in one matrix were
    shown two different interfaces.
    """
    request = _turn_request()
    offered = next(
        parameter
        for action in request.actions
        if action.name == FACT_ACTION
        for parameter in action.parameters
        if parameter.name == FACT_PARAMETER
    )
    assert offered.enum is not None, "the neutral request must itself carry the enum"

    schema = tools_of(turn(request))[FACT_ACTION]["properties"][FACT_PARAMETER]
    assert schema["type"] == "string"
    assert schema["enum"] == list(offered.enum)


@pytest.mark.parametrize(("name", "turn", "tools_of", "case_of"), ADAPTERS)
def test_no_tool_accepts_an_argument_the_scaffold_never_declared(
    name: str, turn: Turn, tools_of: Tools, case_of: Case
) -> None:
    """``additionalProperties: false`` on every tool, for every vendor.

    The runner refuses a call carrying an undeclared argument, so a schema that
    permitted one would turn a stateable constraint into a recorded protocol
    failure — and it would do so for one vendor's models and not another's.
    """
    request = _turn_request()
    declared = {action.name: action for action in request.actions}
    for action_name, parameters in tools_of(turn(request)).items():
        assert parameters["additionalProperties"] is False
        assert set(parameters["properties"]) == {
            parameter.name for parameter in declared[action_name].parameters
        }
        assert parameters["required"] == [
            parameter.name
            for parameter in declared[action_name].parameters
            if parameter.required
        ]
